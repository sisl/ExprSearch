using POMDPs
#using RLESUtils, Observers
import Base: getindex, setindex!, haskey

# State node in the search tree
type StateNode
  n::Array{Int64,1} # number of visits at the node for each action
  Q::Array{Reward,1} # estimated value for each action
  action_map::Vector{Action} # for converting action idxs to action types
  nextstates::Vector{Nullable{UInt64}} #deterministic state transition means 1-1 mapping between actions and sp's
  explored::Bool #true if subtree is fully explored
end
function StateNode(nA, space::AbstractSpace, q0::Float64=0.0, explored::Bool=false)
  n = zeros(Int32, nA)
  Q = fill(q0, nA) #zeros(Reward, nA)
  # creates the action map
  am = Action[]
  for a in iterator(space)
    push!(am, a)
  end
  nextstates = [Nullable{UInt64}() for i = 1:nA]
  StateNode(n, Q, am, nextstates, explored)
end

# MCTS solver type
type MCTSSolver <: POMDPs.Solver
  n_iterations::Int64	# number of iterations during each action() call
  depth::Int64 # the max depth of the tree
  exploration_constant::Float64 # constant balancing exploration and exploitation
  maxmod::Bool
  rng::AbstractRNG # random number generator
  tree::Dict{UInt64, StateNode} # the search tree
end
# solver constructor
function MCTSSolver(;n_iterations::Int64=100,
                    depth::Int64=10,
                    exploration_constant::Float64=1.0,
                    maxmod::Bool=false, #use the max modification when updating q
                    rng = MersenneTwister(1))
  tree = Dict{UInt64, StateNode}() #key is hash(State)
  return MCTSSolver(n_iterations, depth, exploration_constant, maxmod, rng, tree)
end

# MCTS policy type
type MCTSPolicy <: POMDPs.Policy
  mcts::MCTSSolver # containts the solver parameters
  mdp::POMDP # model
  action_space::AbstractSpace # pre-allocated for rollout
  state::State # pre-allocated for sampling
  action::Action # pre-allocated for sampling
  distribution::AbstractDistribution # pre-allocated for memory efficiency
  rollout_policy::Policy # used in the rollout evaluation
  q0::Float64
  best_reward::Reward
  best_state::State
  best_at_eval::Int64
  totalevals::Int64
end
# policy constructor
function MCTSPolicy(mcts::MCTSSolver, mdp::POMDP,
                    rollout_policy=RandomPolicy(mdp, mcts.rng); # random policy is default
                    q0::Float64=0.0,
                    best_at_eval::Int64=0,
                    totalevals::Int64=0)
  # pre-allocate action space
  as = actions(mdp)
  # pre-allocate the state distrbution
  d = create_transition_distribution(mdp)
  s = create_state(mdp)
  a = create_action(mdp)
  best_reward = q0
  best_state = create_state(mdp)
  return MCTSPolicy(mcts, mdp, as, s, a, d, rollout_policy, q0, best_reward, best_state, best_at_eval, totalevals)
end

# for convenience - no computation is done in solve
function POMDPs.solve(solver::MCTSSolver, mdp::POMDP, policy::MCTSPolicy)
  policy
end

# retuns an approximately optimal action
function POMDPs.action(policy::MCTSPolicy, state::State)
  n_iterations = policy.mcts.n_iterations
  depth = policy.mcts.depth
  tree = policy.mcts.tree
  # build the tree
  for n = 1:n_iterations
    isexplored(tree, state) && break #can't do better, exit early

    #@notify_observer(policy.observer, "iteration", [n])
    simulate(policy, state, depth)
  end
  snode = tree[state]
  # find the index of action with highest q val
  i = indmax(snode.Q)
  #@notify_observer(policy.observer, "Qs", [snode.Q])
  #@notify_observer(policy.observer, "best_i", [i])
  #@notify_observer(policy.observer, "tree", [tree, state])
  # use map to conver index to mdp action
  return snode.action_map[i]
end

# runs a simulation from the passed in state to the specified depth
function POMDPs.simulate(policy::MCTSPolicy, state::State, depth::Int64)
  # model parameters
  mdp = policy.mdp
  na = n_actions(mdp, state)
  discount_factor = discount(mdp)
  sp = policy.state
  rng = policy.mcts.rng

  # solver parameters
  n_iterations = policy.mcts.n_iterations
  tree = policy.mcts.tree
  exploration_constant = policy.mcts.exploration_constant

  # if unexplored state add to the tree
  isnew = false
  if !haskey(tree, state)
    as = actions(mdp, state, policy.action_space)
    tree[state] = StateNode(na, as, policy.q0)
    isnew = true
  end

  snode = tree[state]

  if isterminal(mdp, state) #terminal states don't have corresponding snodes?
    snode.explored = true
    r = reward(mdp, state)
    policy.totalevals += 1 #expr eval call
    #@notify_observer(policy.observer, "terminal_reward", [r, state])
    if r > policy.best_reward
      policy.best_reward = r
      copy!(policy.best_state, state)
      policy.best_at_eval = policy.totalevals
    end
    return r
  end

  if isnew #new but not terminal
    return rollout(policy, depth, state)
  end

  # don't choose fully-explored subtrees
  best_i = -1
  best_val = -realmax(Float64)
  for i = 1:na
    if !isexplored(tree, snode.nextstates[i])
      #selection criteria
      val =  snode.Q[i] + exploration_constant * real(sqrt(complex(log(sum(snode.n)) / snode.n[i])))

      if best_i < 0 || val > best_val #accept the first unexplored one or accept if multiple ties for best
        best_val = val
        best_i = i
      end
    end
  end
  i = best_i
  a = snode.action_map[i]
  r = reward(mdp, state, a)
  # transition to a new state
  d = policy.distribution
  d = transition(mdp, state, a, d)
  sp = rand!(rng, sp, d)

  #add this transition to snode
  if !isnull(snode.nextstates[i])
    @assert get(snode.nextstates[i]) == hash(sp)
  else
    snode.nextstates[i] = Nullable{UInt64}(hash(sp))
  end

  # update the Q and n values
  q = r + discount_factor * simulate(policy, sp, depth - 1)
  snode.n[i] += 1 # increase number of node visits by one

  if policy.mcts.maxmod #maxmod
    snode.Q[i] = max(snode.Q[i], q)
  else
    snode.Q[i] += ((q - snode.Q[i]) / (snode.n[i])) # moving average of Q value
  end

  #if all children are fully-explored, mark this node as fully-explored
  snode.explored = isexplored(tree, snode.nextstates)

  return q
end

# recursive rollout to specified depth, returns the accumulated discounted reward
function rollout(policy::MCTSPolicy, depth::Int64, state::State)
  mdp = policy.mdp
  # finish when depth is zero or reach terminal state
  if isterminal(mdp, state)
    r = reward(mdp, state)
    policy.totalevals += 1 #expr eval call
    if r > policy.best_reward
      policy.best_reward = r
      copy!(policy.best_state, state)
      policy.best_at_eval = policy.totalevals
    end
    return r
  end
  d = policy.distribution
  discount_factor = discount(mdp)
  rng = policy.mcts.rng
  sp = policy.state
  # follow the rollout policy
  a = action(policy.rollout_policy, state)
  # compute reward
  r = reward(mdp, state, a)
  # sample the next state
  d = transition(mdp, state, a, d)
  sp = rand!(rng, sp, d)
  return r + discount_factor * rollout(policy, depth - 1, sp)
end

#override to use the hash instead
Base.getindex{V}(D::Dict{UInt64,V}, key::State) = D[hash(key)]
Base.getindex{V}(D::Dict{UInt64,V}, key::Nullable{UInt64}) = D[get(key)]
Base.setindex!{V}(D::Dict{UInt64,V}, value::V, key::State) = D[hash(key)] = value
Base.setindex!{V}(D::Dict{UInt64,V}, value::V, key::Nullable{UInt64}) = D[get(key)] = value
Base.haskey{V}(D::Dict{UInt64,V}, key::State) = haskey(D, hash(key))
Base.haskey{V}(D::Dict{UInt64,V}, key::Nullable{UInt64}) = haskey(D, get(key))

isexplored(tree::Dict{UInt64,StateNode}, s::State) = haskey(tree, s) && tree[s].explored
isexplored(tree::Dict{UInt64,StateNode}, h::Nullable{UInt64}) = !isnull(h) && tree[get(h)].explored

function isexplored(tree::Dict{UInt64,StateNode}, nextstates::Vector{Nullable{UInt64}})
  explored = true
  for h in nextstates
    if !isexplored(tree, h)
      explored = false
      break
    end
  end
  return explored
end
