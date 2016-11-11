# *****************************************************************************
# Written by Ritchie Lee, ritchie.lee@sv.cmu.edu
# *****************************************************************************
# Copyright ã 2015, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration. All
# rights reserved.  The Reinforcement Learning Encounter Simulator (RLES)
# platform is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You
# may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable
# law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
# _____________________________________________________________________________
# Reinforcement Learning Encounter Simulator (RLES) includes the following
# third party software. The SISLES.jl package is licensed under the MIT Expat
# License: Copyright (c) 2014: Youngjun Kim.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
# "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# *****************************************************************************

module DerivTreeMDPs

using ExprSearch, DerivationTrees
using Reexport
using RLESUtils, Observers
using POMDPs
using GrammaticalEvolution
using Iterators

export DerivTreeMDPParams, DerivTreeState, DerivTreeAction, DerivTreeMDP, DerivTreeStateSpace, DerivTreeActionSpace
export DerivTreeTransitionDistr, discount, n_actions, actions, iterator, reward, sync!, step!, get_fitness, get_expr
export create_state, create_action, create_transition_distribution

import DerivationTrees: step!, isterminal, get_expr
import Base: ==, hash, rand!, copy!, push!
import ExprSearch: ExprProblem, get_fitness

type DerivTreeMDPParams
  grammar::Grammar
  max_neg_reward::Float64
  step_reward::Float64
  discount::Float64
  observer::Observer
end

function DerivTreeMDPParams(grammar::Grammar, max_neg_reward::Float64, step_reward::Float64,
                            discount::Float64=1.0; observer::Observer=Observer())
  return DerivTreeMDPParams(grammar, max_neg_reward, step_reward, discount, observer)
end

type DerivTreeAction <: Action
  action_id::Int64
end
DerivTreeAction() = DerivTreeAction(-1)

type DerivTreeMDP <: POMDP
  params::DerivTreeMDPParams
  tree::DerivationTree #true state of the sim
  problem::ExprProblem #used in calling get_fitness
  statehash::UInt64 #hash for current state for sync'ing purposes
  all_actions::Vector{DerivTreeAction}
  userargs::Vector{Any}

  function DerivTreeMDP{T}(p::DerivTreeMDPParams, tree::DerivationTree, problem::ExprProblem, all_actions::Vector{DerivTreeAction}, userargs::Vector{T})
    return new(p, tree, zero(UInt64), problem, all_actions, userargs)
  end

  function DerivTreeMDP(p::DerivTreeMDPParams, tree::DerivationTree, problem::ExprProblem, userargs...)
    mdp = new()
    mdp.params = p
    mdp.tree = tree
    mdp.problem = problem
    mdp.statehash = zero(UInt64)
    mdp.all_actions = generate_all_actions(p.grammar)
    mdp.userargs = [userargs...]
    return mdp
  end
end

#TODO: put mdp or tree inside state so that sync! only depends on s, i.e., s fully contains state
type DerivTreeState <: State
  mdp::DerivTreeMDP #sneak this in to have fully contained state
  past_actions::Vector{Int64} #actions taken since initialize!
end
DerivTreeState(mdp::DerivTreeMDP) = DerivTreeState(mdp, Int64[])

type DerivTreeStateSpace <: AbstractSpace
  #mdp::DerivTreeMDP
  action_id::Int64
end

type DerivTreeActionSpace <: AbstractSpace
  mdp::DerivTreeMDP
  action_ids::Vector{Int64}
end

type DerivTreeTransitionDistr <: AbstractDistribution
  nextstate::DerivTreeState
  DerivTreeTransitionDistr(mdp) = new(DerivTreeState(mdp))
  DerivTreeTransitionDistr(state::DerivTreeState) = new(state)
end

function generate_all_actions(grammar::Grammar)
  return [DerivTreeAction(i) for i = 1:maxlength(grammar)]
end

# returns the discount factor
POMDPs.discount(mdp::DerivTreeMDP) = mdp.params.discount

# returns the number of actions in the problem
function POMDPs.n_actions(mdp::DerivTreeMDP, s::DerivTreeState)
  p = mdp.params
  @notify_observer(p.observer, "debug2", ["n_actions called"])
  sync!(s)
  return length(actionspace(s.mdp.tree))
end

# returns the action space for the entire problem
POMDPs.actions(mdp::DerivTreeMDP) = DerivTreeActionSpace(mdp, collect(1:length(mdp.all_actions)))

# fills the action space with the actions availiable from state s
function POMDPs.actions(mdp::DerivTreeMDP, s::DerivTreeState, as::DerivTreeActionSpace)
  p = mdp.params
  @notify_observer(p.observer, "debug2", ["actions called"])
  sync!(s)
  as.action_ids = collect(actionspace(s.mdp.tree)) #reachable actions
  return as
end

function POMDPs.iterator(action_space::DerivTreeActionSpace)
  p = action_space.mdp.params
  @notify_observer(p.observer, "debug2", ["iterator called"])
  all_actions = action_space.mdp.all_actions
  ids = action_space.action_ids
  return imap(id -> all_actions[id], ids) #iterator avoids allocation
end

# fills d with neighboring states reachable from the s,a pair
function POMDPs.transition(mdp::DerivTreeMDP, s::DerivTreeState, a::DerivTreeAction, d::DerivTreeTransitionDistr)
  p = s.mdp.params
  @notify_observer(p.observer, "debug2", ["transition called"])
  step!(s, d.nextstate, a)
  return d
end

# fills sp with random sample from distribution d
function POMDPs.rand!(rng::AbstractRNG, sp::DerivTreeState, d::DerivTreeTransitionDistr)
  return copy!(sp, d.nextstate)
end

function step!(s::DerivTreeState, sp::DerivTreeState, a::DerivTreeAction)
  p = s.mdp.params
  @notify_observer(p.observer, "debug1", ["mdp step! called"])
  sync!(s)
  step!(s.mdp.tree, a.action_id) #deterministic transition
  copy!(sp, s)
  push!(sp, a)
  s.mdp.statehash = hash(sp) #update sync obj
  return sp
end

get_fitness() = error("fitness not defined")

# returns the immediate reward of being in state s and performing action a
POMDPs.reward(mdp::DerivTreeMDP, s::DerivTreeState, a::DerivTreeAction) = POMDPs.reward(mdp, s)

function POMDPs.reward(mdp::DerivTreeMDP, s::DerivTreeState)
  p = mdp.params
  @notify_observer(p.observer, "debug2", ["reward called"])
  sync!(s)

  tree = mdp.tree
  reward = if iscomplete(tree)
    expr = get_expr(tree)
    -get_fitness(mdp.problem, expr, mdp.userargs...)
  elseif isterminal(tree) #not-compilable
    p.max_neg_reward
  else #each step
    p.step_reward
  end
  return reward
end

# returns a boolean indicating if state s is terminal
function POMDPs.isterminal(mdp::DerivTreeMDP, s::DerivTreeState)
  p = mdp.params
  @notify_observer(p.observer, "debug2", ["isterminal called"])
  sync!(s)
  return isterminal(s.mdp.tree)
end

# initializes a model state
POMDPs.create_state(mdp::DerivTreeMDP) = DerivTreeState(mdp)

# initializes a model action
POMDPs.create_action(mdp::DerivTreeMDP) = DerivTreeAction()

# initializes a distirbution over states
POMDPs.create_transition_distribution(mdp::DerivTreeMDP) = DerivTreeTransitionDistr(mdp)

# if you want to use a random policy you need to implement an action space sampling function
# fills action a with a random action form action_space
function POMDPs.rand!(rng::AbstractRNG, a::DerivTreeAction, action_space::DerivTreeActionSpace)
  a.action_id = rand(rng, action_space.action_ids)
  return a
end

#equate all fields
function ==(s1::DerivTreeState, s2::DerivTreeState)
  s1.past_actions == s2.past_actions
end

#hash all fields
function hash(s::DerivTreeState)
  h = hash(DerivTreeState)
  h = hash(s.mdp, h)
  for a in s.past_actions
    h = hash(a, h)
  end
  return h
end

function sync!(s::DerivTreeState)
  p = s.mdp.params
  @notify_observer(p.observer, "debug2", ["sync! called current=$(s.mdp.statehash), s=$(s.past_actions)"])
  #if sync'd, we're done
  s.mdp.statehash == hash(s) && return
  @notify_observer(p.observer, "debug1", ["sync'ing"])
  #resync
  initialize!(s.mdp.tree)
  for a in s.past_actions
    step!(s.mdp.tree, a)
  end
  s.mdp.statehash = hash(s)
end

reset!(s::DerivTreeState) = empty!(s.past_actions)

function copy!(dst::DerivTreeState, src::DerivTreeState)
  dst.mdp = src.mdp
  resize!(dst.past_actions, length(src.past_actions))
  copy!(dst.past_actions, src.past_actions)
  return dst
end

push!(s::DerivTreeState, a::DerivTreeAction) = push!(s.past_actions, a.action_id)

function get_expr(state::DerivTreeState)
  sync!(state) #sync to state
  return get_expr(state.mdp.tree)
end


end #module
