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

module MCTS2  #ExprSearch.MCTS2

export MCTS2ESParams, MCTS2ESResult, mcts_search, exprsearch, get_reward, SearchParams, SearchResult

include("DerivTreeMDPs.jl")

using Reexport
@reexport using DerivationTrees
@reexport using .DerivTreeMDPs
@reexport using GrammaticalEvolution
@reexport using RLESUtils.Observers
@reexport using GBMCTS
using RLESUtils.GitUtils
using CPUTime

import .DerivTreeMDPs.get_reward
import ..ExprSearch: SearchParams, SearchResult, exprsearch

type MCTS2ESParams <: SearchParams
  #tree params
  tree_params::DerivTreeParams

  #mdp params
  mdp_params::DerivTreeMDPParams

  #mcts iters
  n_iters::Int64
  searchdepth::Int64
  exploration_const::Float64
  q0::Float64
  mcts_observer::Observer

  observer::Observer
end

type MCTS2ESResult <: SearchResult
  tree::DerivationTree
  actions::Vector{Int64}
  reward::Float64
  expr::Union{Symbol,Expr}
  best_at_eval::Int64
  totalevals::Int64
end

exprsearch(p::MCTS2ESParams, userargs...) = mcts2_search(p, userargs...)

function mcts2_search(p::MCTS2ESParams, userargs...)
  @notify_observer(p.observer, "verbose1", ["Starting MCTS2 search"])
  @notify_observer(p.observer, "computeinfo", ["starttime", string(now())])

  tree = DerivTreeParams(p.tree_params) |> DerivationTree
  mdp = DerivTreeMDP(p.mdp_params, tree, userargs...)

  solver = MCTSSolver(n_iterations=p.n_iters, depth=p.searchdepth, exploration_constant=p.exploration_const)
  policy = MCTSPolicy(solver, mdp, observer=p.mcts_observer, q0=p.q0)

  initialize!(tree)
  s = create_state(mdp)

  i = 1
  while !GBMCTS.isexplored(policy.mcts.tree, s) && i < p.n_iters
    @notify_observer(policy.observer, "iteration", [n])
    CPUtic()
    simulate(policy, s, p.searchdepth)
    @notify_observer(p.observer, "cputime", [i, CPUtoq()])
    @notify_observer(p.observer, "current_best", [i, policy.best_reward, policy.best_state])
    @notify_observer(p.observer, "tree", [i, policy.mcts.tree])
    i += 1
  end
  best_reward = policy.best_reward

  sync!(mdp, policy.best_state) #go to best state
  expr = get_expr(tree)
  @notify_observer(p.observer, "result", [best_reward, string(expr), policy.best_at_eval, policy.totalevals])

  #meta info
  @notify_observer(p.observer, "computeinfo", ["endtime",  string(now())])
  @notify_observer(p.observer, "computeinfo", ["hostname", gethostname()])
  @notify_observer(p.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
  @notify_observer(p.observer, "parameters", ["maxsteps", p.tree_params.maxsteps])
  @notify_observer(p.observer, "parameters", ["discount", p.mdp_params.discount])
  @notify_observer(p.observer, "parameters", ["n_iters", p.n_iters])
  @notify_observer(p.observer, "parameters", ["searchdepth", p.searchdepth])
  @notify_observer(p.observer, "parameters", ["exploration_const", p.exploration_const])
  @notify_observer(p.observer, "parameters", ["q0", p.q0])

  return MCTS2ESResult(tree, s.past_actions, best_reward, expr, policy.best_at_eval, policy.totalevals)
end

end #module
