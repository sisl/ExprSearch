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

export MCTSESParams, MCTSESResult, mcts_search

@reexport using GBMCTS
using RLESUtils.GitUtils
using CPUTime

type MCTSESParams <: SearchParams
  #tree params
  tree_params::DerivTreeParams

  #mdp params
  mdp_params::DerivTreeMDPParams

  #mcts iters
  n_iters::Int64
  searchdepth::Int64
  exploration_const::Float64
  mcts_observer::Observer

  safetylimit::Int64

  observer::Observer
end

type MCTSESResult <: SearchResult
  tree::DerivationTree
  actions::Vector{Int64}
  reward::Float64
  expr::Union{Symbol,Expr}
end

exprsearch(p::MCTSESParams) = mcts_search(p)

function mcts_search(p::MCTSESParams)
  @notify_observer(p.observer, "verbose1", ["Starting MCTS search"])
  @notify_observer(p.observer, "computeinfo", ["starttime", string(now())])

  tree = DerivTreeParams(p.tree_params) |> DerivationTree
  mdp = DerivTreeMDP(p.mdp_params, tree)

  solver = MCTSSolver(n_iterations=p.n_iters, depth=p.searchdepth, exploration_constant=p.exploration_const)
  policy = MCTSPolicy(solver, mdp, observer=p.mcts_observer)

  initialize!(tree)
  s = create_state(mdp)
  sp = create_state(mdp)

  i = 1
  while !isterminal(tree) && i < p.safetylimit
    CPUtic()
    @notify_observer(p.observer, "step", [i])
    a = action(policy, s)
    @notify_observer(p.observer, "action", [i, a.action_id])
    step!(mdp, s, sp, a)
    copy!(s, sp)
    @notify_observer(p.observer, "cputime", [i, CPUtoq()])
    i += 1
  end
  total_reward = get_reward(tree)
  expr = get_expr(tree)
  @notify_observer(p.observer, "result", [total_reward, string(expr)])

  #meta info
  @notify_observer(p.observer, "computeinfo", ["endtime",  string(now())])
  @notify_observer(p.observer, "computeinfo", ["hostname", gethostname()])
  @notify_observer(p.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
  @notify_observer(p.observer, "parameters", ["maxsteps", p.tree_params.maxsteps])
  @notify_observer(p.observer, "parameters", ["discount", p.mdp_params.discount])
  @notify_observer(p.observer, "parameters", ["n_iters", p.n_iters])
  @notify_observer(p.observer, "parameters", ["searchdepth", p.searchdepth])
  @notify_observer(p.observer, "parameters", ["exploration_const", p.exploration_const])

  return MCTSESResult(tree, s.past_actions, total_reward, expr)
end
