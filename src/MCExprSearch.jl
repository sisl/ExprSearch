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

"""
Monte Carlo search by drawing uniform samples from the root of the grammar.
Returns the sample with the best result.  On large datasets earlystop can be an effective
time-saver by not evaluating the fitness function over the entire dataset.
"""
module MC

export MCESParams, MCESResult, mc_search, exprsearch, SearchParams, SearchResult
export PMCESParams

using Reexport
@reexport using DerivationTrees
@reexport using GrammaticalEvolution
@reexport using RLESUtils.Observers
using RLESUtils: GitUtils
using CPUTime
using Iterators

import DerivationTrees.initialize!
import ..ExprSearch: SearchParams, SearchResult, exprsearch, ExprProblem, create_grammar, get_fitness
import Base: isless, copy!

type MCESParams <: SearchParams
  #tree params
  maxsteps::Int64

  #MC
  n_samples::Int64 #samples
  earlystop::Bool #enable early stop
  earlystop_div::Int64 #interval to evaluate early stop criteria

  observer::Observer
end
MCESParams(maxsteps::Int64, n_samples::Int64, observer::Observer) = MCESParams(maxsteps, n_samples, true, 10, observer)

type PMCESParams <: SearchParams
  n_threads::Int64
  mc_params::MCESParams
  observer::Observer
end
PMCESParams(n_threads::Int64, mc_params::MCESParams) = PMCESParams(n_threads, mc_params, Observer())

type MCState
  tree::DerivationTree
  fitness::Float64
  expr
end

MCState(tree::DerivationTree) = MCState(tree, realmax(Float64), 0)

type MCESResult <: SearchResult
  tree::DerivationTree
  fitness::Float64
  expr
  best_at_eval::Int64
  totalevals::Int64

  function MCESResult()
    result = new()
    result.fitness = realmax(Float64)
    result.expr = 0
    result.best_at_eval = 0
    result.totalevals = 0
    return result
  end

  function MCESResult(tree::DerivationTree)
    result = MCESResult()
    result.tree = tree
    return result
  end
end

exprsearch(p::MCESParams, problem::ExprProblem, userargs...) = mc_search(p, problem, userargs...)
exprsearch(p::PMCESParams, problem::ExprProblem, userargs...) = pmc_search(p, problem, userargs...)

function pmc_search(p::PMCESParams, problem::ExprProblem, userargs...)
  @notify_observer(p.observer, "computeinfo", ["starttime", string(now())])
  tic()

  results = pmap(1:p.n_threads) do tid
    mc_search(p.mc_params, problem, userargs...)
  end

  result = minimum(results) #best fitness
  totalevals = sum(map(r -> r.totalevals, results))

  @notify_observer(p.observer, "result", [result.fitness, string(result.expr), totalevals])

  #meta info
  computetime_s = toq()
  @notify_observer(p.observer, "computeinfo", ["computetime_s",  computetime_s])
  @notify_observer(p.observer, "computeinfo", ["endtime",  string(now())])
  @notify_observer(p.observer, "computeinfo", ["hostname", gethostname()])
  @notify_observer(p.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
  @notify_observer(p.observer, "parameters", ["maxsteps", p.mc_params.maxsteps])
  @notify_observer(p.observer, "parameters", ["n_samples", p.mc_params.n_samples])
  @notify_observer(p.observer, "parameters", ["earlystop", p.mc_params.earlystop])
  @notify_observer(p.observer, "parameters", ["earlystop_div", p.mc_params.earlystop_div])
  @notify_observer(p.observer, "parameters", ["n_threads", p.n_threads])

  return result
end

function mc_search(p::MCESParams, problem::ExprProblem, userargs...)
  @notify_observer(p.observer, "verbose1", ["Starting MC search"])
  @notify_observer(p.observer, "computeinfo", ["starttime", string(now())])

  grammar = create_grammar(problem)
  tree_params = DerivTreeParams(grammar, p.maxsteps)

  s = MCState(DerivationTree(tree_params))
  result = MCESResult(DerivationTree(tree_params))

  tstart = CPUtime_us()
  for i = 1:p.n_samples
    @notify_observer(p.observer, "iteration", [i])

    ###############
    #MC algorithm
    sample!(s, problem, result.fitness, realmax(Float64), p.earlystop, p.earlystop_div)
    update!(result, s)
    ###############

    @notify_observer(p.observer, "elapsed_cpu_s", [i, float(CPUtime_us() - tstart) * 1e-6])
    @notify_observer(p.observer, "current_best", [i, result.fitness, string(result.expr)])
  end

  @notify_observer(p.observer, "result", [result.fitness, string(result.expr), result.best_at_eval, result.totalevals])

  #meta info
  @notify_observer(p.observer, "computeinfo", ["endtime",  string(now())])
  @notify_observer(p.observer, "computeinfo", ["hostname", gethostname()])
  @notify_observer(p.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
  @notify_observer(p.observer, "parameters", ["maxsteps", p.maxsteps])
  @notify_observer(p.observer, "parameters", ["n_samples", p.n_samples])
  @notify_observer(p.observer, "parameters", ["earlystop", p.earlystop])
  @notify_observer(p.observer, "parameters", ["earlystop_div", p.earlystop_div])

  return result
end

#initialize to random state
function sample!(s::MCState, problem::ExprProblem, bestfitness::Float64, defaultval::Float64,
                 earlystop::Bool, earlystop_div::Int64, retries::Int64=typemax(Int64))
  rand!(s.tree, retries) #random tree
  s.expr = get_expr(s.tree)
  s.fitness = if earlystop
    get_fitness(problem, s.expr, bestfitness, defaultval, earlystop_div)
  else
    get_fitness(problem, s.expr)
  end
  s
end

#update the global best trackers with the current state
function update!(result::MCESResult, s::MCState)
  result.totalevals += 1 #assumes an eval was called prior

  #update globals
  if s.fitness < result.fitness
    copy!(result.tree, s.tree)
    result.fitness = s.fitness
    result.expr = s.expr
    result.best_at_eval = result.totalevals
  end
end

isless(r1::MCESResult, r2::MCESResult) = r1.fitness < r2.fitness

function copy!(dst::MCState, src::MCState)
  copy!(dst.tree, src.tree)
  dst.fitness = src.fitness
  dst.expr = src.expr
end

end #module
