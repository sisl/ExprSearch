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
Does not search. Takes an input sequence and replays it, producing emulated logs
"""
module Ref

export RefESParams, RefESResult, mc_search, exprsearch, SearchParams, SearchResult, get_derivtree

using Reexport
using ExprSearch
using RLESUtils, GitUtils, CPUTimeUtils
@reexport using LinearDerivTrees  #pretty_string
using GrammaticalEvolution
using Observers
using Iterators

import LinearDerivTrees: get_derivtree
import ..ExprSearch: SearchParams, SearchResult, exprsearch, ExprProblem, get_grammar, get_fitness
import Base: isless, copy!

type RefESParams <: SearchParams
  #tree params
  maxsteps::Int64

  #Ref
  actions::Vector{Int64}
  n_samples::Int #output is constant, only to emulate logs

  observer::Observer
end
RefESParams(maxsteps::Int64, actions::Vector{Int64}) = RefESParams(maxsteps, actions, Observer())

type RefESResult <: SearchResult
  tree::LinearDerivTree
  actions::Vector{Int64}
  fitness::Float64
  expr
  best_at_eval::Int64
  totalevals::Int64
end

exprsearch(p::RefESParams, problem::ExprProblem, userargs...) = ref_search(p, problem, userargs...)

get_derivtree(result::RefESResult) = get_derivtree(result.tree)

function ref_search(p::RefESParams, problem::ExprProblem, userargs...)
  @notify_observer(p.observer, "verbose1", ["Starting Ref search"])
  @notify_observer(p.observer, "computeinfo", ["starttime", string(now())])

  grammar = get_grammar(problem)
  tree_params = DerivTreeParams(grammar, p.maxsteps)
  tree = DerivationTree(tree_params)

  tstart = CPUtime_start()
  fitness, expr = playsequence!(tree, problem, p.actions) #tree is modified to match expr
  result = RefESResult(tree, p.actions, fitness, expr, 1, p.n_samples)
  elapsed_cpu_s1 = CPUtime_elapsed_s(tstart) #elapsed time for 1 eval in seconds

  #emulate logs...
  for i = 1:p.n_samples
    @notify_observer(p.observer, "iteration", [i])

    @notify_observer(p.observer, "elapsed_cpu_s", [i, i * elapsed_cpu_s1])
    @notify_observer(p.observer, "current_best", [i, result.fitness, string(result.expr)])
  end

  @notify_observer(p.observer, "result", [result.fitness, string(result.expr), result.best_at_eval, result.totalevals])

  #meta info
  @notify_observer(p.observer, "computeinfo", ["endtime",  string(now())])
  @notify_observer(p.observer, "computeinfo", ["hostname", gethostname()])
  @notify_observer(p.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
  @notify_observer(p.observer, "parameters", ["maxsteps", p.maxsteps])
  @notify_observer(p.observer, "parameters", ["actions", string(p.actions)])
  @notify_observer(p.observer, "parameters", ["totalevals", p.n_samples])

  return result
end

#tree is modified
function playsequence!(tree::DerivationTree, problem::ExprProblem, sequence::Vector{Int64})
  play!(tree, sequence)
  expr = get_expr(tree)
  fitness = get_fitness(problem, expr)

  return fitness, expr
end

end #module
