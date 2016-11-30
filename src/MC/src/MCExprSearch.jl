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
Returns the sample with the best result.  
"""
module MC

export MCESParams, MCESResult, mc_search, exprsearch, SearchParams, SearchResult, get_derivtree

using Reexport
using ExprSearch
using RLESUtils, GitUtils, CPUTimeUtils, Observers, LogSystems
@reexport using LinearDerivTrees  #for pretty strings
using GrammaticalEvolution
using Iterators
using JLD

import LinearDerivTrees: initialize!, get_derivtree
import ..ExprSearch: SearchParams, SearchResult, exprsearch, ExprProblem, get_grammar, get_fitness
import Base: isless, copy!

include("logdefs.jl")

type MCESParams <: SearchParams
    #tree params
    maxsteps::Int64

    #MC
    n_samples::Int64 #samples
    logsys::LogSystem
end
MCESParams(maxsteps::Int64, n_samples::Int64) = MCESParams(maxsteps, n_samples, logsystem())

type MCState
    tree::LinearDerivTree
    fitness::Float64
    expr
end

MCState(tree::LinearDerivTree) = MCState(tree, realmax(Float64), 0)

type MCESResult <: SearchResult
    tree::LinearDerivTree
    actions::Vector{Int64}
    fitness::Float64
    expr
    best_at_eval::Int64
    totalevals::Int64
    
    function MCESResult()
        result = new()
        result.actions = Int64[]
        result.fitness = realmax(Float64)
        result.expr = 0
        result.best_at_eval = 0
        result.totalevals = 0
        result
    end

    function MCESResult(tree::LinearDerivTree)
        result = MCESResult()
        result.tree = tree
        result
    end
end

exprsearch(p::MCESParams, problem::ExprProblem, userargs...) = mc_search(p, problem, userargs...)

get_derivtree(result::MCESResult) = get_derivtree(result.tree)

function mc_search(p::MCESParams, problem::ExprProblem, userargs...)
    @notify_observer(p.logsys.observer, "verbose1", ["Starting MC search"])
    @notify_observer(p.logsys.observer, "computeinfo", ["starttime", string(now())])
    
    grammar = get_grammar(problem)
    tree_params = LDTParams(grammar, p.maxsteps)

    s = MCState(LinearDerivTree(tree_params))
    result = MCESResult(LinearDerivTree(tree_params))

    tstart = CPUtime_start()
    for i = 1:p.n_samples
        @notify_observer(p.logsys.observer, "iteration", [i])
        ###############
        #MC algorithm
        sample!(s, problem)
        update!(result, s)
        ###############

        @notify_observer(p.logsys.observer, "elapsed_cpu_s", [i, CPUtime_elapsed_s(tstart)])
        @notify_observer(p.logsys.observer, "current_best", [i, result.fitness, string(result.expr)])
    end
    @notify_observer(p.logsys.observer, "result", [result.fitness, string(result.expr), 
        result.best_at_eval, result.totalevals])

    #meta info
    @notify_observer(p.logsys.observer, "computeinfo", ["endtime",  string(now())])
    @notify_observer(p.logsys.observer, "computeinfo", ["hostname", gethostname()])
    @notify_observer(p.logsys.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
    @notify_observer(p.logsys.observer, "parameters", ["maxsteps", p.maxsteps])
    @notify_observer(p.logsys.observer, "parameters", ["n_samples", p.n_samples])

    result
end

#initialize to random state
function sample!(s::MCState, problem::ExprProblem, retries::Int64=typemax(Int64))
    rand_with_retry!(s.tree, retries) #sample uniformly
    s.expr = get_expr(s.tree)
    s.fitness = get_fitness(problem, s.expr)
    s
end

#update the global best trackers with the current state
function update!(result::MCESResult, s::MCState)
    result.totalevals += 1 #assumes an eval was called prior

    #update globals
    if s.fitness < result.fitness
        copy!(result.tree, s.tree)
        resize!(result.actions, length(s.tree.actions))
        result.actions = convert(Array, s.tree.actions)
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

type MCESResultSerial <: SearchResult
    actions::Vector{Int64}
    fitness::Float64
    expr
    best_at_eval::Int64
    totalevals::Int64
end
#don't store the tree to JLD, it's too big and causes stackoverflowerror
function JLD.writeas(r::MCESResult)
    MCESResultSerial(r.actions, r.fitness, r.expr, r.best_at_eval, r.totalevals)
end

end #module
