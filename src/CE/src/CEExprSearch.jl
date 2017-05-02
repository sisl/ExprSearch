# *****************************************************************************
# Written by Ritchie Lee, ritchie.lee@sv.cmu.edu
# *****************************************************************************
# Copyright ã ``2015, United States Government, as represented by the
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
Cross Entropy Method Optimization that uses a probabilistic context-free
grammar as the distribution.  
"""
module CE

export CEESParams, CEESResult, ce_search, exprsearch, SearchParams, SearchResult, get_derivtree
export Samples

import Compat.view
using Reexport
using ExprSearch
using PCFGs
using RLESUtils, GitUtils, CPUTimeUtils, Observers, LogSystems, MemPools, RandChannels
import RLESTypes.SymbolTable
import DerivationTrees.get_children
import ExprSearch: get_derivtree, get_expr
using LinearDerivTrees
@reexport using DerivationTrees  #for pretty strings
using CPUTime
using JLD

using DerivTreeVis

import DerivationTrees: initialize!, max_depth
import ..ExprSearch: SearchParams, SearchResult, exprsearch, ExprProblem, get_grammar, get_fitness

typealias MinDepthByRule Dict{Symbol,Int64}
typealias MinDepthByAction Dict{Symbol,Vector{Int64}}

immutable RuleNotFoundException <: Exception end
immutable DepthExceededException <: Exception end

const DEFAULT_EXPR = :()

include("logdefs.jl")

immutable CEESParams <: SearchParams
    num_samples::Int64  #number of samples
    iterations::Int64     #number of iterations
    elite_frac::Float64 #fraction of samples considered elite
    w_new::Float64      #blend new/old distributions (w_new) * pcfg_new + (1-w_new) * pcfg_old
    w_prior::Float64      #blend with prior pcfg = (1-w_prior) * pcfg + (w_prior) * pcfg_prior
    maxsteps::Int64     #maximum number of steps in derivation 
    default_expr        #if derivation is incomplete, use default expr
    logsys::LogSystem   #to manage logging
    userargs::SymbolTable #passed to get_fitness if specified
end
function CEESParams(num_samples::Int64, iterations::Int64, elite_frac::Float64,
    w_new::Float64, w_prior::Float64, maxsteps::Int64, default_expr, 
    logsys::LogSystem=logsystem(); userargs::SymbolTable=SymbolTable()) 
    CEESParams(num_samples, iterations, elite_frac, w_new, w_prior, maxsteps, default_expr, 
        logsys, userargs)
end

type CEESResult <: SearchResult
    tree::DerivationTree
    fitness::Float64
    expr
    best_at_eval::Int64
    totalevals::Int64
end
CEESResult(grammar::Grammar) = CEESResult(DerivationTree(DerivTreeParams(grammar)), realmax(Float64),
    DEFAULT_EXPR, 0, 0)

exprsearch(p::CEESParams, problem::ExprProblem) = ce_search(p, problem::ExprProblem)

get_derivtree(result::CEESResult) = result.tree
get_expr(result::CEESResult) = result.expr
get_fitness(result::CEESResult) = result.fitness

typealias Sample LinearDerivTree
typealias Samples Vector{Sample}

function ce_search(p::CEESParams, problem::ExprProblem)
    @notify_observer(p.logsys.observer, "verbose1", ["Starting CE search"])
    @notify_observer(p.logsys.observer, "computeinfo", ["starttime", string(now())])

    cfg = get_grammar(problem)
    result = CEESResult(cfg) 
    pcfg = PCFG(cfg) #initialize to uniform probabilities
    pcfg_new = copy(pcfg) #for blending of new and old
    pcfg_prior = copy(pcfg) #uniform prior to ensure full support over domain

    tree_params = LDTParams(cfg, p.maxsteps)
    samples = [LinearDerivTree(tree_params; nodepool=MemPool(DerivTreeNode, 20, 100)) 
        for i=1:p.num_samples] 
    rc = RandChannel(p.num_samples, p.maxsteps)
    wrc_vec = [WrappedRandChannel(rc, 0) for i=1:Threads.nthreads()]
    fitness = realmax(Float64)
    tstart = CPUtime_start()
    iter = 1
    while iter <= p.iterations
        # Draw samples from pcfg
        parallel_rand!(wrc_vec, samples, pcfg)

        # Evaluate fitnesses
        fitnesses = parallel_evaluate(p, samples, result, problem, p.default_expr)

        # Sort in ascending order, lower is better
        order = sortperm(fitnesses)
        
        # Compute elites
        elite_n = round(Int, p.elite_frac * p.num_samples)
        elite_samples = samples[order[1:elite_n]] 

        # Fit elite distribution
        fit_mle!(pcfg_new, elite_samples)

        # Blend old and new distributions 
        # pcfg = (w_new) * pcfg_new + (1-w_new) * pcfg_old
        weighted_sum!(pcfg, 1.0-p.w_new, pcfg_new, p.w_new)

        # Include uniform prior
        # pcfg = (1-w_prior) * pcfg + (w_prior) * pcfg_prior
        weighted_sum!(pcfg, 1.0-p.w_prior, pcfg_prior, p.w_prior)

        @assert result.fitness <= fitnesses[1] #result.fitness should be tracking global minimum

        fitness = get_fitness(result) 
        code = string(get_expr(result))
        nevals = iter * p.num_samples

        @notify_observer(p.logsys.observer, "elapsed_cpu_s", [nevals, CPUtime_elapsed_s(tstart)]) 
        @notify_observer(p.logsys.observer, "fitness", Any[iter, fitness])
        @notify_observer(p.logsys.observer, "code", Any[iter, code])
        @notify_observer(p.logsys.observer, "samples", Any[iter, samples])
        @notify_observer(p.logsys.observer, "current_best", [nevals, fitness, code])

        iter += 1
        resample!(rc)
    end

    #dealloc trees in samples
    for s in samples
        rm_tree!(s.derivtree)
    end

    @notify_observer(p.logsys.observer, "result", [result.fitness, string(result.expr), 
        result.best_at_eval, result.totalevals])

    #meta info
    @notify_observer(p.logsys.observer, "computeinfo", ["endtime",  string(now())])
    @notify_observer(p.logsys.observer, "computeinfo", ["hostname", gethostname()])
    @notify_observer(p.logsys.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
    @notify_observer(p.logsys.observer, "computeinfo", ["cpu_time", CPUtime_elapsed_s(tstart)]) 
    @notify_observer(p.logsys.observer, "parameters", ["num_samples", p.num_samples])
    @notify_observer(p.logsys.observer, "parameters", ["iterations", p.iterations])
    @notify_observer(p.logsys.observer, "parameters", ["elite_frac", p.elite_frac])
    @notify_observer(p.logsys.observer, "parameters", ["w_new", p.w_new])
    @notify_observer(p.logsys.observer, "parameters", ["w_prior", p.w_prior])
    @notify_observer(p.logsys.observer, "parameters", ["maxsteps", p.maxsteps])
    @notify_observer(p.logsys.observer, "parameters", ["default_expr", string(p.default_expr)])

    result 
end

function parallel_rand!{T}(wrc_vec::Vector{WrappedRandChannel{T}}, samples::Vector{LinearDerivTree},
    pcfg::PCFG)
    Threads.@threads for i = 1:length(samples)
        wrc = wrc_vec[Threads.threadid()]
        set_channel!(wrc, i)
        rand!(wrc, samples[i], pcfg)
    end
end

#Sequential evaluation of fitnesses
function seq_evaluate(p::CEESParams, samples::Samples, result::CEESResult, problem::ExprProblem, 
    default_expr)
    fitnesses = Array(Float64, p.num_samples)
    i = 1
    for s in samples
        try
            fitness = get_fitness(problem, s.derivtree, p.userargs)
            fitnesses[i] = fitness
            result.totalevals += 1
            if fitness < result.fitness
                result.fitness = fitness
                copy!(result.tree, s.derivtree)
                result.best_at_eval = result.totalevals
                result.expr = get_expr(s) 
            end
        catch e
            if !isa(e, IncompleteException)
                rethrow(e)
            end
            fitnesses[i] = realmax(Float64)
        end
        i += 1
    end
    fitnesses
end

#threaded evaluation of fitnesses
function parallel_evaluate(p::CEESParams, samples::Samples, result::CEESResult, 
    problem::ExprProblem, default_expr)
    fitnesses = zeros(Float64, p.num_samples)
    #evaluate fitness in parallel
    #for i = 1:p.num_samples #use this to disable threads
    Threads.@threads for i = 1:p.num_samples
        try
            #get_fitness must be thread-safe!
            fitnesses[i] = get_fitness(problem, samples[i].derivtree, p.userargs)
        catch e
            if !isa(e, IncompleteException)
                println("Exception caught! ", e)
                rethrow(e)
            end
            fitnesses[i] = realmax(Float64)
        end
    end
    #update global result
    for i = 1:p.num_samples
        fitness = fitnesses[i]
        result.totalevals += 1
        if fitness < result.fitness
            s = samples[i]
            result.fitness = fitness
            copy!(result.tree, s.derivtree)
            result.best_at_eval = result.totalevals
            result.expr = get_expr(s) 
        end
    end
    fitnesses
end

type CEESResultSerial <: SearchResult
    fitness::Float64
    expr
    best_at_eval::Int64
    totalevals::Int64
end

#don't store the tree to JLD, it's too big and causes stackoverflowerror
function JLD.writeas(r::CEESResult)
    CEESResultSerial(r.fitness, r.expr, r.best_at_eval, r.totalevals)
end

end #module
