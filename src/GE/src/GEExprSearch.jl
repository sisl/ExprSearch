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
GrammaticalEvolution
"""
module GE

export GEESParams, GEESResult, ge_search, exprsearch, SearchParams, SearchResult, get_derivtree

using Reexport
using ExprSearch
using RLESUtils, GitUtils, CPUTimeUtils, Observers, LogSystems
import RLESTypes.SymbolTable
using GrammaticalEvolution
@reexport using LinearDerivTrees  #for pretty strings
using CPUTime
using JLD

import ..ExprSearch: SearchParams, SearchResult, exprsearch, ExprProblem, get_grammar, get_fitness,
    get_derivtree, get_expr

include("logdefs.jl")

type GEESParams <: SearchParams
  #GrammaticalEvolution params
  genome_size::Int64
  pop_size::Int64
  maxwraps::Int64
  top_keep::Float64
  top_seed::Float64
  rand_frac::Float64
  prob_mutation::Float64
  mutation_rate::Float64
  default_code::Any
  max_iters::Int64
  logsys::LogSystem
  userargs::SymbolTable
end
GEESParams(genome_size::Int64, pop_size::Int64, maxwraps::Int64, top_keep::Float64,
    top_seed::Float64, rand_frac::Float64, prob_mutation::Float64, mutation_rate::Float64,
    default_code::Any, max_iters::Int64, logsys::LogSystem=logsystem(); 
    userargs::SymbolTable=SymbolTable()) = 
        GEESParams(genome_size, pop_size, maxwraps, top_keep, top_seed, rand_frac, 
        prob_mutation, mutation_rate, default_code, max_iters, logsys, userargs)

type GEESResult <: SearchResult
    tree::LinearDerivTree
    genome::Vector{Int64}
    fitness::Float64
    expr
    best_at_eval::Int64
    totalevals::Int64
end

exprsearch(p::GEESParams, problem::ExprProblem) = ge_search(p, problem::ExprProblem)

get_derivtree(result::GEESResult) = get_derivtree(result.tree)
get_expr(result::GEESResult) = result.expr
get_fitness(result::GEESResult) = result.fitness

function ge_search(p::GEESParams, problem::ExprProblem)
    @notify_observer(p.logsys.observer, "verbose1", ["Starting GE search"])
    @notify_observer(p.logsys.observer, "computeinfo", ["starttime", string(now())])

    grammar = get_grammar(problem)

    tree_params = LDTParams(grammar, p.genome_size)
    tree = LinearDerivTree(tree_params)

    pop = ExamplePopulation(p.pop_size, p.genome_size)
    fitness = realmax(Float64)
    iter = 1
    tstart = CPUtime_start()
    while iter <= p.max_iters
        pop = generate(grammar, pop, p.top_keep, p.top_seed, p.rand_frac, p.prob_mutation, 
            p.mutation_rate, p, problem::ExprProblem, tree)
        fitness = pop[1].fitness #population is sorted, so first entry is the best
        code = pop[1].code
        nevals = iter * p.pop_size
        @notify_observer(p.logsys.observer, "elapsed_cpu_s", [nevals, CPUtime_elapsed_s(tstart)]) 
        @notify_observer(p.logsys.observer, "fitness", Any[iter, fitness])
        @notify_observer(p.logsys.observer, "fitness5", Any[iter, [pop[i].fitness for i=1:5]...])
        @notify_observer(p.logsys.observer, "code", Any[iter, string(code)])
        @notify_observer(p.logsys.observer, "population", Any[iter, pop])
        @notify_observer(p.logsys.observer, "current_best", [nevals, fitness, string(code)])
        iter += 1
    end
    @assert pop.best_ind.fitness == pop.best_fitness <= pop[1].fitness

    fitness = pop.best_fitness
    ind = pop.best_ind
    genome = ind.genome
    expr = ind.code
    best_at_eval = pop.best_at_eval
    totalevals = pop.totalevals

    play!(tree, ind)

    @assert expr == get_expr(tree) "expr=$expr, get_expr(tree)=$(get_expr(tree))"
    
    @notify_observer(p.logsys.observer, "result", [fitness, string(expr), best_at_eval, totalevals])

    #meta info
    @notify_observer(p.logsys.observer, "computeinfo", ["endtime",  string(now())])
    @notify_observer(p.logsys.observer, "computeinfo", ["hostname", gethostname()])
    @notify_observer(p.logsys.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
    @notify_observer(p.logsys.observer, "computeinfo", ["cpu_time", CPUtime_elapsed_s(tstart)]) 
    @notify_observer(p.logsys.observer, "parameters", ["genome_size", p.genome_size])
    @notify_observer(p.logsys.observer, "parameters", ["pop_size", p.pop_size])
    @notify_observer(p.logsys.observer, "parameters", ["maxwraps", p.maxwraps])
    @notify_observer(p.logsys.observer, "parameters", ["top_keep", p.top_keep])
    @notify_observer(p.logsys.observer, "parameters", ["top_seed", p.top_seed])
    @notify_observer(p.logsys.observer, "parameters", ["rand_frac", p.rand_frac])
    @notify_observer(p.logsys.observer, "parameters", ["prob_mutation", p.prob_mutation])
    @notify_observer(p.logsys.observer, "parameters", ["mutation_rate", p.mutation_rate])
    @notify_observer(p.logsys.observer, "parameters", ["default_code", string(p.default_code)])
    @notify_observer(p.logsys.observer, "parameters", ["max_iters", p.max_iters])

    GEESResult(tree, genome, fitness, expr, best_at_eval, totalevals)
end

function GrammaticalEvolution.evaluate!(grammar::Grammar, ind::ExampleIndividual, 
    pop::ExamplePopulation, p::GEESParams, problem::ExprProblem, tree::LinearDerivTree)
    try
        #ind.code = transform(grammar, ind, maxwraps=p.maxwraps)
        play!(tree, ind) 
        ind.code = get_expr(tree)
        ind.fitness = get_fitness(problem, get_derivtree(tree), p.userargs)
        pop.totalevals += 1
        if ind.fitness < pop.best_fitness
            pop.best_fitness = ind.fitness
            pop.best_ind = ind
            pop.best_at_eval = pop.totalevals
        end
    catch e
        #if !isa(e, MaxWrapException)
        if !isa(e, IncompleteException)
            rethrow(e)
        end
        ind.code = p.default_code
        ind.fitness = realmax(Float64)
    end
end

type GEESResultSerial <: SearchResult
    genome::Vector{Int64}
    fitness::Float64
    expr
    best_at_eval::Int64
    totalevals::Int64
end
#don't store the tree to JLD, it's too big and causes stackoverflowerror
function JLD.writeas(r::GEESResult)
    GEESResultSerial(r.genome, r.fitness, r.expr, r.best_at_eval, r.totalevals)
end

end #module
