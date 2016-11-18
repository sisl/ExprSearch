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
Tree-based grammar guided genetic programming that uses a context-free
grammar in BNF.
"""
module GP

export GPESParams, GPESResult, gp_search, exprsearch, SearchParams, SearchResult, get_derivtree

using Reexport
using ExprSearch
using RLESUtils, GitUtils, CPUTimeUtils, Observers
using GrammaticalEvolution
@reexport using DerivationTrees  #for pretty strings
using CPUTime
using JLD

import ..ExprSearch: SearchParams, SearchResult, exprsearch, ExprProblem, get_grammar, get_fitness
import Base: copy!, sort!, isless

immutable CrossoverException <: Exception end
immutable DepthExceededException <: Exception end

immutable GPESParams
    pop_size::Int64
    maxdepth::Int64
    iterations::Int64
    tournament_size::Int64
    top_keep::Float64
    crossover_frac::Float64
    mutate_frac::Float64
    rand_frac::Float64
    default_code
    observer::Observer
end

type GPESResult <: SearchResult
  tree::DerivationTree
  fitness::Float64
  expr
  best_at_eval::Int64
  totalevals::Int64
end

exprsearch(p::GPESParams, problem::ExprProblem, userargs...) = gp_search(p, problem::ExprProblem, userargs...)

get_derivtree(result::GPESResult) = get_derivtree(result.tree)

type GPIndividual
    derivtree::DerivationTree
    expr
    fitness::Nullable{Float64}
end
GPIndividual() = GPIndividual(DerivationTree(), :(false), Nullable{Float64}())

type GPPopulation
    individuals::Vector{GPIndividual}
    best_at_eval::Int64
    totalevals::Int64
end
GPPopulation() = GPPopulation(Array(GPIndividual, 0), 0, 0)

function gp_search(p::GPESParams, problem::ExprProblem, userargs...)
  @notify_observer(p.observer, "verbose1", ["Starting GP search"])
  @notify_observer(p.observer, "computeinfo", ["starttime", string(now())])

  grammar = get_grammar(problem)
  mindepthmap = mindepth(grammar)

  pop = GPPopulation()
  initialize!(pop, p.pop_size, p.maxdepth)
  fitness = realmax(Float64)
  iter = 1
  tstart = CPUtime_start()
  while iter <= p.max_iters
    pop = generate(p, grammar, pop, problem, userargs...)
    ind = best_ind(pop)
    fitness = ind.fitness 
    expr = ind.expr 
    nevals = iter * p.pop_size
    @notify_observer(p.observer, "elapsed_cpu_s", [nevals, CPUtime_elapsed_s(tstart)]) 
    @notify_observer(p.observer, "code", Any[iter, string(code)])
    @notify_observer(p.observer, "population", Any[iter, pop])
    @notify_observer(p.observer, "current_best", [nevals, fitness, code])
    iter += 1
  end
  @assert pop.best_ind.fitness == pop.best_fitness <= pop[1].fitness

  best_at_eval = pop.best_at_eval
  totalevals = pop.totalevals

  return GPESResult(tree, fitness, expr, best_at_eval, totalevals)
end

function isless(x::GPIndividual, y::GPIndividual)
   get(x.fitness, realmax(eltype(x.fitness))) < get(y.fitness, realmax(eltype(y.fitness)))
end

function best_ind(pop::GPPopulation) 
    @assert !isnull(pop.individuals[1].fitness)
    @assert pop.individuals[1] <= pop.individuals[end] 
    pop.individuals[1]
end

"""
Compute minimum depth for each rule
"""
function mindepth(grammar::Grammar)::Dict{Symbol,Int64}
    d = Dict{Symbol,Int64}()
    
    d
end

"""
Initial population
"""
function  initialize!(pop::GPPopulation, pop_size::Int64, maxdepth::Int64)
    resize!(pop, pop_size)

    #

    pop
end

"""
Evaluate fitness function
"""
function evaluate!(pop::GPPopulation, userargs...)
    n = 0 #number of fitness calls
    for ind in pop
        if isnull(ind.fitness)
            try
                ind.expr = get_expr(ind.tree)
                fitness = get_fitness(problem, ind.expr, userargs...)
                ind.fitness = Nullable{Float64}(fitness) 
                pop.totalevals += 1
                if fitness < pop.best_fitness
                    pop.best_fitness = fitness
                    pop.best_ind = ind
                    pop.best_at_eval = pop.totalevals
                end
            catch e
                if !isa(e, IncompleteException)
                    println("Exception in GP evaluate!: ", e)
                    error()
                end
                ind.expr = p.default_code
                ind.fitness = realmax(ind.fitness)
            end
        end
    end
    n
end

"""
Tournament selection
Best fitness wins deterministically
"""
function selection(params::GPESParams, pop::GPPopulation) 
    N = params.tournament_size
    #randomly choose N inds and compare their fitness  
    ids = rand(1:length(pop), N)
    pool = pop[ids] 
    #return the ind with min fitness
    (fitness, id) = findmin(map(ind->ind.fitness, pool))
    pool[id]
end

"""
Crossover
"""
function crossover(ind1::GPIndividual, ind2::GPIndividual)
    ind3 = GPIndividual()
    ind4 = GPIndividual()

    #choose crossover point on ind1

    (ind3, ind4)
end #module


"""
Mutation
"""
function mutate!(ind::GPIndividual)
    #choose random node

end

"""
Adjust to target pop_size
"""
function adjust_to_size!(newpop::GPPopulation, pop::GPPopulation, pop_size::Int64)
    #reproduce if undersize
    while length(newpop) < pop_size
        ind1 = selection(pop)
        push!(newpop, ind1)
    end
    
    #trim to size
    resize!(newpop, pop_size)
    newpop
end

function copy!(dst::GPPopulation, doffset::Int64, src::GPPopulation, soffset::Int64, N::Int64)
    copy!(dst.individuals, doffset, src, src.individuals, N)
end

function sort!(pop::GPPopulation) #lowest (best) fitness first
    sort!(pop.individuals, by=x->x.fitness)
end

"""
Create next generation
"""
function generate(p::GPESParams, grammar::Grammar, pop::GPPopulation, userargs...)
    n_keep = floor(Int64, p.top_keep*p.pop_size)
    n_crossover = floor(Int64, p.crossover_frac*p.pop_size)
    n_mutate = floor(Int64, p.mutate_frac*p.pop_size)
    n_rand = floor(Int64, p.rand_frac*p.pop_size)

    evaluate!(pop, userargs...)
    sort!(pop)
    newpop = GPPopulation()

    #pass through the top ones
    copy!(newpop, 1, pop, 1, n_keep)
    
    #crossover
    n = 0
    while n < n_crossover
        ind1 = selection(pop)
        ind2 = selection(pop) 
        try
            (ind3, ind4) = crossover(ind1, ind2)
            push!(newpop, ind3)
            push!(newpop, ind4)
            n += 2
        catch e
            if !isa(e, CrossoverException)
                rethrow(e)
            end
        end
    end

    #mutate
    n = 0
    while n < n_mutate
        ind1 = selection(pop)
        ind2 = mutate(ind1)
        push!(newpop, ind2)
        n += 1
    end

    #random
    n = 0
    while n < n_rand 
        try
            ind1 = rand(GPIndividual)
            push!(newpop, ind1)
            n += 1
        catch e
            if !isa(e, DepthExceededException)
                rethrow(e)
            end
        end
    end

    #adjust size: trim or reproduce to size
    adjust_to_size!(newpop::GPPopulation, pop::GPPopulation, pop_size::Int64)

    evaluate!(newpop, userargs...)
    sort!(pop)

    newpop
end

end #module
