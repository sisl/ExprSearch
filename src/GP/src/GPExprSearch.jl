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
export GPPopulation, GPIndividual
export MinDepthByRule, MinDepthByAction, min_depth_rule, min_depth_actions

using Reexport
using ExprSearch
using RLESUtils, GitUtils, CPUTimeUtils, RandUtils, Observers
using GrammaticalEvolution
@reexport using DerivationTrees  #for pretty strings
using CPUTime
using JLD

import DerivationTrees: initialize!
import ..ExprSearch: SearchParams, SearchResult, exprsearch, ExprProblem, get_grammar, get_fitness
import Base: copy!, sort!, isless, resize!, empty!, push!, length, start, next, done, rand

typealias MinDepthByRule Dict{Symbol,Int64}
typealias MinDepthByAction Dict{Symbol,Vector{Int64}}

immutable RuleNotFoundException <: Exception end

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
GPESResult(grammar::Grammar) = GPESResult(DerivationTree(DerivTreeParams(grammar)), realmax(Float64),
    :(false), 0, 0)

exprsearch(p::GPESParams, problem::ExprProblem, userargs...) = gp_search(p, problem::ExprProblem, userargs...)

get_derivtree(result::GPESResult) = get_derivtree(result.tree)

type GPIndividual
    derivtree::DerivationTree
    expr
    fitness::Nullable{Float64}
end
GPIndividual(grammar::Grammar) = GPIndividual(DerivationTree(DerivTreeParams(grammar)), :(false), Nullable{Float64}())

type GPPopulation
    individuals::Vector{GPIndividual}
end
GPPopulation() = GPPopulation(Array(GPIndividual, 0))

function gp_search(p::GPESParams, problem::ExprProblem, userargs...)
    @notify_observer(p.observer, "verbose1", ["Starting GP search"])
    @notify_observer(p.observer, "computeinfo", ["starttime", string(now())])

    grammar = get_grammar(problem)
    result = GPESResult(grammar) 

    pop = GPPopulation()
    initialize!(pop, grammar, p.pop_size, p.maxdepth)
    fitness = realmax(Float64)
    iter = 1
    tstart = CPUtime_start()
    while iter <= p.iterations
        println("iteration $iter")
        pop = generate(p, grammar, pop, result, problem, userargs...)
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
    @assert result.fitness <= best_fitness(pop)

    return result
end

start(pop::GPPopulation) = start(pop.individuals)
next(pop::GPPopulation, s) = next(pop.individuals, s)
done(pop::GPPopulation, s) = done(pop.individuals, s)

function isless(x::GPIndividual, y::GPIndividual)
   get(x.fitness, realmax(eltype(x.fitness))) < get(y.fitness, realmax(eltype(y.fitness)))
end

function best_ind(pop::GPPopulation) 
    @assert !isnull(pop.individuals[1].fitness)
    @assert pop.individuals[1] <= pop.individuals[end] 
    pop.individuals[1]
end

function resize!(pop::GPPopulation, N::Int64)
    resize!(pop.individuals, N)
end

push!(pop::GPPopulation, ind::GPIndividual) = push!(pop.individuals, ind)
empty!(pop::GPPopulation) = empty!(pop.individuals)
length(pop::GPPopulation) = length(pop.individuals)

"""
Compute minimum depth for each rule
"""
function min_depth_rule(grammar::Grammar)
    d = MinDepthByRule()
    changed = Dict{Symbol,Bool}()
    for (k,v) in grammar.rules
        d[k] = typemax(Int64)/2
        changed[k] = true
    end
    while any(values(changed))
        for (k,rule) in grammar.rules 
           d_k = 1 + min_depth_rule(d, rule) 
           changed[k] = d[k] != d_k
           d[k] = d_k 
       end
    end
    d
end

min_depth_rule(d::MinDepthByRule, rule::ReferencedRule) = d[rule.symbol]
min_depth_rule(d::MinDepthByRule, rule::Union{RangeRule,Symbol}) = 0 #terminals
min_depth_rule(d::MinDepthByRule, x::Any) = 0 #terminals such as constants?
function min_depth_rule(d::MinDepthByRule, rule::OrRule)
    minimum(map(r->min_depth_rule(d,r), rule.values))
end
function min_depth_rule(d::MinDepthByRule, rule::ExprRule)
    a = filter(r->isa(r,ReferencedRule), rule.args)
    maximum(map(r->min_depth_rule(d,r), a))
end

function min_depth_actions(grammar::Grammar)
    d = min_depth_rule(grammar)
    da = min_depth_actions(d, grammar)
    da
end
function min_depth_actions(d::MinDepthByRule, grammar::Grammar)
    da = MinDepthByAction()
    for (k,rule) in grammar.rules
        da[k] = 1 + min_depth_actions(d, rule)
    end
    da
end
min_depth_actions(d::MinDepthByRule, rule::ReferencedRule) = Int64[d[rule.symbol]]
min_depth_actions(d::MinDepthByRule, rule::Symbol) = zeros(Int64, 1) 
min_depth_actions(d::MinDepthByRule, rule::RangeRule) = zeros(Int64, length(rule.range))
function min_depth_actions(d::MinDepthByRule, rule::OrRule)
    Int64[min_depth_rule(d, v) for v in rule.values]
end
function min_depth_actions(d::MinDepthByRule, rule::ExprRule)
    Int64[min_depth_rule(d, rule)]
end

"""
Initialize population.  Ramped initialization.
"""
function initialize!(pop::GPPopulation, grammar::Grammar, pop_size::Int64, maxdepth::Int64)
    empty!(pop)
    mdr = min_depth_rule(grammar) 
    mda = min_depth_actions(mdr, grammar)
    startmindepth = mdr[:start]

    for d in cycle(startmindepth:maxdepth)
        try
            ind = rand(grammar, mda, d) 
            push!(pop, ind) 
        catch e
            if !isa(e, IncompleteException)
                rethrow(e)
            end
        end
        println("pop: $(length(pop))")
        length(pop) < pop_size || break
    end
    pop
end

"""
Random individual to a target_depth
"""
function rand(grammar::Grammar, mda::MinDepthByAction, target_depth::Int64)
    opennodes = DerivTreeNode[]
    ind = GPIndividual(grammar)
    tree = ind.derivtree 
    node = initialize!(tree)
    push!(opennodes, node)
    while !isempty(opennodes)
        node = pop!(opennodes)
        if isa(node.rule, DecisionRule)
            depths = mda[Symbol(node.rule.name)]
            actions = find(x->x <= target_depth-node.depth, depths) 
            if isempty(actions) 
                warn("init(): no valid actions")
                throw(IncompleteException())
            end
            a = rand_element(actions)
            nodes = expand_node!(tree, node, a)
        else
            nodes = expand_node!(tree, node)
        end
        append!(opennodes, nodes)
    end
    ind
end

"""
Evaluate fitness function
"""
function evaluate!(pop::GPPopulation, result::GPESResult, problem::ExprProblem, userargs...)
    n = 0 #number of fitness calls
    for ind in pop
        if isnull(ind.fitness)
            try
                ind.expr = get_expr(ind.derivtree)
                fitness = get_fitness(problem, ind.expr, userargs...)
                ind.fitness = Nullable{Float64}(fitness) 
                result.totalevals += 1
                if fitness < result.fitness
                    result.fitness = fitness
                    result.tree = ind.derivtree
                    result.best_at_eval = result.totalevals
                    result.expr = ind.expr
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
function selection(pop::GPPopulation, tournament_size::Int64) 
    #randomly choose N inds and compare their fitness  
    ids = rand(1:length(pop), tournament_size)
    pool = pop[ids] 
    #return the ind with min fitness
    (fitness, id) = findmin(map(ind->ind.fitness, pool))
    pool[id]
end

"""
Crossover
"""
function crossover(ind1::GPIndividual, ind2::GPIndividual, grammar::Grammar)
    ind3 = GPIndividual(grammar)
    ind4 = GPIndividual(grammar)

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
    len = doffset + N - 1
    if length(dst) < len #resize up if needed, but not down
        resize!(dst, len)
    end
    copy!(dst.individuals, doffset, src.individuals, soffset, N)
end

function sort!(pop::GPPopulation) #lowest (best) fitness first
    sort!(pop.individuals)
end

"""
Create next generation
"""
function generate(p::GPESParams, grammar::Grammar, pop::GPPopulation, result::GPESResult, 
    problem::ExprProblem, userargs...)

    n_keep = floor(Int64, p.top_keep*p.pop_size)
    n_crossover = floor(Int64, p.crossover_frac*p.pop_size)
    n_mutate = floor(Int64, p.mutate_frac*p.pop_size)
    n_rand = floor(Int64, p.rand_frac*p.pop_size)

    evaluate!(pop, result, problem, userargs...)
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
            (ind3, ind4) = crossover(ind1, ind2, grammar)
            push!(newpop, ind3)
            push!(newpop, ind4)
            n += 2
        catch e
            if !isa(e, RuleNotFoundException)
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
            ind1 = rand(grammar, da, d)
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

    evaluate!(newpop, result, prblem, userargs...)
    sort!(pop)

    newpop
end

end #module
