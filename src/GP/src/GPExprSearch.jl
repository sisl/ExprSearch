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
export crossover, mutate, max_depth

import Compat.view
using Reexport
using ExprSearch
using RLESUtils, GitUtils, CPUTimeUtils, RandUtils, Observers, LogSystems, TreeIterators, TreeUtils
import RLESTypes.SymbolTable
import DerivationTrees.get_children
import ExprSearch: get_derivtree, get_expr
using GrammaticalEvolution
@reexport using DerivationTrees  #for pretty strings
using CPUTime
using JLD

using DerivTreeVis

import DerivationTrees: initialize!, max_depth
import ..ExprSearch: SearchParams, SearchResult, exprsearch, ExprProblem, get_grammar, get_fitness
import Base: copy!, sort!, isless, resize!, empty!, push!, pop!, length, start, next, done, 
    rand, getindex

typealias MinDepthByRule Dict{Symbol,Int64}
typealias MinDepthByAction Dict{Symbol,Vector{Int64}}

immutable RuleNotFoundException <: Exception end
immutable DepthExceededException <: Exception end

const DEFAULT_EXPR = :()

include("logdefs.jl")

immutable GPESParams <: SearchParams
    pop_size::Int64
    maxdepth::Int64
    iterations::Int64
    tournament_size::Int64
    top_keep::Float64
    crossover_frac::Float64
    mutate_frac::Float64
    rand_frac::Float64
    default_expr
    logsys::LogSystem
    userargs::SymbolTable #passed to get_fitness if specified
end
GPESParams(pop_size::Int64, maxdepth::Int64, iterations::Int64, tournament_size::Int64, 
    top_keep::Float64, crossover_frac::Float64, mutate_frac::Float64, rand_frac::Float64,
    default_expr, logsys::LogSystem=logsystem(); userargs::SymbolTable=SymbolTable()) = 
        GPESParams(pop_size, maxdepth, iterations, tournament_size, top_keep, 
        crossover_frac, mutate_frac, rand_frac, default_expr, logsys, userargs)

type GPESResult <: SearchResult
    tree::DerivationTree
    fitness::Float64
    expr
    best_at_eval::Int64
    totalevals::Int64
end
GPESResult(grammar::Grammar) = GPESResult(DerivationTree(DerivTreeParams(grammar)), realmax(Float64),
    DEFAULT_EXPR, 0, 0)

exprsearch(p::GPESParams, problem::ExprProblem) = gp_search(p, problem::ExprProblem)

get_derivtree(result::GPESResult) = result.tree
get_expr(result::GPESResult) = result.expr
get_fitness(result::GPESResult) = result.fitness

type GPIndividual
    derivtree::DerivationTree
    expr
    fitness::Nullable{Float64}
end
GPIndividual(grammar::Grammar) = GPIndividual(DerivationTree(DerivTreeParams(grammar)), 
    DEFAULT_EXPR, Nullable{Float64}())

type GPPopulation
    individuals::Vector{GPIndividual}
end
GPPopulation() = GPPopulation(Array(GPIndividual, 0))

function gp_search(p::GPESParams, problem::ExprProblem)
    @notify_observer(p.logsys.observer, "verbose1", ["Starting GP search"])
    @notify_observer(p.logsys.observer, "computeinfo", ["starttime", string(now())])

    grammar = get_grammar(problem)
    mdr = min_depth_rule(grammar) 
    mda = min_depth_actions(mdr, grammar)
    result = GPESResult(grammar) 

    pop = GPPopulation()
    initialize!(pop, grammar, mdr, mda, p.pop_size, p.maxdepth)
    fitness = realmax(Float64)
    iter = 1
    tstart = CPUtime_start()
    while iter <= p.iterations
        pop = generate(p, grammar, mda, pop, result, problem)
        ind = best_ind(pop)
        fitness = get(ind.fitness)
        code = string(ind.expr)
        nevals = iter * p.pop_size
        @notify_observer(p.logsys.observer, "elapsed_cpu_s", [nevals, CPUtime_elapsed_s(tstart)]) 
        @notify_observer(p.logsys.observer, "fitness", Any[iter, fitness])
        @notify_observer(p.logsys.observer, "code", Any[iter, code])
        @notify_observer(p.logsys.observer, "population", Any[iter, pop])
        @notify_observer(p.logsys.observer, "current_best", [nevals, fitness, code])
        iter += 1
    end
    @assert result.fitness <= get(best_ind(pop).fitness)

    #dealloc pop
    for ind in pop
        rm_tree!(ind.derivtree)
    end

    @notify_observer(p.logsys.observer, "result", [result.fitness, string(result.expr), 
        result.best_at_eval, result.totalevals])

    #meta info
    @notify_observer(p.logsys.observer, "computeinfo", ["endtime",  string(now())])
    @notify_observer(p.logsys.observer, "computeinfo", ["hostname", gethostname()])
    @notify_observer(p.logsys.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
    @notify_observer(p.logsys.observer, "computeinfo", ["cpu_time", CPUtime_elapsed_s(tstart)]) 
    @notify_observer(p.logsys.observer, "parameters", ["pop_size", p.pop_size])
    @notify_observer(p.logsys.observer, "parameters", ["maxdepth", p.maxdepth])
    @notify_observer(p.logsys.observer, "parameters", ["iterations", p.iterations])
    @notify_observer(p.logsys.observer, "parameters", ["tournament_size", p.tournament_size])
    @notify_observer(p.logsys.observer, "parameters", ["top_keep", p.top_keep])
    @notify_observer(p.logsys.observer, "parameters", ["crossover_frac", p.crossover_frac])
    @notify_observer(p.logsys.observer, "parameters", ["mutate_frac", p.mutate_frac])
    @notify_observer(p.logsys.observer, "parameters", ["rand_frac", p.rand_frac])
    @notify_observer(p.logsys.observer, "parameters", ["default_expr", string(p.default_expr)])

    result 
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

pop!(pop::GPPopulation) = pop!(pop.individuals)
push!(pop::GPPopulation, ind::GPIndividual) = push!(pop.individuals, ind)
empty!(pop::GPPopulation) = empty!(pop.individuals)
length(pop::GPPopulation) = length(pop.individuals)
getindex(pop::GPPopulation, index) = getindex(pop.individuals, index)

function copy!(dst::GPIndividual, src::GPIndividual)
   copy!(dst.derivtree, src.derivtree)
   dst.expr = src.expr
   dst.fitness = src.fitness #isbits, so can just copy
end

max_depth(ind::GPIndividual) = max_depth(ind.derivtree)

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
           d_k = min_depth_rule(d, rule) 
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
    1 + minimum(map(r->min_depth_rule(d,r), rule.values))
end
function min_depth_rule(d::MinDepthByRule, rule::ExprRule)
    a = filter(r->isa(r,ReferencedRule), rule.args)
    1 + maximum(map(r->min_depth_rule(d,r), a))
end
function min_depth_rule(d::MinDepthByRule, rule::AndRule)
    a = filter(r->isa(r,ReferencedRule), rule.values)
    1 + maximum(map(r->min_depth_rule(d,r), a))
end

"""
Compute minimum depth per action of decision rule
"""
function min_depth_actions(grammar::Grammar)
    d = min_depth_rule(grammar)
    da = min_depth_actions(d, grammar)
    da
end
function min_depth_actions(d::MinDepthByRule, grammar::Grammar)
    da = MinDepthByAction()
    for (k,rule) in grammar.rules
        da[k] = min_depth_actions(d, rule)
    end
    da
end
min_depth_actions(d::MinDepthByRule, rule::ReferencedRule) = Int64[d[rule.symbol]]
min_depth_actions(d::MinDepthByRule, rule::Union{Symbol,Terminal}) = zeros(Int64, 1) 
min_depth_actions(d::MinDepthByRule, rule::RangeRule) = zeros(Int64, length(rule.range))
function min_depth_actions(d::MinDepthByRule, rule::OrRule)
    1 + Int64[min_depth_rule(d, v) for v in rule.values]
end
function min_depth_actions(d::MinDepthByRule, rule::Union{AndRule,ExprRule})
    Int64[min_depth_rule(d, rule)]
end

"""
Initialize population.  Ramped initialization.
"""
function initialize!(pop::GPPopulation, grammar::Grammar, mdr::MinDepthByRule, 
    mda::MinDepthByAction, pop_size::Int64, maxdepth::Int64)

    empty!(pop)
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
        length(pop) < pop_size || break
    end
    pop
end

function resample_subtree!(tree::DerivationTree, node::DerivTreeNode, mda::MinDepthByAction,
    target_depth::Int64)
    rm_node(tree, node.children) #return child nodes
    opennodes = DerivTreeNode[node]
    tree.nopen = 1
    rand_subtree!(tree, node, opennodes, mda, target_depth) 
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
    rand_subtree!(tree, node, opennodes, mda, target_depth)
    ind
end

function rand_subtree!(tree::DerivationTree, node::DerivTreeNode, opennodes::Vector{DerivTreeNode}, 
    mda::MinDepthByAction, target_depth::Int64)
    while !isempty(opennodes)
        node = pop!(opennodes)
        if is_decision(node) 
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
end
"""
Evaluate fitness function
"""
function evaluate!(p::GPESParams, pop::GPPopulation, result::GPESResult, problem::ExprProblem, 
    default_expr)
    for ind in pop
        if isnull(ind.fitness)
            try
                ind.expr = get_expr(ind.derivtree)
                fitness = get_fitness(problem, ind.derivtree, p.userargs)
                ind.fitness = Nullable{Float64}(fitness) 
                result.totalevals += 1
                if fitness < result.fitness
                    result.fitness = fitness
                    copy!(result.tree, ind.derivtree)
                    result.best_at_eval = result.totalevals
                    result.expr = ind.expr
                end
            catch e
                if !isa(e, IncompleteException)
                    rethrow(e)
                end
                ind.expr = default_expr
                ind.fitness = Nullable{Float64}(realmax(Float64))
            end
        end
    end
end

"""
Tournament selection
Best fitness wins deterministically
"""
function selection(pop::GPPopulation, tournament_size::Int64) 
    #randomly choose N inds and compare their fitness  
    ids = randperm(length(pop))
    #sorted, just return the one with the lowest index
    id = minimum(view(ids, 1:tournament_size)) 
    pop[id]
end

TreeUtils.get_children(node::DerivTreeNode) = DerivationTrees.get_children(node)

function findbyrule(ind::GPIndividual, name::AbstractString)
    candidates = DerivTreeNode[]
    for node in tree_iter(ind.derivtree.root) #TODO: replace with AbstractTrees.jl interface
        if node.rule.name == name 
            push!(candidates, node)
        end
    end
    candidates
end

"""
Crossover
"""
function crossover(ind1::GPIndividual, ind2::GPIndividual, grammar::Grammar, maxdepth::Int64)
    ind3 = GPIndividual(grammar)
    ind4 = GPIndividual(grammar)
    copy!(ind3.derivtree, ind1.derivtree)
    copy!(ind4.derivtree, ind2.derivtree)

    node3 = rand_node(n->!is_terminal(n), ind3.derivtree.root) #choose crossover point on ind3
    candidates = findbyrule(ind4, node3.rule.name) #find corresponding on ind4
    isempty(candidates) && throw(RuleNotFoundException())
    node4 = rand_element(candidates)
    swap_children!(node3, node4)
    if max_depth(ind3) > maxdepth || max_depth(ind4) > maxdepth
        throw(DepthExceededException())
    end
    (ind3, ind4)
end #module

"""
Mutation
"""
function mutate(ind1::GPIndividual, grammar::Grammar, mda::MinDepthByAction, target_depth::Int64)
    ind2 = GPIndividual(grammar)
    copy!(ind2.derivtree, ind1.derivtree)

    #choose random node
    tree = ind2.derivtree
    node = rand_node(tree.root)
    resample_subtree!(tree, node, mda, target_depth)
    ind2
end

"""
Adjust to target pop_size
"""
function adjust_to_size!(newpop::GPPopulation, pop::GPPopulation, grammar::Grammar, 
    pop_size::Int64, tournament_size::Int64)
    #reproduce if undersize
    while length(newpop) < pop_size
        ind1 = selection(pop, tournament_size)
        ind2 = GPIndividual(grammar)
        copy!(ind2, ind1)
        push!(newpop, ind2)
    end
    
    #trim to size
    while length(newpop) > pop_size
        ind = pop!(newpop)
        rm_tree!(ind.derivtree)
    end
    newpop
end

function sort!(pop::GPPopulation) #lowest (best) fitness first
    sort!(pop.individuals)
end

"""
Create next generation
"""
function generate(p::GPESParams, grammar::Grammar, mda::MinDepthByAction, 
    pop::GPPopulation, result::GPESResult, problem::ExprProblem)

    #println("enter generate")
    n_keep = floor(Int64, p.top_keep*p.pop_size)
    n_crossover = floor(Int64, p.crossover_frac*p.pop_size)
    n_mutate = floor(Int64, p.mutate_frac*p.pop_size)
    n_rand = floor(Int64, p.rand_frac*p.pop_size)

    evaluate!(p, pop, result, problem, p.default_expr)
    sort!(pop)
    newpop = GPPopulation()

    #pass through the top ones
    for i = 1:n_keep
        ind1 = GPIndividual(grammar)
        copy!(ind1, pop[i])
        push!(newpop, ind1)
    end
    
    #crossover
    n = 0
    while n < n_crossover
        #println("crossover: $n")
        ind1 = selection(pop, p.tournament_size)
        ind2 = selection(pop, p.tournament_size) 
        try
            (ind3, ind4) = crossover(ind1, ind2, grammar, p.maxdepth)
            push!(newpop, ind3)
            push!(newpop, ind4)
            n += 2
        catch e
            if !isa(e, Union{RuleNotFoundException,DepthExceededException})
                rethrow(e)
            end
        end
    end

    #mutate
    n = 0
    while n < n_mutate
        #println("mutate: $n")
        ind1 = selection(pop, p.tournament_size)
        try
            ind2 = mutate(ind1, grammar, mda, p.maxdepth)
            push!(newpop, ind2)
            n += 1
        catch e
            if !isa(e, IncompleteException)
                rethrow(e)
            end
        end
    end

    #random
    n = 0
    while n < n_rand 
        #println("random: $n")
        try
            ind1 = rand(grammar, mda, p.maxdepth)
            push!(newpop, ind1)
            n += 1
        catch e
            if !isa(e, IncompleteException)
                rethrow(e)
            end
        end
    end

    #adjust size: trim or reproduce to size
    adjust_to_size!(newpop, pop, grammar, p.pop_size, p.tournament_size)

    #dealloc old pop
    for ind in pop
        rm_tree!(ind.derivtree)
    end

    evaluate!(p, newpop, result, problem, p.default_expr)
    sort!(newpop)

    newpop
end

type GPESResultSerial <: SearchResult
    fitness::Float64
    expr
    best_at_eval::Int64
    totalevals::Int64
end

#don't store the tree to JLD, it's too big and causes stackoverflowerror
function JLD.writeas(r::GPESResult)
    GPESResultSerial(r.fitness, r.expr, r.best_at_eval, r.totalevals)
end

end #module
