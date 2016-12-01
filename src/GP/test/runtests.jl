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

using ExprSearch, SymbolicRegression, DerivTreeVis
using ExprSearch.GP
using RLESUtils, LogSystems, Loggers
using Base.Test

const DIR = dirname(@__FILE__)
const RESULTDIR = joinpath(DIR, "..", "..", "..", "results") 

const MINDEPTHBYRULE = MinDepthByRule(
    :start => 2,
    :ex => 2,
    :sum => 3,
    :product => 3,
    :value => 1,
    :digit => 0)

const MINDEPTHBYACTION = MinDepthByAction(
    :start => Int64[2],
    :ex => Int64[4,4,3,2],
    :sum => Int64[3],
    :product => Int64[3],
    :value => Int64[1,1,1],
    :digit => Int64[0,0,0,0,0,0,0,0,0,0])

"""
Example usage:
symbolic_gp(; seed=1)
"""
function symbolic_gp(;outdir::AbstractString=joinpath(RESULTDIR, "Symbolic_GP"),
                     seed=1,
                     logfileroot::AbstractString="symbolic_gp_log",

                     pop_size::Int64=1000,
                     maxdepth::Int64=10,
                     iterations::Int64=10,
                     tournament_size::Int64=20,
                     top_keep::Float64=0.1,
                     crossover_frac::Float64=0.4,
                     mutate_frac::Float64=0.2,
                     rand_frac::Float64=0.2,
                     default_code::Any=0.0,

                     gt_file::AbstractString="gt_easy.jl",

                     vis::Bool=true)
    srand(seed)
    mkpath(outdir)

    logsys = GP.logsystem()
    empty_listeners!(logsys)
    send_to!(STDOUT, logsys, ["verbose1", "current_best_print", "result"])
    logs = TaggedDFLogger()
    send_to!(logs, logsys, ["code", "computeinfo", "current_best", "elapsed_cpu_s", "fitness",
        "fitness5", "parameters", "result"])

    problem = Symbolic(gt_file)
  
    gp_params = GPESParams(pop_size, maxdepth, iterations, tournament_size, top_keep,
        crossover_frac, mutate_frac, rand_frac, default_code, logsys)
  
    result = exprsearch(gp_params, problem)

    outfile = joinpath(outdir, "$(logfileroot).txt")
    save_log(outfile, logs)

    if vis
        derivtreevis(get_derivtree(result), joinpath(outdir, "$(logfileroot)_derivtreevis"))
    end
    @show result.expr
    return result
end

function test_mindepth()
    problem = Symbolic("gt_easy.jl")
    grammar = get_grammar(problem)
    d = min_depth_rule(grammar)
    @test d == MINDEPTHBYRULE 
    da = min_depth_actions(d, grammar)
    @test da == MINDEPTHBYACTION
end

function test_rand(target_depth=5)
    problem = Symbolic("gt_easy.jl")
    grammar = get_grammar(problem)
    mda = min_depth_actions(grammar)
    for i = 1:5
        ind = rand(grammar, mda, target_depth)
        @show get_expr(ind.derivtree)
    end
end

using RLESUtils, TreeUtils
function test_randnode()
    problem = Symbolic()
    p = GPESParams(10,5,10,10,0.1,0.4,0.2,0.2,0.0)
    grammar = get_grammar(problem)
    mda = min_depth_actions(grammar)
    ind = rand(grammar, mda, 5) 
    node = rand_node(ind.derivtree.root)
end

using DerivTreeVis
function test_crossover()
    problem = Symbolic()
    p = GPESParams(10,5,10,10,0.1,0.4,0.2,0.2,0.0)
    grammar = get_grammar(problem)
    mda = min_depth_actions(grammar)
    ind1 = rand(grammar, mda, 5) 
    node1 = rand_node(ind1.derivtree.root)
    derivtreevis(ind1.derivtree, "node1")
    ind2 = rand(grammar, mda, 5) 
    node2 = rand_node(ind2.derivtree.root)
    derivtreevis(ind2.derivtree, "node2")
    (ind3, ind4) = crossover(ind1, ind2, grammar,5)
    derivtreevis(ind3.derivtree, "node3")
    derivtreevis(ind4.derivtree, "node4")
end

function test_maxdepth()
    problem = Symbolic()
    grammar = get_grammar(problem)
    mda = min_depth_actions(grammar)
    for i = 1:5
        ind = rand(grammar, mda, 5)
        @show max_depth(ind)
        derivtreevis(ind.derivtree, "ind$i")
    end
end

function test_mutate()
    problem = Symbolic()
    p = GPESParams(10,5,10,10,0.1,0.4,0.2,0.2,0.0)
    grammar = get_grammar(problem)
    mda = min_depth_actions(grammar)
    ind1 = rand(grammar, mda, 10) 
    derivtreevis(ind1.derivtree, "ind1")
    ind2 = mutate(ind1, grammar, mda, 10) 
    derivtreevis(ind2.derivtree, "ind2")
end
