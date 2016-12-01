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
Grammar-Based Expression Search.
Available algorithms: Simulated Annealing (SA), Monte Carlo (MC), Grammatical Evolution (GE),
Monte Carlo Tree Search (MCTS) (no committing steps).

Usage: using ExprSearch.MC; result = exprsearch(p, problem)
"""
module ExprSearch

export ExprProblem, get_grammar, get_fitness
export SearchParams, SearchResult, exprsearch
export get_expr, get_derivtree

const MODULEDIR = joinpath(dirname(@__FILE__), "..", "modules")

using Reexport
@reexport using GrammaticalEvolution
using RLESUtils, ModLoader
import RLESTypes.SymbolTable

load_to_path(MODULEDIR)
const PKGS = readdir(MODULEDIR)

using DerivationTrees
import DerivationTrees.get_expr

abstract ExprProblem 
abstract SearchParams
abstract SearchResult

exprsearch(p::SearchParams, problem::ExprProblem) = error("Please use a submodule.")

get_grammar(problem::ExprProblem) = error("ExprSearch::get_grammar() not defined")
get_fitness(problem::ExprProblem, derivtree::DerivationTree, 
    userargs::SymbolTable) = error("ExprSearch::get_fitness() not defined")

get_expr(::Void) = ""
get_fitness(::Void) = realmax(Float64) 

get_derivtree(result::SearchResult) = error("ExprSearch::get_derivtree() not defined")

"""
Test an individual submodule
"""
function test(pkgs::AbstractString...; coverage::Bool=false)
  cd(() -> Pkg.Entry.test(AbstractString[pkgs...]; coverage=coverage), MODULEDIR)
end

"""
Test all submodules in modules folder.  Don't stop on error.
"""
function testall()
    for pkg in PKGS
        try
            test(pkg)
        catch
            println("Error in $pkg")
        end
    end
end

include("GP/src/GPExprSearch.jl") #Genetic Programming (standard tree-based)

include("GE/src/GEExprSearch.jl") #Grammatical Evolution

include("MC/src/MCExprSearch.jl") #Monte Carlo (random)

include("PMC/src/PMCExprSearch.jl") #Parallel Monte Carlo

include("MCTS/src/MCTSExprSearch.jl") #Monte Carlo Tree Search without committing steps

#include("Ref/src/RefExprSearch.jl") #Ref

#include("SA/src/SAExprSearch.jl") #SA

end #module


