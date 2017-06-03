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
Grammar-based data mining problem
"""
module GBDMProblems

export GBDMProblem, FitnessFunction, apply_expr

using GrammaticalEvolution
using RLESUtils, DataFrameSets, Interpreter
using ExprSearch, DerivationTrees

import RLESTypes.SymbolTable
import GBDTs.apply_expr

abstract FitnessFunction

immutable GBDMProblem{T} <: ExprProblem
    data::DFSetLabeled{T}
    grammar::Grammar
    fitness_function::FitnessFunction
    symtables::Vector{SymbolTable}
end

function GBDMProblem{T}(data::DFSetLabeled{T}, grammar::Grammar, 
    fitness_function::FitnessFunction, symtable::SymbolTable)
    symtables = [deepcopy(symtable) for i=1:Threads.nthreads()]
    GBDMProblem(data, grammar, fitness_function, symtables) 
end

function ExprSearch.get_fitness(problem::GBDMProblem, derivtree::DerivationTree, 
    userargs::SymbolTable=SymbolTable())
    get_fitness(problem.fitness_function, problem, derivtree, userargs)
end

ExprSearch.get_grammar(problem::GBDMProblem) = problem.grammar

"""
Apply expr to data to get vector of predicted labels (vector of bools since
Boolean expression)
"""
function apply_expr{T}(problem::GBDMProblem{T}, ids::Vector{Int64}, expr)
    apply_expr(problem, view(getrecords(problem.data), ids), expr)
end

function apply_expr{T}(problem::GBDMProblem{T}, records::AbstractVector{DataFrame}, expr)
    symtable = problem.symtables[Threads.threadid()]
    expr_labels = Vector{Bool}(length(records))
    for i = 1:length(records)
        # manually inlined eval_expr
        symtable[:D] = records[i] 
        expr_labels[i] = interpret(symtable, expr)
    end
    expr_labels #::Vector{Bool}
end

end #module
