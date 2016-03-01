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

using ExprSearch
using DerivationTrees
using Base.Test

function test1()
  const MAXSTEPS = 25

  function create_grammar()
    @grammar grammar begin
      start = ex
      ex = sum | product | (ex) | value
      sum = Expr(:call, :+, ex, ex)
      product = Expr(:call, :*, ex, ex)
      value = :x | :y | digit
      digit = 0:9
    end
    return grammar
  end

  grammar = create_grammar()

  params = DerivTreeParams(grammar, MAXSTEPS)
  tree = DerivationTree(params)

  srand(1)

  rand!(tree)
  get_expr(tree)

  size(tree)
  count(x->isa(x.rule, DecisionRule), tree)

  for i = 1:500
    try
      @time begin
        node = rand(tree)
        rand!(tree, node, 1)
      end
      get_expr(tree)
    end
  end

  tree1 = DerivationTree(params)
  rand!(tree1)
  @show get_expr(tree1)
  @show size(tree1)
  tree2 = DerivationTree(params)
  rand!(tree2)
  @show get_expr(tree2)
  @show size(tree2)
  copy!(tree2, tree1)
  @show get_expr(tree2)
end
