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
Linearized Derivation tree (aka parse tree or syntax tree).  Wraps DerivationTrees.
Default is DFS order.
"""
module LinearDerivTrees 

export Grammar
export LDTParams, LDTActions, LinearDerivTree, AbstractTreeOrder, DepthFirst, BreadthFirst  
export initialize!, step!, play!, actionspace, isdone, iscomplete, get_derivtree, get_expr
export rand_with_retry!, maxlength

using Reexport
using ExprSearch 
@reexport using DerivationTrees
using GrammaticalEvolution
using DataStructures
using RLESUtils, Observers

import GrammaticalEvolution.Grammar
import DerivationTrees: initialize!, actionspace, iscomplete, get_expr, pretty_string
import ExprSearch: get_derivtree
import Base: length, push!, convert, rand!, getindex, empty!, copy!

const STACKSIZE = 50 #initial stack allocation size, default value 1024 allocates too much unused memory

abstract AbstractTreeOrder

immutable DepthFirst <: AbstractTreeOrder end
immutable BreadthFirst <: AbstractTreeOrder end

immutable LDTParams
    grammar::Grammar
    maxsteps::Int64
    order::AbstractTreeOrder
end
LDTParams(grammar::Grammar, maxsteps::Int64) = LDTParams(grammar, maxsteps, DepthFirst())

type LDTActions
    actions::Vector{Int64}
end
LDTActions() = LDTActions(Int64[])
LDTActions(ind::ExampleIndividual) = LDTActions(ind.genome+1)

type LinearDerivTree 
    params::LDTParams
    derivtree::DerivationTree
    opennodes::Stack
    actions::LDTActions
    observer::Observer
end

function LinearDerivTree(params::LDTParams; observer::Observer=Observer()) 
    derivtree = DerivationTree(DerivTreeParams(params.grammar))
    stack = Stack(DerivTreeNode, STACKSIZE)
    actions = LDTActions()
    LinearDerivTree(params, derivtree, stack, actions, observer) 
end

push!(actions::LDTActions, a::Int64) = push!(actions.actions, a)
length(actions::LDTActions) = length(actions.actions)
getindex(actions::LDTActions, ids) = getindex(actions.actions, ids)

function copy!(dst::LDTActions, src::LDTActions) 
    resize!(dst.actions, length(src.actions))
    copy!(dst.actions, src.actions)
end

convert(::Type{Array}, a::LDTActions) = copy(a.actions)
empty!(a::LDTActions) = empty!(a.actions)

get_expr(tree::LinearDerivTree) = get_expr(tree.derivtree)
empty!(stack::Stack) = empty!(stack.store)

function copy!(dst::LinearDerivTree, src::LinearDerivTree)
    dst.params = src.params
    copy!(dst.derivtree, src.derivtree)
    dst.opennodes = deepcopy(src.opennodes)
    copy!(dst.actions, src.actions)
    dst.observer = src.observer 
    dst
end

get_derivtree(tree::LinearDerivTree) = tree.derivtree

function pretty_string(tree::LinearDerivTree, fmt::Format, capitalize::Bool=false) 
    pretty_string(tree.derivtree, fmt, capitalize)
end

function initialize!(tree::LinearDerivTree)
  @notify_observer(tree.observer, "verbose1", ["initialize! called"])
  root = initialize!(tree.derivtree)
  empty!(tree.opennodes)
  push!(tree.opennodes, root)
  empty!(tree.actions)
  process_non_decisions!(tree)
end

function step!(tree::LinearDerivTree, a::Int64)
  @notify_observer(tree.observer, "verbose1", ["step! called"])
  opennodes = tree.opennodes
  if isempty(opennodes)
    return #we're done
  end
  as = actionspace(tree)
  a = ((a-1) % length(as)) + 1 #1-indexed 
  push!(tree.actions, a)
  node = pop!(opennodes)
  children = expand_node!(tree.derivtree, node, a)
  add_nodes!(tree.params.order, opennodes, children)
  process_non_decisions!(tree)
end

play!(tree::LinearDerivTree, x) = play!(tree, LDTActions(x))

function play!(tree::LinearDerivTree, actions::LDTActions)
  initialize!(tree)
  n = 0
  while !isdone(tree) && n < length(actions)
    n += 1
    step!(tree, actions[n])
  end
  return n #number of actions actually used
end

#iterator functions
#start(tree::LinearDerivTree) = start(tree.root)
#next(tree::LinearDerivTree, stack::Stack) = next(tree.root, stack)
#done(tree::LinearDerivTree, stack::Stack) = done(tree.root, stack)
#
##iterate through nodes in depth-first search order
#function start(startnode::DerivTreeNode)
  #stack = Stack(DerivTreeNode, STACKSIZE)
  #push!(stack, startnode)
  #return stack
#end
#
#function next(startnode::DerivTreeNode, stack::Stack)
  #node = pop!(stack)
  #for i = length(node.children):-1:1
    #push!(stack, node.children[i])
  #end
  #return node, stack
#end
#
#done(startnode::DerivTreeNode, stack::Stack) = isempty(stack)

function iscomplete(tree::LinearDerivTree) 
    @assert length(tree.opennodes) == tree.derivtree.nopen #sanity check only
    @assert isempty(tree.opennodes) == iscomplete(tree.derivtree) #sanity check only
    iscomplete(tree.derivtree)
end

function isdone(tree::LinearDerivTree)
  return iscomplete(tree) || length(tree.actions) >= tree.params.maxsteps
end

function process_non_decisions!(tree::LinearDerivTree)
  opennodes = tree.opennodes
  while !isempty(opennodes) && !isa(top(opennodes).rule, DecisionRule)
    node = pop!(opennodes)
    children = expand_node!(tree.derivtree, node)
    add_nodes!(tree.params.order, opennodes, children)
  end
end

function add_nodes!(::DepthFirst, opennodes::Stack, nodes_to_add::Vector{DerivTreeNode})
    for i = length(nodes_to_add):-1:1 
        push!(opennodes, nodes_to_add[i])
    end
end

function add_nodes!(::BreadthFirst, opennodes::Stack, nodes_to_add::Vector{DerivTreeNode})
    for i = length(nodes_to_add):-1:1 
        push!(opennodes, nodes_to_add[i])
    end
end

"""
Range of valid action numbers at next open node 
"""
function actionspace(tree::LinearDerivTree)
  if isempty(tree.opennodes)
    return 0:-1 #special -1 action for terminal state
  end
  node = top(tree.opennodes)
  return actionspace(node)
end

"""
Generate a random tree. 
Returns true if completed successfully, false otherwise
"""
function rand!(tree::LinearDerivTree)
    initialize!(tree)
    while !isdone(tree)
      as = actionspace(tree)
      step!(tree, rand(as))
    end
    return iscomplete(tree)
end
"""
Generate a random tree up to a maximum number of retries to get a complete tree. 
"""
function rand_with_retry!(tree::LinearDerivTree, retries::Int64=5)
    while retries > 0
        compl = rand!(tree) 
        if compl
            return true
        end
        retries -= 1
    end
    return false
end

end #module
