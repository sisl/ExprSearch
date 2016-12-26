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
Derivation tree (aka parse tree or syntax tree) based on GrammaticalEvolution.jl grammar rules.
Warning: not all rules are supported
"""
module DerivationTrees

export DerivTreeParams, DerivationTree, DerivTreeNode, DecisionRule, get_expr,
    maxlength, get_children
export initialize!, actionspace, iscomplete, isleaf, expand_node!, is_decision, 
    is_terminal, swap_children!, max_depth, rm_tree!, rm_node, count_leafs, 
    count_nonleafs
export IncompleteException

import Compat.ASCIIString
using RLESUtils, MemPools, TreeIterators, TreeUtils
using Reexport
@reexport using GrammaticalEvolution

import Base: length, copy!

typealias DecisionRule Union{OrRule, RangeRule, RepeatedRule} #rules that require a decision

type DerivTreeParams
    grammar::Grammar
end

type DerivTreeNode
    cmd::ASCIIString
    rule::Union{Rule,Void}
    action::Int64
    depth::Int64
    children::Vector{DerivTreeNode}
end
function DerivTreeNode(rule::Union{Rule,Void}=nothing, depth::Int64=0,
                       cmd::AbstractString="", action::Int64=-1)
    return DerivTreeNode(cmd, rule, action, depth, DerivTreeNode[])
end

const NODEPOOL = MemPool(DerivTreeNode,10000,1000000)

type DerivationTree
    params::DerivTreeParams
    root::DerivTreeNode
    nopen::Int64
    maxactions::Int64 #maximum size of actionspace
    nodepool::MemPool
end

function DerivationTree(p::DerivTreeParams, nodepool::MemPool=NODEPOOL) 
    root = DerivTreeNode(p.grammar.rules[:start])
    maxactions = maxlength(p.grammar)
    tree = DerivationTree(p, root, 0, maxactions, nodepool)
    tree
end

include("formatter.jl")

immutable IncompleteException <: Exception end

const EMPTYARRAY = DerivTreeNode[]

is_decision(node::DerivTreeNode) = isa(node.rule, DecisionRule)
is_terminal(node::DerivTreeNode) = isempty(node.children)

#reset tree, children are deallocated, root is kept
function initialize!(tree::DerivationTree)
    rm_node(tree, tree.root.children)
    initialize!(tree.root)
    tree.root.rule = tree.params.grammar.rules[:start]
    tree.nopen = 1
    tree.root
end

#reset node, children are emptied
function initialize!(node::DerivTreeNode)
    node.rule = nothing
    node.depth = 0
    node.cmd = ""
    node.action = -1
    empty!(node.children)
    node
end

function mk_node(tree::DerivationTree)
    node = checkout(tree.nodepool)
    initialize!(node)
    node
end

function rm_tree!(tree::DerivationTree)
    rm_node(tree, [tree.root])
end

function rm_node(tree::DerivationTree, nodes::Vector{DerivTreeNode})
    return_to_pool(tree.nodepool, nodes)
    nothing
end

#returns children to pool before emptying
function return_to_pool(nodepool::MemPool, nodes::Vector{DerivTreeNode})
    for node in nodes
        return_to_pool(nodepool, node)
    end
    empty!(nodes)
end

function return_to_pool(nodepool::MemPool, node::DerivTreeNode)
    return_to_pool(nodepool, node.children) #return children first
    checkin(nodepool, node) #return self
end

function iscomplete(tree::DerivationTree) 
    @assert tree.nopen >= 0 #sanity check
    tree.nopen == 0 
end

expand_node!(tree::DerivationTree, node::DerivTreeNode) = expand_node!(tree, node, node.rule)
expand_node!(tree::DerivationTree, node::DerivTreeNode, a::Int64) = 
    expand_node!(tree, node, node.rule, a)

# Open nodes are allocated (and returned), but not populated until they are processed
###########################
### expand_node! nonterminals
function expand_node!(tree::DerivationTree, node::DerivTreeNode, rule::OrRule, a::Int64)
    tree.nopen -= 1  
    @assert tree.nopen >= 0 #sanity check
    node.action = a
    node.cmd = rule.name
    idx = ((a - 1) % length(rule.values)) + 1 #1-indexed
    child = mk_node(tree)
    child.rule = rule.values[idx]
    child.depth = node.depth + 1
    push!(node.children, child)
    tree.nopen += length(node.children)
    node.children
end

function expand_node!(tree::DerivationTree, node::DerivTreeNode, rule::ReferencedRule)
  #don't create a child node for reference rules, shortcut through
    node.rule = tree.params.grammar.rules[rule.symbol]
    DerivTreeNode[node]
end

function expand_node!(tree::DerivationTree, node::DerivTreeNode, rule::RepeatedRule, a::Int64)
    tree.nopen -= 1  
    @assert tree.nopen >= 0 #sanity check
    node.cmd = rule.name
    reps = ((a - 1) % length(rule.range)) + rule.range.start
    for i = 1:reps
        child = mk_node(tree)
        child.rule = rule.value
        child.depth = node.depth + 1
        push!(node.children, child)
    end
    tree.nopen += length(node.children)
    node.children
end

function expand_node!(tree::DerivationTree, node::DerivTreeNode, rule::AndRule)
    tree.nopen -= 1  
    @assert tree.nopen >= 0 #sanity check
    node.cmd = rule.name
    for subrule in rule.values
        child = mk_node(tree)
        child.rule = subrule
        child.depth = node.depth + 1
        push!(node.children, child)
    end
    tree.nopen += length(node.children)
    node.children
end

function expand_node!(tree::DerivationTree, node::DerivTreeNode, rule::ExprRule)
    tree.nopen -= 1  
    @assert tree.nopen >= 0 #sanity check
    node.cmd = rule.name
    for arg in rule.args
        if isa(arg, Rule)
            child = mk_node(tree)
            child.rule = arg
            child.depth = node.depth + 1
            push!(node.children, child)
        end
    end
    tree.nopen += length(node.children)
    node.children
end

###########################
### Terminals
function expand_node!(tree::DerivationTree, node::DerivTreeNode, rule::RangeRule, a::Int64)
    tree.nopen -= 1  
    @assert tree.nopen >= 0 #sanity check
    node.action = a
    EMPTYARRAY #use a pre-allocated empty array
end

function expand_node!(tree::DerivationTree, node::DerivTreeNode, rule::Terminal)
    tree.nopen -= 1  
    @assert tree.nopen >= 0 #sanity check
    EMPTYARRAY #use a pre-allocated empty array
end

###########################
### get_expr
function get_expr(tree::DerivationTree) #entry
    iscomplete(tree) ? get_expr(tree.root) : throw(IncompleteException())
end

get_expr(node::DerivTreeNode) = get_expr(node, node.rule)
get_expr(node::DerivTreeNode, rule::Terminal) = rule.value

function get_expr(node::DerivTreeNode, rule::RangeRule)
    value = ((node.action - 1) % length(rule.range)) + rule.range.start
    if rule.action != nothing
        value = rule.action(value)
    end
    value
end

function get_expr(node::DerivTreeNode, rule::OrRule)
    child = node.children[1]
    value = get_expr(child, child.rule)
    if rule.action != nothing
        value = rule.action(value)
    end
    value
end

function get_expr(node::DerivTreeNode, rule::RepeatedRule)
    values = Any[]
    for child in node.children
        push!(values, get_expr(child, child.rule))
    end
    if rule.action !== nothing
        values = rule.action(values)
    end
    values
end

function get_expr(node::DerivTreeNode, rule::AndRule)
    values = Any[]
    for child in node.children
        push!(values, get_expr(child, child.rule))
    end
    if rule.action !== nothing
        values = rule.action(values)
    end
    values
end

function get_expr(node::DerivTreeNode, rule::ExprRule)
    xs = Any[]
    child_i = 1
    for arg in rule.args
        if isa(arg, Rule)
            child = node.children[child_i]
            push!(xs, get_expr(child, child.rule))
            child_i += 1
        else
            push!(xs, arg)
        end
    end
    Expr(xs...)
end

###########################
###
"""
Range of valid action numbers at next open node 
"""
actionspace(node::DerivTreeNode) = actionspace(node, node.rule)
actionspace(node::DerivTreeNode, rule::OrRule) = 1:length(rule.values)
actionspace(node::DerivTreeNode, rule::RangeRule) = 1:length(rule.range)
actionspace(node::DerivTreeNode, rule::RepeatedRule) = 1:length(rule.range)

###########################
###

#entry
"""
Returns the highest length of actionspace of a grammar
"""
maxlength(grammar::Grammar) = reduce(max, 0, map(length, values(grammar.rules)))

#decision rules
Base.length(rule::OrRule) = length(rule.values)
Base.length(rule::RangeRule) = length(rule.range)
#other rules
Base.length(rule::Rule) = 1

function copy!(dst::DerivationTree, src::DerivationTree)
    initialize!(dst)
    dst.params = src.params
    copy!(dst.root, src.root, dst.nodepool)
    dst.nopen = src.nopen
    dst.maxactions = src.maxactions
    dst.nodepool = src.nodepool
end

function copy!(dst::DerivTreeNode, src::DerivTreeNode, nodepool::MemPool{DerivTreeNode})
    dst.rule = src.rule
    dst.depth = src.depth
    dst.cmd = src.cmd
    dst.action = src.action
    for srcchild in src.children
        dstchild = checkout(nodepool)
        copy!(dstchild, srcchild, nodepool)
        push!(dst.children, dstchild)
    end
end

isleaf(node::DerivTreeNode) = isempty(node.children)
get_children(node::DerivTreeNode) = node.children

function swap_children!(x::DerivTreeNode, y::DerivTreeNode)
    ch = x.children
    x.children = y.children
    y.children = ch
end

max_depth(tree::DerivationTree) = max_depth(tree.root)
max_depth(node::DerivTreeNode) = traverse(x->x.depth, (x,y)->max(x,y), node)

count_leafs(tree::DerivationTree) = count_nodes(isleaf, tree.root)
count_nonleafs(tree::DerivationTree) = count_nodes(x->!isleaf(x), tree.root)
length(tree::DerivationTree) = count_nodes(tree.root)

end #module
