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

#Derivation tree for GrammaticalEvolution
#Warning: not all rules are supported
module DerivationTrees

export DerivTreeParams, DerivationTree, DerivTreeNode, DecisionRule, get_expr, maxlength
export initialize!, step!, isterminal, actionspace, iscomplete, play!
export IncompleteException

using RLESUtils: Observers, MemPools
using Reexport
@reexport using GrammaticalEvolution
using DataStructures

import Base: empty!, length, start, next, done, size, count, rand, rand!, copy!

typealias DecisionRule Union{OrRule, RangeRule, RepeatedRule} #rules that require a decision

const STACKSIZE = 50 #initial stack allocation size, default value 1024 allocates too much unused memory

type DerivTreeParams
  grammar::Grammar
  maxsteps::Int64
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

type DerivationTree
  params::DerivTreeParams
  root::DerivTreeNode
  opennodes::Stack
  nsteps::Int64 #track number of steps taken
  maxactions::Int64 #maximum size of actionspace
  nodepool::MemPool
  observer::Observer
end

function DerivationTree(p::DerivTreeParams, nodepool::MemPool=MemPool(DerivTreeNode,200,400); observer::Observer=Observer())
  root = DerivTreeNode(p.grammar.rules[:start])
  maxactions = maxlength(p.grammar)
  tree = DerivationTree(p, root, Stack(DerivTreeNode, STACKSIZE), 0, maxactions, nodepool, observer)
  return tree
end

include("formatter.jl")

immutable IncompleteException <: Exception end

#reset tree, children are deallocated
function reset!(tree::DerivationTree)
  empty!(tree.opennodes)
  tree.nsteps = 0
  return2pool(tree.nodepool, tree.root.children)
  reset!(tree.root)
  tree.root.rule = tree.params.grammar.rules[:start]
end

#reset node, children are emptied
#can use kwargs as shorthand to set variables
function reset!(node::DerivTreeNode)
  node.rule = nothing
  node.depth = 0
  node.cmd = ""
  node.action = -1
  empty!(node.children)
end

#returns children to pool before emptying
function return2pool(nodepool::MemPool, nodes::Vector{DerivTreeNode})
  for node in nodes
    return2pool(nodepool, node)
  end
  empty!(nodes)
end

function return2pool(nodepool::MemPool, node::DerivTreeNode)
  return2pool(nodepool, node.children) #return children first
  checkin(nodepool, node) #return self
end

function initialize!(tree::DerivationTree)
  @notify_observer(tree.observer, "verbose1", ["initialize! called"])
  reset!(tree)
  push!(tree.opennodes, tree.root)
  process_non_decisions!(tree)
end

function step!(tree::DerivationTree, a::Int64)
  @notify_observer(tree.observer, "verbose1", ["step! called"])
  opennodes = tree.opennodes
  if isempty(opennodes)
    return #we're done
  end
  tree.nsteps += 1
  node = pop!(opennodes)
  process!(tree, node, node.rule, a)
  process_non_decisions!(tree)
end

function isterminal(tree::DerivationTree)
  return iscomplete(tree) || tree.nsteps >= tree.params.maxsteps
end

iscomplete(tree::DerivationTree) = isempty(tree.opennodes)

function process_non_decisions!(tree::DerivationTree)
  opennodes = tree.opennodes
  while !isempty(opennodes) && !isa(top(opennodes).rule, DecisionRule)
    node = pop!(opennodes)
    process!(tree, node, node.rule)
  end
end

###########################
### process! nonterminals
function process!(tree::DerivationTree, node::DerivTreeNode, rule::OrRule, a::Int64)
  node.action = a
  node.cmd = rule.name
  idx = ((a - 1) % length(rule.values)) + 1
  child = checkout(tree.nodepool)
  reset!(child)
  child.rule = rule.values[idx]
  child.depth = node.depth + 1
  push!(node.children, child)
  push!(tree.opennodes, child)
end

function process!(tree::DerivationTree, node::DerivTreeNode, rule::ReferencedRule)
  #don't create a child node for reference rules, shortcut through
  #node.cmd = rule.name
  node.rule = tree.params.grammar.rules[rule.symbol]
  push!(tree.opennodes, node)
end

function process!(tree::DerivationTree, node::DerivTreeNode, rule::RepeatedRule, a::Int64)
  node.cmd = rule.name
  reps = ((a - 1) % length(rule.range)) + rule.range.start
  for i = 1:reps
    child = checkout(tree.nodepool)
    reset!(child)
    child.rule = rule.value
    child.depth = node.depth + 1
    push!(node.children, child)
  end
  #load stack in reverse order (to mirror GE behavior)
  for i = length(node.children):-1:1
    push!(tree.opennodes, node.children[i])
  end
end

#= not tested...
function process!(tree::DerivationTree, node::DerivTreeNode, rule::AndRule)
  for subrule in rule.values
    child = DerivTreeNode(rule.name, subrule, node.depth + 1)
    push!(node.children, child)
    push!(tree.opennodes, child)
  end
end
=#

function process!(tree::DerivationTree, node::DerivTreeNode, rule::ExprRule)
  node.cmd = rule.name
  for arg in rule.args
    if isa(arg, Rule)
      child = checkout(tree.nodepool)
      reset!(child)
      child.rule = arg
      child.depth = node.depth + 1
      push!(node.children, child)
    end
  end
  #load stack in reverse order
  for i = length(node.children):-1:1
    push!(tree.opennodes, node.children[i])
  end
end

###########################
### Terminals
function process!(tree::DerivationTree, node::DerivTreeNode, rule::RangeRule, a::Int64)
  node.action = a
  #node.cmd = string(get_expr(node, rule))
end

function process!(tree::DerivationTree, node::DerivTreeNode, rule::Terminal)
  #node.cmd = string(rule.value)
end

#function process!(tree::DerivationTree, node::DerivTreeNode, x)
#  node.cmd = string(x)
#end

###########################
### get_expr

function get_expr(tree::DerivationTree) #entry
  return iscomplete(tree) ? get_expr(tree.root) : throw(IncompleteException())
end

get_expr(node::DerivTreeNode) = get_expr(node, node.rule)
get_expr(node::DerivTreeNode, rule::Terminal) = rule.value

function get_expr(node::DerivTreeNode, rule::RangeRule)
  value = ((node.action - 1) % length(rule.range)) + rule.range.start

  if rule.action != nothing
    value = rule.action(value)
  end

  return value
end

function get_expr(node::DerivTreeNode, rule::OrRule)
  child = node.children[1]
  value = get_expr(child, child.rule)

  if rule.action != nothing
    value = rule.action(value)
  end

  return value
end

function get_expr(node::DerivTreeNode, rule::RepeatedRule)
  values = Any[]
  for child in node.children
    push!(values, get_expr(child, child.rule))
  end

  if rule.action !== nothing
    values = rule.action(values)
  end

  return values
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
  return Expr(xs...)
end

###########################
###
function actionspace(tree::DerivationTree)
  if isempty(tree.opennodes)
    return 0:-1 #special -1 action for terminal state
  end
  node = top(tree.opennodes)
  return actionspace(node, node.rule)
end

actionspace(node::DerivTreeNode, rule::OrRule) = 1:length(rule.values)
actionspace(node::DerivTreeNode, rule::RangeRule) = 1:length(rule.range)
actionspace(node::DerivTreeNode, rule::RepeatedRule) = 1:length(rule.range)

###########################
###
Base.empty!(stack::Stack) = empty!(stack.store)

#entry
maxlength(grammar::Grammar) = reduce(max, 0, map(length, values(grammar.rules)))

#decision rules
Base.length(rule::OrRule) = length(rule.values)
Base.length(rule::RangeRule) = length(rule.range)
#other rules
Base.length(rule::Rule) = 1

function play!(tree::DerivationTree, ind::ExampleIndividual)
  return play!(tree, ind.genome + 1) #genome is zero-indexed, actions are 1-indexed
end

function play!(tree::DerivationTree, actions::Vector{Int64})
  initialize!(tree)
  n = 0
  while !isterminal(tree)
    n += 1
    step!(tree, actions[n])
  end
  return n #number of actions actually used
end

start(tree::DerivationTree) = start(tree.root)
next(tree::DerivationTree, stack::Stack) = next(tree.root, stack)
done(tree::DerivationTree, stack::Stack) = done(tree.root, stack)

#iterate through nodes in depth-first search order
function start(startnode::DerivTreeNode)
  stack = Stack(DerivTreeNode, STACKSIZE)
  push!(stack, startnode)
  return stack
end

function next(startnode::DerivTreeNode, stack::Stack)
  node = pop!(stack)
  for i = length(node.children):-1:1
    push!(stack, node.children[i])
  end
  return node, stack
end

done(startnode::DerivTreeNode, stack::Stack) = isempty(stack)

count(tree::DerivationTree) = count(tree.root)
count(node::DerivTreeNode) = count(x -> true, node)
count(f::Function, tree::DerivationTree) = count(f::Function, tree.root)

function count(f::Function, node::DerivTreeNode)
  n = f(node)
  for child in node.children
    n += count(f, child)
  end
  return n
end

#size of subtree / count number of nodes in tree/subtree (recursive implementation)
size(tree::DerivationTree) = count(tree)

#uniformly randomly select a node from tree
rand(tree::DerivationTree) = rand(tree.root)

#uniformly randomly select a node from subtree using reservoir sampling
function rand(startnode::DerivTreeNode)
  current = startnode
  n = 1
  for node in startnode
    if rand() <= 1/n
      current = node
    end
    n += 1
  end
  return current
end

#generate a random tree, returns true if complete tree was attained
function rand!(tree::DerivationTree, retries::Int64=5)
  while retries > 0
    initialize!(tree)
    while !isterminal(tree)
      as = actionspace(tree)
      step!(tree, rand(as))
    end
    if iscomplete(tree)
      return true
    end
    retries -= 1
  end
  return false
end

#drop subtree starting at given node and init derivation starting from here
function initialize!(tree::DerivationTree, startnode::DerivTreeNode)
  tree.nsteps -= count(x -> isa(x.rule, DecisionRule), startnode) #undo subtree effect on nsteps
  return2pool(tree.nodepool, startnode.children) #discard old subtree
  empty!(tree.opennodes)

  push!(tree.opennodes, startnode)
  process_non_decisions!(tree)
  startnode
end

#generate a (new) random subtree starting at startnode, returns true if a complete tree was attained
function rand!(tree::DerivationTree, startnode::DerivTreeNode, retries::Int64=5)
  while retries > 0
    initialize!(tree, startnode)
    while !isterminal(tree)
      as = actionspace(tree)
      step!(tree, rand(as))
    end
    if iscomplete(tree)
      return true
    end
    retries -= 1
  end
  return false
end

function copy!(dst::DerivationTree, src::DerivationTree)
  reset!(dst)
  copy!(dst.root, src.root, dst.nodepool)
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

end #module
