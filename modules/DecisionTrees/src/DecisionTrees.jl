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
Generic decision tree based on callbacks
"""
module DecisionTrees

export DTParams, DecisionTree, DTNode, build_tree, classify, isleaf, get_max_depth, 
    get_members, count_leafs, count_nonleafs
export get_truth, get_splitter, get_split_labels #to be overridden
export update_members!

using StatsBase
using RLESUtils, TreeUtils, TreeIterators, MathUtils
import RLESTypes: SymbolTable
import Base: length
import TreeIterators.get_children

type DTParams
    num_data::Int64 #number of datapoints in training set
    maxdepth::Int64
    predict_type::Type #T1
    label_type::Type #T2
    userargs::SymbolTable #additional variables in callabacks
    id::Int64 #hack, use this to store state of node id's assigned, init to 0

    DTParams(num_data::Int64, maxdepth::Int64, predict_type::Type, label_type::Type,
        userargs::SymbolTable) = new(num_data, maxdepth, predict_type, label_type,
        userargs, 0) #initialize id=0
end

type DTNode{T1,T2}
    id::Int64
    depth::Int64
    members::Vector{Int64} #indices into data starting at 1
    split_rule::Any #object used in callback for split rule
    children::Dict{T1,DTNode{T1,T2}} #key=split_rule predicts, value=child node, T1=predict label type
    label::T2 #T2=label type
    confidence::Float64
end

type DecisionTree{T1,T2}
    root::DTNode{T1,T2}
end

#User should override these functions
get_truth() = error("get_truth not defined")
get_splitter() = error("get_splitter not defined")
get_split_labels() = error("get_split_labels not defined")

get_children(node::DTNode) = collect(values(node.children))

function build_tree(p::DTParams)
    members = collect(1:p.num_data)
    depth = 0
    root = process_child(p, members, depth)
    DecisionTree(root)
end

function process_child(p::DTParams, members::Vector{Int64}, depth::Int64)
    p.id += 1 #node id's are ordered depth-first
    id = p.id
    members_copy = deepcopy(members)
    labels = get_truth(members, p.userargs)
    label = mode(labels)
    confidence = proportion(labels, label) #proportion of times label appears
    T1 = p.predict_type
    T2 = p.label_type
    @assert issubtype(eltype(labels), T2)
    children = Dict{T1,DTNode{T1,T2}}()
    if confidence == 1.0 || depth >= p.maxdepth #all outputs same, or reached stopping criterion
        split_rule = nothing
    else #consider splitting
        split_rule = get_splitter(members, p.userargs)
        if split_rule != nothing #get_splitter returns nothing if can't find a good split
            predict_labels = get_split_labels(split_rule, members, p.userargs)
            @assert issubtype(eltype(predict_labels), T1)
            predict_set = sort(unique(predict_labels))
            if length(predict_set) > 1 #continue only if there's a split
                child_nodes = map(predict_set) do label_class 
                    child_members = members[find(x -> x == label_class, predict_labels)]
                    child = process_child(p, child_members, depth + 1)
                    return child
                end
                for (label_class, child_node) in zip(predict_set, child_nodes)
                    children[label_class] = child_node
                end
            end
        end
    end
    DTNode{T1,T2}(id, depth, members_copy, split_rule, children, label, confidence)
end

classify{T1,T2,T3}(p::DTParams, tree::DecisionTree{T1,T2}, record::T3) = 
    classify(p, tree.root, record) 

function classify{T1,T2,T3}(p::DTParams, node::DTNode{T1,T2}, record::T3)
    if isempty(node.children) #leaf node
        return node.label::T2
    end
    label = get_split_labels(node.split_rule, [record], p.userargs)[1]
    child = node.children[label]
    return classify(p, child, record)
end

isleaf{T1,T2}(node::DTNode{T1,T2}) = isempty(node.children)

get_max_depth{T1,T2}(dtree::DecisionTree{T1,T2}) = get_max_depth(dtree.root)
function get_max_depth{T1,T2}(node::DTNode{T1,T2})
    if isleaf(node)
        return node.depth
    else
        return map(get_max_depth, values(node.children)) |> maximum
    end
end

get_members{T1,T2}(tree::DecisionTree{T1,T2}) = get_members(tree.root)
get_members{T1,T2}(node::DTNode{T1,T2}) = node.members

count_nonleafs{T1,T2}(tree::DecisionTree{T1,T2}) = 
    count_nodes(x->!DecisionTrees.isleaf(x), tree.root)

count_leafs{T1,T2}(tree::DecisionTree{T1,T2}) = 
    count_nodes(DecisionTrees.isleaf, tree.root)

length{T1,T2}(tree::DecisionTree{T1,T2}) = count_nodes(tree.root)

function update_members!{T1,T2}(tree::DecisionTree{T1,T2}, members::Vector{Int64}, 
    userargs::SymbolTable) 
    update_members!(tree.root, members, userargs)
end
function update_members!{T1,T2}(node::DTNode{T1,T2}, members::Vector{Int64},
    userargs::SymbolTable)
    resize!(node.members, length(members))
    copy!(node.members, members)
    if isempty(node.children) #leaf node
        return
    end
    labels = get_split_labels(node.split_rule, members, userargs)
    label_set = sort(unique(labels))
    for l in label_set 
        child_members = members[find(x -> x == l, labels)]
        update_members!(node.children[l], child_members, userargs)
    end
end

end #module
