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

import Compat.ASCIIString

using ExprSearch, DerivationTrees, LinearDerivTrees
using RLESUtils, Confusion, TreeIterators, TreeUtils
using DecisionTrees

function logsystem()
    logsys = LogSystem()

    register_log!(logsys, "parameters", ["parameter", "value"], [ASCIIString, Any])
    register_log!(logsys, "computeinfo", ["parameter", "value"], 
        [ASCIIString, ASCIIString]) 
    register_log!(logsys, "elapsed_cpu_s", ["decision_id", "CPUtime_s"], 
        [Int64, Float64]) 
    register_log!(logsys, "split_result", ["decision_id", "expr", "fitness"], 
        [Int64, ASCIIString, Float64]) 
    register_log!(logsys, "members", ["decision_id", "members"], [Int64, ASCIIString]) 
    register_log!(logsys, "classifier_metrics", ["truepos", "trueneg", "falsepos",
        "falseneg", "precision", "recall", "accuracy", "f1_score"], 
        [Int64, Int64, Int64, Int64, Float64, Float64, Float64, Float64], "result",
        x->begin
            result = x[1]
            members = get_members(result.decision_tree)
            pred = map(x->classify(result.dt_params, result.decision_tree, x), members)
            pred = pred .== 1 
            truth = get_truth(members, result.dt_params.userargs)
            truth = truth .== 1
            conf_mat = ConfusionMat(pred, truth)
            [conf_mat.truepos, conf_mat.trueneg, conf_mat.falsepos, conf_mat.falseneg, 
                precision(conf_mat), recall(conf_mat), accuracy(conf_mat), 
                f1_score(conf_mat)]
        end)
    register_log!(logsys, "interpretability_metrics", ["num_rules", "avg_rule_length",
        "num_nodes", "num_leaf", "avg_deriv_tree_num_nodes", "avg_deriv_tree_num_leafs"], 
        [Int64, Float64, Int64, Int64, Float64, Float64], "result",
        x->begin
            result = x[1]
            #decision tree
            num_rules = DecisionTrees.count_nonleafs(result.decision_tree)
            num_leaf = DecisionTrees.count_leafs(result.decision_tree)
            sum_rule_length = traverse(x->begin
                    x.split_rule != nothing ? length(string(x.split_rule.expr)) : 0
                end, 
                +, result.decision_tree.root)
            avg_rule_length = sum_rule_length / (num_rules + num_leaf)

            #each derivation tree
            num_derivnodes = Int64[] 
            num_derivleafs = Int64[]
            for node in tree_iter(result.decision_tree.root)
                if node.split_rule != nothing
                    push!(num_derivnodes, length(get_derivtree(node.split_rule)))
                    push!(num_derivleafs, DerivationTrees.count_leafs(
                        get_derivtree(node.split_rule)))
                end
            end
            avg_deriv_num_nodes = mean(num_derivnodes)
            avg_deriv_num_leafs = mean(num_derivleafs)
            [num_rules, avg_rule_length, (num_rules + num_leaf), num_leaf, avg_deriv_num_nodes,
                avg_deriv_num_leafs]
        end)

    register_log!(logsys, "verbose1", ["msg"], [ASCIIString])
    register_log!(logsys, "split_result_print", ["msg"], [ASCIIString], "split_result_raw", 
        x->begin
            decision_id, result = x 
            fitness = result.fitness
            expr = string(get_expr(result))
            expr_short = take(expr, 50) |> join
            ["decision_id=$(decision_id), fitness=$(signif(fitness, 4)), expr=$(expr_short)"]
        end)

    logsys
end
