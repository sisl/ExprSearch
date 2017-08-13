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

module GBDTs

export GBDTParams, GBDTResult, induce_tree, apply_expr, classify

using ExprSearch, DecisionTrees
using RLESUtils, DataFrameSets, GitUtils, CPUTimeUtils, LogSystems, Observers, 
    TreeIterators
using JLD

import RLESTypes.SymbolTable
import DecisionTrees: get_truth, get_splitter, get_split_labels, classify

#user should implemented these for their problem
apply_expr(problem::ExprProblem, ids::Vector{Int64}, expr) = 
    error("user should override apply_expr()")

include("logdefs.jl")

type GBDTParams
    problem::ExprProblem
    num_data::Int64
    split_params::SearchParams
    maxdepth::Int64
    predict_type::Type #T1
    label_type::Type #T2
    logsys::LogSystem
end
GBDTParams(problem::ExprProblem, num_data::Int64, split_params::SearchParams, 
    maxdepth::Int64, predict_type::Type, label_type::Type) = 
    GBDTParams(problem, split_params,
    maxdepth, predict_type, label_type, logsystem())

type GBDTParamsSerial
    num_data::Int64
    split_params::SearchParams
    maxdepth::Int64
    predict_type::Type #T1
    label_type::Type #T2
end

function JLD.writeas(p::GBDTParams)
    GBDTParamsSerial(p.num_data, p.split_params, p.maxdepth, p.predict_type, p.label_type)
end

type GBDTResult
    params::GBDTParams 
    dt_params::DTParams #need this for classify to work
    decision_tree::DecisionTree
end

function induce_tree(p::GBDTParams)
    @notify_observer(p.logsys.observer, "verbose1", ["Starting decision tree induction..."])
    @notify_observer(p.logsys.observer, "computeinfo", ["starttime", string(now())])

    tstart = CPUtime_start()
    dt_userargs = SymbolTable(:params=>p, :decision_id=>0)
    dt_params = DTParams(p.num_data, p.maxdepth, p.predict_type, p.label_type, dt_userargs)
    dtree = build_tree(dt_params)
    result = GBDTResult(p, dt_params, dtree)
    
    @notify_observer(p.logsys.observer, "result", [result])

    for node in tree_iter(dtree.root)
        @notify_observer(p.logsys.observer, "members", [node.id, string(node.members)])
    end

    #meta info
    @notify_observer(p.logsys.observer, "computeinfo", ["endtime",  string(now())])
    @notify_observer(p.logsys.observer, "computeinfo", ["hostname", gethostname()])
    @notify_observer(p.logsys.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
    @notify_observer(p.logsys.observer, "computeinfo", ["cpu_time",     
        string(CPUtime_elapsed_s(tstart))])
    @notify_observer(p.logsys.observer, "parameters", ["num_data", p.num_data])
    @notify_observer(p.logsys.observer, "parameters", ["maxdepth", p.maxdepth])
    @notify_observer(p.logsys.observer, "parameters", ["predict_type", string(p.predict_type)])
    @notify_observer(p.logsys.observer, "parameters", ["label_type", string(p.label_type)])

    result
end

function get_truth(members::Vector{Int64}, dt_userargs::SymbolTable)
    p = dt_userargs[:params]
    labels(p.problem.data, members)
end

function get_splitter(members::Vector{Int64}, dt_userargs::SymbolTable)
    tstart = CPUtime_start()

    p = dt_userargs[:params]
    dt_userargs[:decision_id] += 1
    decision_id = dt_userargs[:decision_id]

    p.split_params.userargs[:ids] = members
    split_result = exprsearch(p.split_params, p.problem)

    @notify_observer(p.logsys.observer, "elapsed_cpu_s", [decision_id, CPUtime_elapsed_s(tstart)]) 
    @notify_observer(p.logsys.observer, "split_result_raw", [decision_id, split_result])
    @notify_observer(p.logsys.observer, "split_result", [decision_id, 
        string(get_expr(split_result)), get_fitness(split_result)])

    split_result
end

function get_split_labels(result::SearchResult, members::Vector{Int64}, 
    dt_userargs::SymbolTable) 

    p = dt_userargs[:params]
    expr = get_expr(result) 
    apply_expr(p.problem, members, expr)
end
function get_split_labels(result::SearchResult, records::AbstractVector{DataFrame}, 
    dt_userargs::SymbolTable) 

    p = dt_userargs[:params]
    expr = get_expr(result) 
    apply_expr(p.problem, records, expr)
end

function classify{T}(result::GBDTResult, v::Vector{T})
   [classify(result, v[i]) for i=1:length(v)] 
end
function classify(result::GBDTResult, x)
    classify(result.dt_params, result.decision_tree, x)
end

end #module
