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
Computes the min depths by rule and action for a grammar
"""
module MinDepths

export MinDepthByRule, MinDepthByAction
export min_depth_rule, min_depth_actions 

using ExprSearch
using GrammaticalEvolution

typealias MinDepthByRule Dict{Symbol,Int64}
typealias MinDepthByAction Dict{Symbol,Vector{Int64}}

"""
Compute minimum depth for each rule
"""
function min_depth_rule(grammar::Grammar)
    d = MinDepthByRule()
    changed = Dict{Symbol,Bool}()
    for (k,v) in grammar.rules
        d[k] = typemax(Int64)/2
        changed[k] = true
    end
    while any(values(changed))
        for (k,rule) in grammar.rules 
           d_k = min_depth_rule(d, rule) 
           changed[k] = d[k] != d_k
           d[k] = d_k 
       end
    end
    d
end

min_depth_rule(d::MinDepthByRule, rule::ReferencedRule) = d[rule.symbol]
min_depth_rule(d::MinDepthByRule, rule::Union{RangeRule,Symbol}) = 0 #terminals
min_depth_rule(d::MinDepthByRule, x::Any) = 0 #terminals such as constants?
function min_depth_rule(d::MinDepthByRule, rule::OrRule)
    1 + minimum(map(r->min_depth_rule(d,r), rule.values))
end
function min_depth_rule(d::MinDepthByRule, rule::ExprRule)
    a = filter(r->isa(r,ReferencedRule), rule.args)
    1 + maximum(map(r->min_depth_rule(d,r), a))
end
function min_depth_rule(d::MinDepthByRule, rule::AndRule)
    a = filter(r->isa(r,ReferencedRule), rule.values)
    1 + maximum(map(r->min_depth_rule(d,r), a))
end

"""
Compute minimum depth per action of decision rule
"""
function min_depth_actions(grammar::Grammar)
    d = min_depth_rule(grammar)
    da = min_depth_actions(d, grammar)
    da
end
function min_depth_actions(d::MinDepthByRule, grammar::Grammar)
    da = MinDepthByAction()
    for (k,rule) in grammar.rules
        da[k] = min_depth_actions(d, rule)
    end
    da
end
min_depth_actions(d::MinDepthByRule, rule::ReferencedRule) = Int64[d[rule.symbol]]
min_depth_actions(d::MinDepthByRule, rule::Union{Symbol,Terminal}) = zeros(Int64, 1) 
min_depth_actions(d::MinDepthByRule, rule::RangeRule) = zeros(Int64, length(rule.range))
function min_depth_actions(d::MinDepthByRule, rule::OrRule)
    1 + Int64[min_depth_rule(d, v) for v in rule.values]
end
function min_depth_actions(d::MinDepthByRule, rule::Union{AndRule,ExprRule})
    Int64[min_depth_rule(d, rule)]
end

end #module
