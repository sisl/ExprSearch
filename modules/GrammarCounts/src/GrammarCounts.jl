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

module GrammarCounts

export rulecounts, decisioncounts

using RLESUtils, SwapBuffers
using DerivationTrees

typealias CountDict Dict{Symbol,Float64}
typealias DecisionCounts Dict{Symbol,Vector{Float64}}

"""
Returns a Dict where key is symbol of decision rule, value is a vector of counts for each action
Example:
see runtests.jl
"""
function decisioncounts(grammar::Grammar, r_cnts::CountDict)
  d_cnts = DecisionCounts()
  for (name, rule) in grammar.rules
    if isa(rule, DecisionRule)
      d_cnts[name] = vec_count(grammar, rule, r_cnts)
    end
  end
  d_cnts
end

#vector of counts, by decision
function vec_count(grammar::Grammar, rule::OrRule, logcnts0::CountDict)
  [count(grammar, r, logcnts0) for r in rule.values]
end

function vec_count(grammar::Grammar, rule::RangeRule, logcnts0::CountDict)
  ones(length(rule.range))
end

#non-decisions
vec_count(grammar::Grammar, rule, logcnts0::CountDict) = count(grammar, rule, logcnts0)

using Debug
#"""
#Returns a Dict where key is symbol of grammar rule and value is counts
#"""
function rulecounts(grammar::Grammar, depth::Int64)
  logcntsA = initcounts(grammar)
  logcntsB = initcounts(grammar)
  sbuf = SwapBuffer(logcntsA, logcntsB)

  for d = 1:depth
    logcnts0 = active(sbuf) #active contains current
    logcnts1 = inactive(sbuf) #inactive contains next
    for (name, rule) in grammar.rules
      logcnts1[name] = count(grammar, rule, logcnts0)
    end
    @show logcnts1[:bin_vec], logcnts1[:and], logcnts1[:or]
    swap!(sbuf)
  end
  active(sbuf)
end

function initcounts(grammar::Grammar)
  logcnts = CountDict()
  for (k, v) in grammar.rules
    logcnts[k] = init(v)
  end
  logcnts
end

init(rule::OrRule) = 0
init(rule::ExprRule) = 0
init(rule::ReferencedRule) = 0

#terminals
init(rule::RangeRule) = 1.0 #log(e * length(rule.range))
init(rule::Terminal) = 1.0

function count(grammar::Grammar, rule::OrRule, logcnts0::CountDict)
  maximum(vec_count(grammar, rule, logcnts0)) #sum the counts --- approximate using max of the log
end

function count(grammar::Grammar, rule::ExprRule, logcnts0::CountDict)
  logcnt = 0.0
  for arg in rule.args
    if isa(arg, Rule)
      logcnt += count(grammar, arg, logcnts0)
    end
  end
  logcnt
end

function count(grammar::Grammar, rule::ReferencedRule, logcnts0::CountDict)
  logcnts0[rule.symbol] #passthrough
end

#terminals are not recursive
count(grammar::Grammar, rule::RangeRule, logcnts0::CountDict) = init(rule)
count(grammar::Grammar, rule::Terminal, logcnts0::CountDict) = init(rule)

end #module
