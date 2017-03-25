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
Probabilistic Context-Free Grammar
Wraps a CFG and adds a parallel data structure for the probabilities
"""
module PCFGs

export PCFG, fit_mle!, weighted_sum!

using GrammaticalEvolution
using LinearDerivTrees
using AbstractTrees
using RLESUtils, StatUtils
import Base: rand, rand!, getindex, setindex!, normalize!, fill!, copy

typealias CFG Grammar
typealias ProbDict Dict{Symbol,Vector{Float64}}

type PCFG
    cfg::CFG
    probs::ProbDict #probability vector for each Or rule
end

"""
PCFG constructor that initializes all probabilities to uniform
"""
function PCFG(cfg::CFG)
    probs = ProbDict()
    for (sym,rule) in cfg.rules
        if isa(rule, OrRule) || isa(rule, RangeRule)
            probs[sym] = uniform_probs(length(rule))
        end
    end
    PCFG(cfg, probs)
end

getindex(pcfg::PCFG, index) = pcfg.probs[index]
setindex!(pcfg::PCFG, index, val) = pcfg.probs[index] = val

"""
Draw a random sample from pcfg.  The output is in-place
into tree.
"""
function rand!(tree::LinearDerivTree, pcfg::PCFG)
    probs = pcfg.probs
    initialize!(tree)
    while !isdone(tree)
        as = actionspace(tree)
        sym = get_sym(tree)
        step!(tree, weighted_rand(as, probs[sym]))
    end
    iscomplete(tree)
end

"""
Draw random samples according to pcfg and output them in-place into samples.
"""
function rand!(samples::Vector{LinearDerivTree}, pcfg::PCFG)
    for s in samples
        rand!(s, pcfg)
    end
end

"""
Draw N random samples from pcfg.  Derivation trees are limited to maxsteps.
Outputs a vector of LinearDerivTrees.
"""
function rand(pcfg::PCFG, N::Int64, maxsteps::Int64)
    params = LDTParams(pcfg.cfg, maxsteps)
    samples = Array(LinearDerivTree, N)
    for i = 1:N
        s = LinearDerivTree(params)
        rand!(s, pcfg, maxsteps)
        samples[i] = s
    end
    samples
end

"""
Compute the maximum likelihood estimator from all the transitions in the
trees in samples and store the output in-lce into pcfg.
"""
function fit_mle!(pcfg::PCFG, samples::Vector{LinearDerivTree})
    fill!(pcfg, 0.0)
    for s in samples
        for node in PreOrderDFS(s.derivtree.root)
            if isa(node.rule, DecisionRule)
                sym = Symbol(node.rule.name)
                a = node.action
                pcfg[sym][a] += 1.0
            end
        end
    end
    normalize!(pcfg)
end

"""
Computes the weighted sum of two pcfgs and returns the output in-place
pcfg = w1 * pcfg + w2 * pcfg2
"""
function weighted_sum!(pcfg::PCFG, w1::Float64, pcfg2::PCFG, w2::Float64)
    for (k,v1) in pcfg.probs
        v2 = pcfg2.probs[k] 
        v1[:] = w1*v1 + w2*v2
    end
end

"""
Fill all probabilities in pcfg to value x
"""
function fill!(pcfg::PCFG, x)
    for v in values(pcfg.probs)
        fill!(v, x)
    end
end

"""
Normalize the sum of each probability vector in pcfg
"""
function normalize!(pcfg::PCFG)
    for v in values(pcfg.probs)
        s = sum(v)
        if s > 0.0 #make sure valid denom
            map!(x->x/s, v)
        else #set to uniform, catches NaNs and 0.0
            len = length(v)
            fill!(v, 1.0/len)
        end
    end
end

function copy(pcfg::PCFG)
    PCFG(pcfg.cfg, deepcopy(pcfg.probs))
end

end #module
