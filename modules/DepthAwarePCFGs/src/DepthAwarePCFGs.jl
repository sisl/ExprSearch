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
Depth aware Probabilistic Context-Free Grammar
Wraps a PCFG and adds depth information 
"""
module DepthAwarePCFGs

export DepthAwarePCFG, fit_mle!, weighted_sum!

using GrammaticalEvolution
using ExprSearch, LinearDerivTrees, PCFGs, MinDepths
using RLESUtils, StatUtils

import PCFGs: fit_mle!, weighted_sum!
import Base: rand, rand!, normalize!, fill!, copy

typealias CFG Grammar
typealias ProbDict Dict{Symbol,Vector{Float64}}

type DepthAwarePCFG
    pcfg::PCFG
    depths_by_rule::MinDepthByRule 
    depths_by_action::MinDepthByAction 
end

"""
DepthAwarePCFG constructor 
"""
function DepthAwarePCFG(cfg::CFG)
    DepthAwarePCFG(
        PCFG(cfg),
        min_depth_rule(cfg),
        min_depth_actions(cfg)
        )
end

"""
Draw a random sample from depth-aware pcfg.  The output is in-place
into tree.
"""
function rand!(tree::LinearDerivTree, dpcfg::DepthAwarePCFG, maxdepth::Int64)
    rand!(Base.GLOBAL_RNG, tree, dpcfg, maxdepth)
end
function rand!(rng::AbstractRNG, tree::LinearDerivTree, dpcfg::DepthAwarePCFG, 
    maxdepth::Int64)
    probs = dpcfg.pcfg.probs
    initialize!(tree)
    while !isdone(tree)
        actions = actionspace(tree)
        node = current_opennode(tree)
        depths = dpcfg.depths_by_action[Symbol(node.rule.name)] 
        w = probs[get_sym(tree)]
        ids = find(x->x <= maxdepth-node.depth, depths)
        @assert !isempty(ids) #shouldn't be empty if started sampling from root
        aid = weighted_rand(rng, ids, w[ids])
        step!(tree, actions[aid])
    end
    iscomplete(tree)
end

"""
Draw N random samples from depth-aware pcfg.  Derivation trees are limited to depth of maxdepth.
Defaults to unlimited number of steps (non-terminal expansions)
Outputs a vector of LinearDerivTrees.
"""
function rand(dpcfg::DepthAwarePCFG, N::Int64, maxdepth::Int64)
    rand(Base.GLOBAL_RNG, dpcfg, N, maxdepth)
end
function rand(rng::AbstractRNG, dpcfg::DepthAwarePCFG, N::Int64, maxdepth::Int64;
    maxsteps::Int64=typemax(Int64))
    params = LDTParams(dpcfg.pcfg.cfg, maxsteps)
    samples = Array(LinearDerivTree, N)
    for i = 1:N
        s = LinearDerivTree(params)
        rand!(rng, s, dpcfg, maxdepth)
        samples[i] = s
    end
    samples
end

"""
Draw random samples according to pcfg and output them in-place into samples.
"""
function rand!(samples::Vector{LinearDerivTree}, dpcfg::DepthAwarePCFG, maxdepth::Int64)
    rand!(Base.GLOBAL_RNG, samples, dpcfg, maxdepth)
end
function rand!(rng::AbstractRNG, samples::Vector{LinearDerivTree}, dpcfg::DepthAwarePCFG,
    maxdepth::Int64)
    for i = 1:length(samples)
        rand!(rng, samples[i], dpcfg, maxdepth)
    end
end

"""
Compute the maximum likelihood estimator from all the transitions in the
trees in samples and store the output in-place into pcfg.
"""
function fit_mle!(dpcfg::DepthAwarePCFG, samples::Vector{LinearDerivTree})
    #passthrough
    fit_mle!(dpcfg.pcfg, samples)
end

"""
Computes the weighted sum of two pcfgs and returns the output in-place
pcfg = w1 * pcfg + w2 * pcfg2
"""
function weighted_sum!(dpcfg::DepthAwarePCFG, w1::Float64, dpcfg2::DepthAwarePCFG, w2::Float64)
    #passthrough
    weighted_sum!(dpcfg.pcfg, w1, dpcfg2.pcfg, w2)
end

"""
Fill all probabilities in pcfg to value x
"""
function fill!(dpcfg::DepthAwarePCFG, x)
    #passthrough
    fill!(dpcfg.pcfg, x)
end

"""
Normalize the sum of each probability vector in pcfg
"""
normalize!(dpcfg::DepthAwarePCFG) = normalize!(dpcfg.pcfg)

function copy(dpcfg::DepthAwarePCFG)
    DepthAwarePCFG(copy(dpcfg.pcfg), dpcfg.depths_by_rule, dpcfg.depths_by_action)
end

end #module

