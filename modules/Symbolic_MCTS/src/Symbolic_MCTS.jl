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

module Symbolic_MCTS

export symbolic_mcts

using ExprSearch, SymbolicRegression, DerivTreeVis
using ExprSearch.MCTS
using RLESUtils, Observers, LogSystems, Loggers
using Base.Test

const DIR = dirname(@__FILE__)
const RESULTDIR = joinpath(DIR, "..", "..", "..", "results") 

"""
Example usage:
symbolic_mcts(; seed=1)
"""
function symbolic_mcts(;outdir::AbstractString=joinpath(RESULTDIR, "Symbolic_MCTS"),
                     seed=1,
                     logfileroot::AbstractString="symbolic_mcts_log",

                     n_iters::Int64=50000,
                     searchdepth::Int64=30,
                     explorationconst::Float64=2000.0,
                     q0::Float64=-1000.0,
                     maxsteps::Int64=20,
                     max_neg_reward::Float64=-1000.0,
                     step_reward::Float64=0.0,
                     maxmod::Bool=false, #use the max update mod

                     gt_file::AbstractString="gt_easy.jl",

                     loginterval::Int64=1000,
                     vis::Bool=true)
    srand(seed)
    mkpath(outdir)

    logsys = MCTS.logsystem()
    empty_listeners!(logsys)
    send_to!(STDOUT, logsys, ["verbose1", "result"])
    send_to!(STDOUT, logsys, "current_best"; interval=loginterval)
    logs = TaggedDFLogger()
    send_to!(logs, logsys, ["computeinfo", "parameters", "result"])
    send_to!(logs, logsys,  "current_best"; interval=loginterval)
    send_to!(logs, logsys,  "elapsed_cpu_s"; interval=loginterval)

    problem = Symbolic(gt_file)
    mcts_params = MCTSESParams(maxsteps, max_neg_reward, step_reward, n_iters, searchdepth,
                             explorationconst, maxmod, q0, seed, logsys)
    result = exprsearch(mcts_params, problem)

    outfile = joinpath(outdir, "$(logfileroot).txt")
    save_log(outfile, logs)

    if vis
        derivtreevis(get_derivtree(result), joinpath(outdir, "$(logfileroot)_derivtreevis"))
    end
    @show result.expr
    return result
end

end #module
