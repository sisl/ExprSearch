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

module Symbolic_CE

export symbolic_ce

using ExprSearch, SymbolicRegression, DerivTreeVis
using ExprSearch.CE
using RLESUtils, LogSystems, Loggers
using Base.Test

const DIR = dirname(@__FILE__)
const RESULTDIR = joinpath(DIR, "..", "..", "..", "results") 

"""
Example usage:
symbolic_ce(; seed=1)
"""
function symbolic_ce(;outdir::AbstractString=joinpath(RESULTDIR, "Symbolic_CE"),
                     seed=1,
                     logfileroot::AbstractString="symbolic_ce_log",

                     num_samples::Int64=2000,
                     iterations::Int64=50,
                     elite_frac::Float64=0.6,
                     w_new::Float64=0.4,
                     w_prior::Float64=0.1,
                     maxdepth::Int64=10,
                     randchannel_width::Int64=60,
                     default_code::Any=0.0,

                     gt_file::AbstractString="gt_easy.jl",

                     vis::Bool=true,
                     vis_type::AbstractString="TEX"
                     )
    srand(seed)
    mkpath(outdir)

    logsys = CE.logsystem()
    empty_listeners!(logsys)
    send_to!(STDOUT, logsys, ["verbose1", "current_best_print", "result"])
    logs = TaggedDFLogger()
    send_to!(logs, logsys, ["code", "computeinfo", "current_best", "elapsed_cpu_s", "fitness",
        "parameters", "result"])

    problem = Symbolic(gt_file)

    ce_params = CEESParams(num_samples, iterations, elite_frac, w_new, 
        w_prior, maxdepth, default_code, randchannel_width, logsys)

    result = exprsearch(ce_params, problem)

    #manually push! extra info to log
    push!(logs, "parameters", ["seed", seed])
    push!(logs, "parameters", ["gt_file", gt_file])

    outfile = joinpath(outdir, "$(logfileroot).txt")
    save_log(outfile, logs)

    if vis
        derivtreevis(get_derivtree(result), joinpath(outdir, "$(logfileroot)_derivtreevis");
            output=vis_type)
    end
    @show result.expr
    return result
end

end #module
