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
Symbolic Study comparing performance of CE under different parameters
Usage:
Use script.jl
then combine_and_plot()
"""
module Symbolic_CE_ParamStudy

export combine_logs
export master_log, master_plot
export combine_and_plot

import Compat: ASCIIString, UTF8String

using RLESUtils, Loggers, MathUtils, LatexUtils, Sweeper, LogJoiner
using DataFrames
using PGFPlots, TikzPictures

const STUDYNAME = "Symbolic_CE_ParamStudy"

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")
const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")

const MASTERLOG_FILE = joinpath(RESULTDIR, STUDYNAME, "masterlog.csv.gz")
const PLOTLOG_FILE =  joinpath(RESULTDIR, STUDYNAME, "plotlog.csv.gz")
const PLOTFILEROOT = joinpath(RESULTDIR, STUDYNAME, "plots")


const PARAM_COLS = [:num_samples, :iterations, :elite_frac, :w_new, :w_prior, :maxsteps]
const REDUCE_COLS = [:seed, :fitness]
const MASTER_COLS = vcat(PARAM_COLS, REDUCE_COLS)

resultpath(dir::ASCIIString="") = joinpath(RESULTDIR, dir)
studypath(dir::ASCIIString="") = joinpath(RESULTDIR, STUDYNAME, dir)

function combine_logs()
    dir = studypath()
    logs = logjoin(dir, "symbolic_ce_log.txt", ["result", "parameters"], joinpath(dir, "joined"); 
        transpose_syms=Union{Symbol,Void}[nothing, :parameter])
    logs
end

function master_log()
    dir = studypath()
    logs = load_log(TaggedDFLogger, joinpath(dir, "joined.txt"))
    master_log(logs)
end
function master_log(logs::TaggedDFLogger)
    masterlog = join(logs["parameters"], logs["result"], on=:name)
    writetable(MASTERLOG_FILE, masterlog)
    masterlog
end

master_plot(; kwargs...) = master_plot(readtable(MASTERLOG_FILE); kwargs...)
"""
Subsamples the collected data at 'subsample' rate to plot at a lower rate than collected
"""
function master_plot(masterlog::DataFrame)
    D = masterlog 

    #aggregate over seed
    D = aggregate(D[MASTER_COLS], PARAM_COLS, 
        [mean, std, length, SEM]) #operations applied over :seed and :fitness

    #workaround for naming in julia 0.4
    rename!(D, Symbol("fitness_MathUtils.SEM"), :fitness_SEM) 
    rename!(D, Symbol("seed_MathUtils.SEM"), :seed_SEM) 

    writetable(PLOTLOG_FILE, D)

    #= td = TikzDocument() =#
    #= algo_names = unique(D[:algorithm]) =#
    #= n_algos = length(algo_names) =#
    #=  =#
    #= #nevals_vs_fitness =#
    #= plotarray = Plots.Plot[] =#
    #= for i = 1:n_algos  =#
    #=     D1 = D[D[:algorithm].==algo_names[i], [:nevals, :fitness_mean, :fitness_SEM]] =#
    #=     push!(plotarray, Plots.Linear(D1[:nevals], D1[:fitness_mean],  =#
    #=         errorBars=ErrorBars(; y=D1[:fitness_SEM]),    =#
    #=         legendentry=escape_latex(algo_names[i]))) =#
    #= end =#
    #= tp = PGFPlots.plot(Axis(plotarray, xlabel="Number of Evaluations", ylabel="Fitness", =#
    #=     title="Fitness vs. Number of Evaluations", legendPos="north east")) =#
    #= push!(td, tp)  =#
    #=  =#
    #= #nevals_vs_elapsed_cpu =#
    #= empty!(plotarray) =#
    #= for i = 1:n_algos =#
    #=     D1 = D[D[:algorithm].==algo_names[i], [:nevals, :elapsed_cpu_s_mean,  =#
    #=         :elapsed_cpu_s_SEM]] =#
    #=     push!(plotarray, Plots.Linear(D1[:nevals], D1[:elapsed_cpu_s_mean],  =#
    #=         errorBars=ErrorBars(;y=D1[:elapsed_cpu_s_SEM]),  =#
    #=         legendentry=escape_latex(algo_names[i]))) =#
    #= end =#
    #= tp = PGFPlots.plot(Axis(plotarray, xlabel="Number of Evaluations", ylabel="Elapsed CPU Time (s)", =#
    #=     title="Elapsed CPU Time vs. Number of Evaluations", legendPos="north west")) =#
    #= push!(td, tp) =#
    #=  =#
    #= save(PDF(PLOTFILEROOT * ".pdf"), td) =#
    #= save(TEX(PLOTFILEROOT * ".tex"), td) =#
end

function combine_and_plot()
    logs = combine_logs()
    mlog = master_log(logs)
    plog = master_plot() 
end

end #module

