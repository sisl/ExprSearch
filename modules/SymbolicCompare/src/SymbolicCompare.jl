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
SymbolicRegression Study comparing performance of GP, MC, MCTS, GE, CE.
Single-threaded versions are used for more stable comparison.
use script.jl
then combine_and_plot()
"""
module SymbolicCompare

export combine_mc_logs, combine_mcts_logs, 
    combine_ge_logs, combine_gp_logs, combine_ce_logs, combine_logs
export master_log, master_plot
export combine_and_plot, set_studyname

using ExprSearch: GP, MC, MCTS, GE, CE
using Symbolic_GP, Symbolic_GE, Symbolic_MC, Symbolic_MCTS, Symbolic_CE 
using RLESUtils, Loggers, MathUtils, LatexUtils, Sweeper, LogJoiner

using DataFrames
using PGFPlots, TikzPictures

STUDYNAME = "SymbolicCompare"
const MC_NAME = "Symbolic_MC"
const MCTS_NAME = "Symbolic_MCTS"
const GE_NAME = "Symbolic_GE"
const GP_NAME = "Symbolic_GP"
const CE_NAME = "Symbolic_CE"

const CONFIGDIR = joinpath(dirname(@__FILE__), "..", "config")
const RESULTDIR = joinpath(dirname(@__FILE__), "..", "..", "..", "results")

resultpath(dir::String="") = joinpath(RESULTDIR, dir)
studypath(dir::String="") = joinpath(RESULTDIR, STUDYNAME, dir)
_masterlog_file() = joinpath(RESULTDIR, STUDYNAME, "masterlog.csv.gz")
_plotlog_file() =  joinpath(RESULTDIR, STUDYNAME, "plotlog.csv.gz")
_plotfileroot() = joinpath(RESULTDIR, STUDYNAME, "plots")

set_studyname(studyname::AbstractString) = global STUDYNAME = studyname

function combine_mc_logs()
    dir = studypath(MC_NAME)
    logjoin(dir, "symbolic_mc_log.txt", ["current_best", "elapsed_cpu_s"], [:name, :nevals],
        joinpath(dir, "subdirjoined"))
end
function combine_mcts_logs()
    dir = studypath(MCTS_NAME)
    logjoin(dir, "symbolic_mcts_log.txt", ["current_best", "elapsed_cpu_s"], [:name, :nevals],
        joinpath(dir, "subdirjoined"))
end
function combine_ge_logs()
    dir = studypath(GE_NAME)
    logjoin(dir, "symbolic_ge_log.txt", ["current_best", "elapsed_cpu_s"], [:name, :nevals],
        joinpath(dir, "subdirjoined"))
end
function combine_gp_logs()
    dir = studypath(GP_NAME)
    logjoin(dir, "symbolic_gp_log.txt", ["current_best", "elapsed_cpu_s"], [:name, :nevals],
        joinpath(dir, "subdirjoined"))
end
function combine_ce_logs()
    dir = studypath(CE_NAME)
    logjoin(dir, "symbolic_ce_log.txt", ["current_best", "elapsed_cpu_s"], [:name, :nevals],
        joinpath(dir, "subdirjoined"))
end

#TODO: clean this up...
function master_log(; b_mc=true, b_mcts=true, b_ge=true, b_gp=true, b_ce=true)
    masterlog = DataFrame([Int64, Float64, Float64, String, String, String], 
        [:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name], 0)

    #MC
    if b_mc
        dir = studypath(MC_NAME)
        logs = load_log(LogFile(joinpath(dir, "subdirjoined.txt")))
        D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
        D[:algorithm] = fill("MC", nrow(D))
        rename!(D, :iter, :nevals)
        append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])
    end

    #MCTS
    if b_mcts
        dir = studypath(MCTS_NAME)
        logs = load_log(LogFile(joinpath(dir, "subdirjoined.txt")))
        D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:iter, :name])
        D[:algorithm] = fill("MCTS", nrow(D))
        rename!(D, :iter, :nevals)
        append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])
    end

    #GE
    if b_ge
        dir = studypath(GE_NAME)
        logs = load_log(LogFile(joinpath(dir, "subdirjoined.txt")))
        D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:nevals, :name])
        D[:algorithm] = fill("GE", nrow(D))
        append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])
    end

    #GP
    if b_gp
        dir = studypath(GP_NAME)
        logs = load_log(LogFile(joinpath(dir, "subdirjoined.txt")))
        D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:nevals, :name])
        D[:algorithm] = fill("GP", nrow(D))
        append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])
    end

    #CE
    if b_ce
        dir = studypath(CE_NAME)
        logs = load_log(LogFile(joinpath(dir, "subdirjoined.txt")))
        D = join(logs["elapsed_cpu_s"], logs["current_best"], on=[:nevals, :name])
        D[:algorithm] = fill("CE", nrow(D))
        append!(masterlog, D[[:nevals, :elapsed_cpu_s, :fitness, :expr, :algorithm, :name]])
    end

    writetable(_masterlog_file(), masterlog)
    masterlog
end

master_plot(; kwargs...) = master_plot(readtable(_masterlog_file()); kwargs...)

"""
Subsamples the collected data at 'subsample' rate to plot at a lower rate than collected
Set subsample to 1 to disable.
"""
function master_plot(masterlog::DataFrame; 
    plotlog_file::AbstractString=_plotlog_file(), plotfileroot::AbstractString=_plotfileroot(),
    subsample::Int64=2000)
    D = masterlog 

    #aggregate over seed
    D = aggregate(D[[:nevals, :elapsed_cpu_s, :fitness, :algorithm]], [:nevals, :algorithm], 
        [mean, std, length, SEM])
    D = D[rem(D[:nevals], subsample) .== 0, :] #subsample data

    #workaround for naming in julia 0.4
    rename!(D, Symbol("fitness_MathUtils.SEM"), :fitness_SEM) 
    rename!(D, Symbol("elapsed_cpu_s_MathUtils.SEM"), :elapsed_cpu_s_SEM) 

    writetable(plotlog_file, D)

    td = TikzDocument()
    algo_names = unique(D[:algorithm])
    n_algos = length(algo_names)

    #nevals_vs_fitness
    plotarray = Plots.Plot[]
    for i = 1:n_algos 
        D1 = D[D[:algorithm].==algo_names[i], [:nevals, :fitness_mean, :fitness_SEM]]
        push!(plotarray, Plots.Linear(D1[:nevals], D1[:fitness_mean], 
            errorBars=ErrorBars(; y=D1[:fitness_SEM]),   
            legendentry=escape_latex(algo_names[i])))
    end
    tp = PGFPlots.plot(Axis(plotarray, xlabel="Number of Evaluations", ylabel="Fitness",
        title="Fitness vs. Number of Evaluations", legendPos="north east"))
    push!(td, tp) 
    
    #nevals_vs_elapsed_cpu
    empty!(plotarray)
    for i = 1:n_algos
        D1 = D[D[:algorithm].==algo_names[i], [:nevals, :elapsed_cpu_s_mean, 
            :elapsed_cpu_s_SEM]]
        push!(plotarray, Plots.Linear(D1[:nevals], D1[:elapsed_cpu_s_mean], 
            errorBars=ErrorBars(;y=D1[:elapsed_cpu_s_SEM]), 
            legendentry=escape_latex(algo_names[i])))
    end
    tp = PGFPlots.plot(Axis(plotarray, xlabel="Number of Evaluations", ylabel="Elapsed CPU Time (s)",
        title="Elapsed CPU Time vs. Number of Evaluations", legendPos="north west"))
    push!(td, tp)

    #elapsed_cpu_vs_fitness
    empty!(plotarray)
    for i = 1:n_algos
        D1 = D[D[:algorithm].==algo_names[i], [:elapsed_cpu_s_mean, 
            :elapsed_cpu_s_SEM, :fitness_mean, :fitness_SEM]]
        push!(plotarray, Plots.Linear(D1[:elapsed_cpu_s_mean], D1[:fitness_mean], 
            errorBars=ErrorBars(;x=D1[:elapsed_cpu_s_SEM], y=D1[:fitness_SEM]), 
            legendentry=escape_latex(algo_names[i])))
    end
    tp = PGFPlots.plot(Axis(plotarray, xlabel="Elapsed CPU Time (s)", ylabel="Fitness",
        title="Fitness vs. Elapsed CPU Time", legendPos="north east"))
    push!(td, tp)

    save(PDF(plotfileroot * ".pdf"), td)
    save(TEX(plotfileroot * ".tex"), td)
end

function combine_and_plot(; subsample=2000, b_mc=true, b_mcts=true, b_ge=true, 
    b_gp=true, b_ce=true)
    b_ge && combine_ge_logs()
    b_gp && combine_gp_logs()
    b_mc && combine_mc_logs()
    b_mcts && combine_mcts_logs()
    b_ce && combine_ce_logs()
    ml = master_log(b_mc=b_mc, b_mcts=b_mcts, b_ge=b_ge, b_gp=b_gp, b_ce=b_ce)
    master_plot(ml; subsample=subsample)
end

end #module

