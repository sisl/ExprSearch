using Iterators
using RLESUtils, RunUtils, IFTTTUtils

const COMP = gethostname()

V_seeds = collect(1:20) 
V_efs = [0.1, 0.25, 0.5, 0.6]
V_wnews = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
V_wprior = [0.01, 0.05, 0.1, 0.2, 0.4]
V_maxsteps = [20, 40, 60]

template_ce(seed, elite_frac, w_new, w_prior, maxsteps) =
"""
using ExprSearch,Symbolic_CE,RLESUtils,IFTTTUtils
config=Dict{Symbol,Any}()
config[:seed]=$seed
config[:elite_frac]=$(elite_frac)
config[:w_new]=$(w_new)
config[:w_prior]=$(w_prior)
config[:maxsteps]=$maxsteps
symbolic_ce(; outdir=Pkg.dir("ExprSearch/results/Symbolic_CE_ParamStudy/seed$(seed)ef$(elite_frac)wnew$(w_new)wprior$(w_prior)maxsteps$maxsteps"), config...)
"""

notifydone() = sendifttt(;value1="$COMP done")

A = JuliaSource[]
for args in product(V_seeds, V_efs, V_wnews, V_wprior, V_maxsteps)
    push!(A, JuliaSource(template_ce(args...)))
end

#include("script.jl")
#fs=julia_process(A, np); notifydone()
