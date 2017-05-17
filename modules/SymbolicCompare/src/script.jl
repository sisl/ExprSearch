using RLESUtils, RunUtils, IFTTTUtils

const COMP = gethostname()
const I_START = 1 
const I_END = 50
const NOTIFY = false

#uses default config

template_gp(i) =
"""
using ExprSearch,Symbolic_GP,RLESUtils,IFTTTUtils
symbolic_gp(;seed=$i, outdir=Pkg.dir("ExprSearch/results/SymbolicCompare/Symbolic_GP/seed$i"))
$NOTIFY && sendifttt(;value1="gp,$i", value2="$COMP")
"""

template_ge(i) =
"""
using ExprSearch,Symbolic_GE,RLESUtils,IFTTTUtils
symbolic_ge(;seed=$i, outdir=Pkg.dir("ExprSearch/results/SymbolicCompare/Symbolic_GE/seed$i"))
$NOTIFY && sendifttt(;value1="ge,$i", value2="$COMP")
"""

template_mc(i) =
"""
using ExprSearch,Symbolic_MC,RLESUtils,IFTTTUtils
symbolic_mc(;seed=$i, outdir=Pkg.dir("ExprSearch/results/SymbolicCompare/Symbolic_MC/seed$i"))
$NOTIFY && sendifttt(;value1="mc,$i", value2="$COMP")
"""

template_mcts(i) =
"""
using ExprSearch,Symbolic_MCTS,RLESUtils,IFTTTUtils
symbolic_mcts(;seed=$i, outdir=Pkg.dir("ExprSearch/results/SymbolicCompare/Symbolic_MCTS/seed$i"))
$NOTIFY && sendifttt(;value1="mcts,$i", value2="$COMP")
"""

template_ce(i) =
"""
using ExprSearch,Symbolic_CE,RLESUtils,IFTTTUtils
symbolic_ce(;seed=$i, outdir=Pkg.dir("ExprSearch/results/SymbolicCompare/Symbolic_CE/seed$i"))
$NOTIFY && sendifttt(;value1="ce,$i", value2="$COMP")
"""

notifydone() = sendifttt(;value1="$COMP done")

A = JuliaSource[]
append!(A, [JuliaSource(template_ce(i)) for i=I_START:I_END])
append!(A, [JuliaSource(template_ge(i)) for i=I_START:I_END])
append!(A, [JuliaSource(template_gp(i)) for i=I_START:I_END])
append!(A, [JuliaSource(template_mc(i)) for i=I_START:I_END])
#append!(A, [JuliaSource(template_mcts(i)) for i=I_START:I_END])

#include("script.jl")
#fs=julia_process(A,np); notifydone()
