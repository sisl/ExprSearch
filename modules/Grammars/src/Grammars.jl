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
Collection of generic grammars
"""
module Grammars

using GrammaticalEvolution
using RLESUtils
using ExprSearch, DerivationTrees

import RLESTypes.SymbolTable

include("operators.jl")

function time_series_realonly1(real_feat_ids::Vector{Int64}, num_real_vals::Int64, 
    real_vals::Vector, colnames::Vector{String}, colnames_full::Vector{String})

    @grammar grammar begin
        start = bin

        bin = always | eventually | implies 
        always = Expr(:call, :G, bin_vec) #global
        eventually = Expr(:call, :F, bin_vec) #future
        implies = Expr(:call, :Y, bin_vec, bin_vec)

        #produces a bin_vec
        bin_vec = and | or | not  | eq | lt | lte | gt | gte
        and = Expr(:call, :&, bin_vec, bin_vec)
        or = Expr(:call, :|, bin_vec, bin_vec)
        not = Expr(:call, :!, bin_vec)

        #comparisons
        eq = bin_eq | real_eq 
        bin_eq = Expr(:call, :eq, bin_vec, bin_vec)
        real_eq = Expr(:call, :eq, real_vec, real_vec) | Expr(:call, :feq, :D, real_feat_id, :V, real_val_id)
        lt = Expr(:call, :lt, real_vec, real_vec) | Expr(:call, :flt, :D, real_feat_id, :V, real_val_id)
        lte = Expr(:call, :lte, real_vec, real_vec) | Expr(:call, :flte, :D, real_feat_id, :V, real_val_id)
        gt = Expr(:call, :gt, real_vec, real_vec) | Expr(:call, :fgt, :D, real_feat_id, :V, real_val_id)
        gte = Expr(:call, :gte, real_vec, real_vec) | Expr(:call, :fgte, :D, real_feat_id, :V, real_val_id)

        #read features
        real_vec = Expr(:call, :g, :D, real_feat_id)
    end
    grammar.rules[:real_feat_id] = OrRule("real_feat_id", [Terminal("", real_feat_ids[i]) 
        for i=1:length(real_feat_ids)], nothing)
    grammar.rules[:real_val_id] = RangeRule("real_val_id", 1:num_real_vals, nothing)

    symtable = SymbolTable(
    :g => get_ref,
    :F => eventually,
    :G => globally,
    :Y => implies,
    :eq => eq,
    :lt => lt,
    :lte => lte,
    :gt => gt,
    :gte => gte,
    :feq => feq,
    :flt => flt,
    :flte => flte,
    :fgt => fgt,
    :fgte => fgte,
    :| => or,
    :& => and,
    :! => not 
    ) 

    function get_format_pretty{T<:AbstractString}(real_vals::Vector, colnames::Vector{T})
        fmt = Format()

        fmt["always"] = (cmd, args) -> "G($(args[1]))"
        fmt["eventually"] = (cmd, args) -> "F($(args[1]))"

        bin_infix(cmd, args, insym) = "($(args[1]) $insym $(args[2]))"
        fmt["and"] = (cmd, args) -> bin_infix(cmd, args, "&")
        fmt["or"] = (cmd, args) -> bin_infix(cmd, args, "|")
        fmt["not"] = (cmd, args) -> "!($(args[1]))"
        fmt["implies"] = (cmd, args) -> "$(args[1]) => $(args[2])"

        bin_infix_eq(cmd, args) = bin_infix(cmd, args, ".==")
        bin_infix_lt(cmd, args) = bin_infix(cmd, args, ".<")
        bin_infix_lte(cmd, args) = bin_infix(cmd, args, ".<=")
        bin_infix_gt(cmd, args) = bin_infix(cmd, args, ".>")
        bin_infix_gte(cmd, args) = bin_infix(cmd, args, ".>=")
        fmt["bin_eq"] = bin_infix_eq
        fmt["real_eq.1"] = bin_infix_eq
        fmt["lt.1"] = bin_infix_lt
        fmt["lte.1"] = bin_infix_lte
        fmt["gt.1"] = bin_infix_gt
        fmt["gte.1"] = bin_infix_gte

        function bin_infix_f(cmd, args, insym) 
            "($(colnames[parse(Int,args[1])]) $insym $(round(get_val(real_vals,parse(Int,args[1]),parse(Int,args[2])),4))"
        end
        bin_infix_feq(cmd, args) = bin_infix_f(cmd, args, ".==")
        bin_infix_flt(cmd, args) = bin_infix_f(cmd, args, ".<")
        bin_infix_flte(cmd, args) = bin_infix_f(cmd, args, ".<=")
        bin_infix_fgt(cmd, args) = bin_infix_f(cmd, args, ".>")
        bin_infix_fgte(cmd, args) = bin_infix_f(cmd, args, ".>=")
        fmt["real_eq.2"] = bin_infix_feq
        fmt["lt.2"] = bin_infix_flt
        fmt["lte.2"] = bin_infix_flte
        fmt["gt.2"] = bin_infix_fgt
        fmt["gte.2"] = bin_infix_fgte

        feat(cmd, args) = "$(colnames[parse(Int, args[1])])"
        fmt["real_vec"] = feat

        fmt
    end

    function get_format_natural{T<:AbstractString}(real_vals::Vector, colnames::Vector{T})
        fmt = Format()

        fmt["always"] = (cmd, args) -> "for all time, $(args[1])"
        fmt["eventually"] = (cmd, args) -> "at some point, $(args[1])"

        bin_infix(cmd, args, insym) = "[$(args[1]) $insym $(args[2])]"
        fmt["and"] = (cmd, args) -> bin_infix(cmd, args, "and")
        fmt["or"] = (cmd, args) -> bin_infix(cmd, args, "or")
        fmt["not"] = (cmd, args) -> "[it is not true that $(args[1])]"
        fmt["implies"] = (cmd, args) -> "whenever $(args[1]), it is also true that $(args[2])"

        bin_infix_eq(cmd, args) = bin_infix(cmd, args, "is equal to")
        bin_infix_lt(cmd, args) = bin_infix(cmd, args, "is less than")
        bin_infix_lte(cmd, args) = bin_infix(cmd, args, "is less than or equal to")
        bin_infix_gt(cmd, args) = bin_infix(cmd, args, "is greater than")
        bin_infix_gte(cmd, args) = bin_infix(cmd, args, "is greater than or equal to")
        fmt["bin_eq"] = bin_infix_eq
        fmt["real_eq.1"] = bin_infix_eq
        fmt["lt.1"] = bin_infix_lt
        fmt["lte.1"] = bin_infix_lte
        fmt["gt.1"] = bin_infix_gt
        fmt["gte.1"] = bin_infix_gte

        bin_infix_f(cmd, args, insym) = "[$(colnames[parse(Int,args[1])]) $insym $(round(get_val(real_vals,parse(Int,args[1]),parse(Int,args[2])),4))]"
        bin_infix_feq(cmd, args) = bin_infix_f(cmd, args, "is equal to")
        bin_infix_flt(cmd, args) = bin_infix_f(cmd, args, "is less than")
        bin_infix_flte(cmd, args) = bin_infix_f(cmd, args, "is less than or equal to")
        bin_infix_fgt(cmd, args) = bin_infix_f(cmd, args, "is greater than")
        bin_infix_fgte(cmd, args) = bin_infix_f(cmd, args, "is greater than or equal to")
        fmt["real_eq.2"] = bin_infix_feq
        fmt["lt.2"] = bin_infix_flt
        fmt["lte.2"] = bin_infix_flte
        fmt["gt.2"] = bin_infix_fgt
        fmt["gte.2"] = bin_infix_fgte

        feat(cmd, args) = "[$(colnames[parse(Int, args[1])])]"
        fmt["real_vec"] = feat

        fmt
    end

    fmt_pretty = get_format_pretty(real_vals, colnames)
    fmt_natural = get_format_natural(real_vals, colnames_full)

    (grammar, symtable, fmt_pretty, fmt_natural)
end

end #module
