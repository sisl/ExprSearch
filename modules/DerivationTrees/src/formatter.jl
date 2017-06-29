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

export Format, pretty_string, f_args, get_cmd

using RLESUtils, StringUtils

typealias Format Dict{String,Function} #usage: D[cmd] = f(cmd, args)

f_args(cmd, args) = "$cmd(" * join(args,", ") * ")"
get_cmd(cmd, args) = "$cmd"

function pretty_string(tree::DerivationTree, fmt::Format, capitalize::Bool=false)
  s = pretty_string(tree.root, tree.root.rule, fmt)
  return capitalize ? capitalize_first(s) : s
end

function pretty_base(node::DerivTreeNode, fmt::Format)
  cmd = node.cmd
  args = map(x -> pretty_string(x, x.rule, fmt), node.children)
  if haskey(fmt, cmd)
    out = fmt[cmd](cmd, args) #user callback
  elseif isempty(args)
    out = get_cmd(cmd, args) #return cmd
  elseif length(args) == 1
    out = args[1] #passthrough
  else
    out = f_args(cmd, args) #f(arg1, arg2, ...)
  end
  return out
end

function pretty_string(node::DerivTreeNode, rule::RangeRule, fmt::Format)
  node.cmd = string(get_expr(node, rule))
  return pretty_base(node, fmt)
end

function pretty_string(node::DerivTreeNode, rule::Terminal, fmt::Format)
  node.cmd = string(rule.value)
  return pretty_base(node, fmt)
end

pretty_string(node::DerivTreeNode, rule::Rule, fmt::Format) = pretty_base(node, fmt)
