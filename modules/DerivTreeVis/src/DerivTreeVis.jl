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
Visualizer for DrivationTrees.jl
Produces a TikzQTree
"""
module DerivTreeVis
#TODO: update this to use AbstractTrees.jl interface

export derivtreevis

using ExprSearch, DerivationTrees
using RLESUtils, TreeToJSON, TikzQTrees
using Iterators

function derivtreevis(tree::DerivationTree, outfileroot::AbstractString;
    format::Symbol=:TEXPDF)

    get_name(tree::DerivationTree) = get_name(tree.root)
    function get_name(node::DerivTreeNode)
        cmd_text = node.cmd
        rule_text = split(string(typeof(node.rule)), ".")[end]
        action_text = string(node.action)
        expr_text = string(get_expr(node))
        text = join([cmd_text, rule_text, action_text, expr_text], "\\\\")
        return text
    end

    get_children(tree::DerivationTree) = get_children(tree.root)
    get_children(node::DerivTreeNode) = imap(x -> ("", x), node.children)
    get_depth(tree::DerivationTree) = get_depth(tree.root)
    get_depth(node::DerivTreeNode) = node.depth
    
    viscalls = VisCalls(get_name, get_children, get_depth)
    write_json(tree, viscalls, "$(outfileroot).json")
    plottree("$(outfileroot).json", outfileroot="$(outfileroot)";
        format=format)
end

end #module
