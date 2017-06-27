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
Parallel Monte Carlo search by drawing uniform samples from the root of the grammar.
Returns the sample with the best result.  
"""
module PMC

warn("PMC is deprecated")
export PMCESParams, MCESParams, get_derivtree

using ExprSearch
using ExprSearch.MC
using RLESUtils, GitUtils, CPUTimeUtils, Observers, LogSystems

import ExprSearch: SearchParams, exprsearch, ExprProblem

include("logdefs.jl")

type PMCESParams <: SearchParams
    n_threads::Int64
    mc_params::MCESParams
    logsys::LogSystem
end
PMCESParams(n_threads::Int64, mc_params::MCESParams) = PMCESParams(n_threads, mc_params, logsystem())

exprsearch(p::PMCESParams, problem::ExprProblem, userargs...) = pmc_search(p, problem, userargs...)

function pmc_search(p::PMCESParams, problem::ExprProblem, userargs...)
    @notify_observer(p.logsys.observer, "verbose1", ["Starting PMC Search..."])
    @notify_observer(p.logsys.observer, "computeinfo", ["starttime", string(now())])

    tic()
    results = pmap(1:p.n_threads) do tid
        mc_search(p.mc_params, problem, userargs...)
    end
    result = minimum(results) #best fitness
    totalevals = sum(map(r -> r.totalevals, results))
    
    @notify_observer(p.logsys.observer, "result", [result.fitness, string(result.expr),
        0, totalevals])

    #meta info
    computetime_s = toq()
    @notify_observer(p.logsys.observer, "computeinfo", ["computetime_s",  computetime_s])
    @notify_observer(p.logsys.observer, "computeinfo", ["endtime",  string(now())])
    @notify_observer(p.logsys.observer, "computeinfo", ["hostname", gethostname()])
    @notify_observer(p.logsys.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
    @notify_observer(p.logsys.observer, "parameters", ["maxsteps", p.mc_params.maxsteps])
    @notify_observer(p.logsys.observer, "parameters", ["n_samples", p.mc_params.n_samples])
    @notify_observer(p.logsys.observer, "parameters", ["n_threads", p.n_threads])

  return result
end

end #module
