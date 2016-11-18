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
Simulated annealing search with geometrically decreasing temperature schedule.
Propose by randomly choosing an existing node from the parse tree,
and replacing the subtree by a randomly generated one.  Use PSA for parallel version over n_threads.
"""
module SA

export SAESParams, SAESResult, sa_search, exprsearch, SearchParams, SearchResult, get_derivtree
export accept_prob, estimate_temp_params
export psa_search, PSAESParams

using Reexport
using ExprSearch
using RLESUtils, GitUtils, SwapBuffers, CPUTimeUtils, Observers
using DerivationTrees
using GrammaticalEvolution
using Iterators

import DerivationTrees.initialize!
import ..ExprSearch: SearchParams, SearchResult, exprsearch, ExprProblem, get_grammar, get_fitness
import Base: isless, copy!

type SAESParams <: SearchParams
  #tree params
  maxsteps::Int64

  #SA
  T1::Float64 #initial temperature
  alpha::Float64 #decay rate
  n_epochs::Int64 #iterations
  n_starts::Int64 #number of starts

  observer::Observer
end

type PSAESParams <: SearchParams
  n_threads::Int64
  sa_params::SAESParams
  observer::Observer
end

PSAESParams(n_threads::Int64, sa_params::SAESParams) = PSAESParams(n_threads, sa_params, Observer())

type SAState
  tree::DerivationTree
  fitness::Float64
  expr
end

SAState(tree::DerivationTree) = SAState(tree, realmax(Float64), 0)

type SAESResult <: SearchResult
  tree::DerivationTree
  fitness::Float64
  expr
  best_at_eval::Int64
  totalevals::Int64

  function SAESResult()
    result = new()
    result.fitness = realmax(Float64)
    result.expr = 0
    result.best_at_eval = 0
    result.totalevals = 0
    return result
  end

  function SAESResult(tree::DerivationTree)
    result = SAESResult()
    result.tree = tree
    return result
  end
end

exprsearch(p::SAESParams, problem::ExprProblem, userargs...) = sa_search(p, problem, userargs...)
exprsearch(p::PSAESParams, problem::ExprProblem, userargs...) = psa_search(p, problem, userargs...)

function psa_search(p::PSAESParams, problem::ExprProblem, userargs...)
  @notify_observer(p.observer, "computeinfo", ["starttime", string(now())])
  tic()

  results = pmap(1:p.n_threads) do tid
    sa_search(p.sa_params, problem, userargs...)
  end

  result = minimum(results) #best fitness
  totalevals = sum(map(r -> r.totalevals, results))

  @notify_observer(p.observer, "result", [result.fitness, string(result.expr), totalevals])

  #meta info
  computetime_s = toq()
  @notify_observer(p.observer, "computeinfo", ["computetime_s",  computetime_s])
  @notify_observer(p.observer, "computeinfo", ["endtime",  string(now())])
  @notify_observer(p.observer, "computeinfo", ["hostname", gethostname()])
  @notify_observer(p.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
  @notify_observer(p.observer, "parameters", ["maxsteps", p.sa_params.maxsteps])
  @notify_observer(p.observer, "parameters", ["T1", p.sa_params.T1])
  @notify_observer(p.observer, "parameters", ["alpha", p.sa_params.alpha])
  @notify_observer(p.observer, "parameters", ["n_epochs", p.sa_params.n_epochs])
  @notify_observer(p.observer, "parameters", ["n_starts", p.sa_params.n_starts])
  @notify_observer(p.observer, "parameters", ["n_threads", p.n_threads])

  return result
end

function sa_search(p::SAESParams, problem::ExprProblem, userargs...)
  @notify_observer(p.observer, "verbose1", ["Starting SA search"])
  @notify_observer(p.observer, "computeinfo", ["starttime", string(now())])

  grammar = get_grammar(problem)
  tree_params = DerivTreeParams(grammar, p.maxsteps)
  s_buffer = SwapBuffer(SAState(DerivationTree(tree_params)),
                        SAState(DerivationTree(tree_params)))

  result = SAESResult()

  tstart = CPUtime_start()
  for j = 1:p.n_starts
    s = active(s_buffer)
    initialize!(s, problem, result) #initialize randomly

    T = p.T1
    for i = 1:p.n_epochs
      @notify_observer(p.observer, "iteration", [j, i])
      @notify_observer(p.observer, "temperature", [j, i, T])

      ###########################################
      # SA algorithm
      s = active(s_buffer)
      sp = inactive(s_buffer)
      neighbor!(sp, s, problem, result)

      #only accept complete trees
      if iscomplete(sp.tree) && accept(s, sp, T)
        swap!(s_buffer)
      end

      #update temperature
      T *= p.alpha
      ###########################################

      @notify_observer(p.observer, "elapsed_cpu_s", [j, i, CPUtime_elapsed_s(tstart)])
      @notify_observer(p.observer, "current_best", [j, i, result.fitness, string(result.expr)])
    end
    gc()
  end

  @notify_observer(p.observer, "result", [result.fitness, string(result.expr), result.best_at_eval, result.totalevals])

  #meta info
  @notify_observer(p.observer, "computeinfo", ["endtime",  string(now())])
  @notify_observer(p.observer, "computeinfo", ["hostname", gethostname()])
  @notify_observer(p.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
  @notify_observer(p.observer, "parameters", ["maxsteps", p.maxsteps])
  @notify_observer(p.observer, "parameters", ["T1", p.T1])
  @notify_observer(p.observer, "parameters", ["alpha", p.alpha])
  @notify_observer(p.observer, "parameters", ["n_epochs", p.n_epochs])
  @notify_observer(p.observer, "parameters", ["n_starts", p.n_starts])

  return result
end

#initialize to random state
function initialize!(s::SAState, problem::ExprProblem, retries::Int64=typemax(Int64))
  rand!(s.tree; retries=retries) #random tree
  s.expr = get_expr(s.tree)
  s.fitness = get_fitness(problem, s.expr)
end

function initialize!(s::SAState, problem::ExprProblem, result::SAESResult,
                     retries::Int64=typemax(Int64))
  initialize!(s, problem, retries)
  update!(result, s)
end

#get a new neighbor and update global trackers
function neighbor!(sp::SAState, s::SAState, problem::ExprProblem, retries::Int64=20)
  while retries > 0
    copy!(sp, s)
    #perturb and evaluate
    if perturb!(sp.tree) #true if successful
      sp.expr = get_expr(sp.tree)
      sp.fitness = get_fitness(problem, sp.expr)
      return sp #keep looping until successful
    end
    retries -= 1
  end
  error("neighbor!(): Max retries exceeded")
end

#update the global best trackers with the current state
function update!(result::SAESResult, s::SAState)
  result.totalevals += 1 #assumes an eval was called prior

  #update globals
  if s.fitness < result.fitness
    result.tree = s.tree
    result.fitness = s.fitness
    result.expr = s.expr
    result.best_at_eval = result.totalevals
  end
end

#get neighbor and update
function neighbor!(sp::SAState, s::SAState, problem::ExprProblem, result::SAESResult)
  neighbor!(sp, s, problem)
  update!(result, sp)
end

#randomly perturb the tree
function perturb!(tree::DerivationTree, retries::Int64=5)
  node = rand(tree) #uniformly select a node
  b_success = rand!(tree, node, retries) #randomly change the subtree
  return b_success
end

#accept rule for simulated annealing, always accept a downhill step or sidestep
function accept(s::SAState, sp::SAState, T::Float64)
  return sp.fitness <= s.fitness || exp((s.fitness - sp.fitness) / T) > rand()
end

############
#for estimating parameters before actual run
function accept_prob(problem::ExprProblem, maxsteps::Int64, P1::Float64, N::Int64)
  sum_uphill = 0.0
  n_uphill = 0

  grammar = get_grammar(problem)
  tree_params = DerivTreeParams(grammar, maxsteps)
  tree = DerivationTree(tree_params)

  s = SAState(tree)
  initialize!(s, problem)

  for i = 1:N
    #println("i=", i)
    sp = neighbor(s, problem)

    #println("s=", s.expr, ", fitness=", s.fitness)
    #println("sp=", sp.expr, ", fitness=", sp.fitness)

    iscomplete(sp.tree) || continue

    if sp.fitness <= s.fitness || rand() < P1
      #track uphill moves
      if sp.fitness > s.fitness
        sum_uphill += sp.fitness - s.fitness
        n_uphill += 1
      end
      s = sp
    end
  end

  avg_uphill = n_uphill != 0.0 ? sum_uphill / n_uphill : 0.0
  P = n_uphill / N

  #P should be close to P1

  return (avg_uphill, P)
end

#Reference: Park and Kim, "A Systematic Procedure for Setting Parameters in Simulated Annealing"
function estimate_temp_params(problem::ExprProblem,
                           P1::Float64,
                           n_epochs::Int64,
                           Tfinal::Float64,
                           maxsteps::Int64,
                           N::Int64,
                           ntrials::Int64=1000)

  avg_uphills = Float64[]
  Ps = Float64[]
  for i = 1:ntrials
    avg_uphill, P = accept_prob(problem, maxsteps, P1, N)
    push!(avg_uphills, avg_uphill)
    push!(Ps, P)
    println("avg_uphill=", avg_uphill, ", P=", P)
  end

  deltabar = mean(avg_uphills)
  #P1 = mean(Ps)
  T1 = -deltabar / log(P1)
  println("deltabar=", deltabar, ", P1=", P1, ", T1=", T1)

  alpha = (Tfinal / T1)^(1 / n_epochs)
  println("T1=", T1, ", alpha=", alpha, ", epochs=", n_epochs)

  return T1, alpha, n_epochs
end

isless(r1::SAESResult, r2::SAESResult) = r1.fitness < r2.fitness

function copy!(dst::SAState, src::SAState)
  copy!(dst.tree, src.tree)
  dst.fitness = src.fitness
  dst.expr = src.expr
end

end #module
