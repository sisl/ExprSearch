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

module SA  #ExprSearch.SA simulated annealing

export SAESParams, SAESResult, sa_search, exprsearch, SearchParams, SearchResult
export accept_prob, estimate_temp_params

using Reexport
@reexport using DerivationTrees
@reexport using GrammaticalEvolution
@reexport using RLESUtils.Observers
using RLESUtils.GitUtils
using CPUTime
using Iterators

import DerivationTrees.initialize!
import ..ExprSearch: SearchParams, SearchResult, exprsearch, ExprProblem, create_grammar, get_fitness

type SAESParams <: SearchParams
  #tree params
  maxsteps::Int64

  #SA
  T1::Float64 #initial temperature
  alpha::Float64 #decay rate
  n_epochs::Int64

  observer::Observer
end

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
end

SAESResult(tree::DerivationTree) = SAESResult(tree, realmax(Float64), 0, 0, 0)
SAESResult(s::SAState) = SAESResult(s.tree, s.fitness, s.expr, 1, 1)

exprsearch(p::SAESParams, problem::ExprProblem, userargs...) = sa_search(p, problem::ExprProblem, userargs...)

function sa_search(p::SAESParams, problem::ExprProblem, userargs...)
  @notify_observer(p.observer, "verbose1", ["Starting SA search"])
  @notify_observer(p.observer, "computeinfo", ["starttime", string(now())])

  grammar = create_grammar(problem)
  tree_params = DerivTreeParams(grammar, p.maxsteps)
  tree = DerivationTree(tree_params)

  s = SAState(tree)
  initialize!(s, problem)
  result = SAESResult(s)

  T = p.T1
  for i = 1:p.n_epochs
    println("i=", i)
    @notify_observer(p.observer, "iteration", [i])
    @notify_observer(p.observer, "temperature", [i, T])

    CPUtic()

    ###########################################
    # SA algorithm
    sp = neighbor(s, problem, result)

    #only accept complete trees
    if iscomplete(sp.tree) && accept(s, sp, T)
      s = sp
    end

    #update temperature
    T *= p.alpha
    ###########################################

    @notify_observer(p.observer, "cputime", [i, CPUtoq()])
    @notify_observer(p.observer, "current_best", [i, result.fitness, result.expr])
  end

  @notify_observer(p.observer, "result", [result.fitness, string(result.expr), result.best_at_eval, result.totalevals])

  #meta info
  @notify_observer(p.observer, "computeinfo", ["endtime",  string(now())])
  @notify_observer(p.observer, "computeinfo", ["hostname", gethostname()])
  @notify_observer(p.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
  @notify_observer(p.observer, "parameters", ["maxsteps", p.maxsteps])
  @notify_observer(p.observer, "parameters", ["temp_schedule", p.temp_schedule])

  return result
end

function initialize!(s::SAState, problem::ExprProblem, retries::Int64=typemax(Int64))
  rand!(s.tree, retries) #random tree
  s.expr = get_expr(s.tree)
  s.fitness = get_fitness(problem, s.expr)
end

#get a new neighbor and update globals
function neighbor(s::SAState, problem::ExprProblem)
  sp = deepcopy(s)

  #perturb and evaluate
  if perturb!(sp.tree) #true if sucessful
    sp.expr = get_expr(sp.tree)
    sp.fitness = get_fitness(problem, sp.expr)
  else
    sp = s
  end
  return sp
end

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

function neighbor(s::SAState, problem::ExprProblem, result::SAESResult)
  sp = neighbor(s, problem)
  update!(result, sp)
  return sp
end

#randomly perturb the tree
function perturb!(tree::DerivationTree, retries::Int64=typemax(Int64))
  node = rand(tree) #uniformly select a node
  b_success = rand!(node, tree, retries) #randomly change the subtree
  return b_success
end

#accept rule for simulated annealing, always accept a downhill step
function accept(s::SAState, sp::SAState, T::Float64)
  return sp.fitness < s.fitness || exp((s.fitness - sp.fitness) / T) > rand()
end

function accept_prob(problem::ExprProblem, maxsteps::Int64, P1::Float64, N::Int64)
  sum_uphill = 0.0
  n_uphill = 0

  grammar = create_grammar(problem)
  tree_params = DerivTreeParams(grammar, maxsteps)
  tree = DerivationTree(tree_params)

  s = SAState(tree)
  initialize!(s, problem)

  for i = 1:N
    #println("i=", i)
    sp = neighbor(s, problem)

    #println("s=", s.expr, ", fitness=", s.fitness)
    #println("sp=", sp.expr, ", fitness=", sp.fitness)

    if iscomplete(sp.tree) && rand() < P1
      #println("accepted")
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
  P1 = mean(Ps)
  T1 = -deltabar / log(P1)
  println("deltabar=", deltabar, ", P1=", P1, ", T1=", T1)

  alpha = (Tfinal / T1)^(1 / n_epochs)
  println("T1=", T1, ", alpha=", alpha, ", epochs=", n_epochs)

  return T1, alpha, n_epochs
end

end #module
