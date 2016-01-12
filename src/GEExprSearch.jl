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

module GE

export GEESParams, GEESResult, ge_search, exprsearch, SearchParams, SearchResult
export stop, get_fitness

using Reexport
@reexport using GrammaticalEvolution
@reexport using RLESUtils.Observers
using RLESUtils.GitUtils
using CPUTime

import ..ExprSearch: SearchParams, SearchResult, exprsearch

type GEESParams <: SearchParams
  #GrammaticalEvolution params
  grammar::Grammar
  genome_size::Int64
  pop_size::Int64
  maxwraps::Int64
  top_percent::Float64
  prob_mutation::Float64
  mutation_rate::Float64
  default_code::Expr
  max_iters::Int64

  ge_observer::Observer

  observer::Observer
end

type GEESResult <: SearchResult
  genome::Vector{Int64}
  fitness::Float64
  expr::Union{Symbol,Expr}
end

exprsearch(p::GEESParams) = ge_search(p)

stop() = false #dispatch doesn't work? otherwise would use stop(xs...) as a default
get_fitness() = error("fitness function not defined") #user should override this,
#dispatch doesn't work otherwise would use get_fitness(xs...) as a default

function GrammaticalEvolution.evaluate!(grammar::Grammar, ind::ExampleIndividual, p::GEESParams)
  try
    ind.code = transform(grammar, ind, maxwraps=p.maxwraps)
    ind.fitness = get_fitness(ind.code)
  catch e
    if !isa(e, MaxWrapException)
      s = take(string(e), 50) |> join
      println("exception = $s")
      s = take(string(ind.code), 50) |> join
      len = length(string(ind.code))
      println("length=$len, code: $(s)")
      f = open("errorlog.txt", "a") #log to file
      println(f, typeof(e))
      println(f, string(ind.code))
      close(f)
    end
    ind.code = p.default_code
    ind.fitness = realmax(Float64)
  end
end

function ge_search(p::GEESParams)
  @notify_observer(p.observer, "verbose1", ["Starting GE search"])
  @notify_observer(p.observer, "computeinfo", ["starttime", string(now())])

  pop = ExamplePopulation(p.pop_size, p.genome_size)
  fitness = realmax(Float64)
  iter = 1
  while !stop(iter, fitness) && iter <= p.max_iters
    CPUtic()
    pop = generate(p.grammar, pop, p.top_percent, p.prob_mutation, p.mutation_rate, p)
    fitness = pop[1].fitness #population is sorted, so first entry is the best
    code = pop[1].code
    @notify_observer(p.observer, "iteration_time", Any[iter, CPUtoq()])
    @notify_observer(p.observer, "fitness", Any[iter, fitness])
    @notify_observer(p.observer, "fitness5", Any[iter, [pop[i].fitness for i = 1:5]...])
    @notify_observer(p.observer, "code", Any[iter, string(code)])
    @notify_observer(p.observer, "population", Any[iter, pop])
    @notify_observer(p.observer, "best_individual", [iter, fitness, code])
    iter += 1
  end
  ind = pop[1]
  genome = ind.genome
  fitness = ind.fitness
  expr = ind.code
  @notify_observer(p.observer, "result", [fitness, string(expr)])

  #meta info
  @notify_observer(p.observer, "computeinfo", ["endtime",  string(now())])
  @notify_observer(p.observer, "computeinfo", ["hostname", gethostname()])
  @notify_observer(p.observer, "computeinfo", ["gitSHA",  get_SHA(dirname(@__FILE__))])
  @notify_observer(p.observer, "parameters", ["genome_size", p.genome_size])
  @notify_observer(p.observer, "parameters", ["pop_size", p.pop_size])
  @notify_observer(p.observer, "parameters", ["maxwraps", p.maxwraps])
  @notify_observer(p.observer, "parameters", ["top_percent", p.top_percent])
  @notify_observer(p.observer, "parameters", ["prob_mutation", p.prob_mutation])
  @notify_observer(p.observer, "parameters", ["mutation_rate", p.mutation_rate])
  @notify_observer(p.observer, "parameters", ["default_code", string(p.default_code)])
  @notify_observer(p.observer, "parameters", ["max_iters", p.max_iters])

  return GEESResult(genome, fitness, expr)
end

end #module
