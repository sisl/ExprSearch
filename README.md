# ExprSearch.jl

Author: Ritchie Lee, Carnegie Mellon University Silicon Valley, ritchie.lee@sv.cmu.edu

In the face of big data, gaining insights by manually sifting through data is no longer practical.  Machine learning methods typically rely on opaque statistical models.  Although these may provide good input/output behavior, the results are not conducive to human understanding.  We explore machine learning tasks guided by a grammar for interpretability.  We learn expressions derived from the grammar, optimizing both fit to data and interpretability. 

## Overview

ExprSearch is a collection of algorithms for solving grammar-based expression discovery problems.  These problems are traditionally tackled using Genetic Programming.  Other methods also exist.  The following algorithms are currently available:

* GP - Genetic Programming 
* GE - Grammatical Evolution (via the GrammaticalEvolution package)
* CE - Cross-Entropy method
* MC - Monte Carlo
* MCTS - Monte Carlo Tree Search

## Problems

A problem extends ``ExprProblem`` and defines various specifics for a given grammar optimization problem, including:

* Grammar - From which expressions should be derived. Defines domain of search space.  The semantics of the grammar can be defined arbitrarily by the user.  For example, subsets of temporal logic can be used in time-series analysis.
* Fitness function - A function that maps an expression to a real number indicating the quality of the expression.  Lower is better.  For example, for a classification task, this may be misclassification rate.

The following example problems are available. 

* Symbolic regression ("SymbolicRegression") - Reconstruct/rediscover the symbolic form of a mathematical expression from evaluation data only.

## Installation

Julia 0.5 is required.

* Pkg.clone("https://github.com/sisl/ExprSearch.jl.git", "ExprSearch")
* Pkg.build("ExprSearch") to automatically install dependencies
* Recommended, Pkg.test("ExprSearch")

### Main Dependencies

These packages are automatically fetched by the build script:

* RLESUtils.jl - Misc tools and utils

### Useful Locations to Know

* PKGDIR/ExprSearch/modules - Contains (support) submodules
* PKGDIR/ExprSearch/src - Contains the source files for the main algorithms

## Grammar-based Expression Search 

The algorithms all follow the same form.  We will use Monte Carlo as an example.

```julia
using ExprSearch.MC #make MC algorithm available
problem = Symbolic(...) #problem is defined by a type that extends ExprProblem
p = MCESParams(...) #choose the algorithm by populating its input params object that extends SearchParams
result = exprsearch(p::SearchParams, problem::ExprProblem) #algorithm dispatched on p::SearchParams
#result is of type MCSearchResult that extends SearchResult
```

## Grammar-based Decision Trees (GBDT)

To learn a decision tree using grammar-based expression search as a subroutine, first set the params for the expression search algorithm, then pass that as an argument into the gbdt params object.

```julia
data = dataset("MyDataset")
grammar, symtable, _, _ = Grammars.time_series_realonly1(...)
fitness_function = FitnessFunctions.Gini_NumNodes(w_metric, w_num_nodes)
problem = GBDMProblem(data, grammar, fitness_function, symtable)
gp_params = GPESParams(...)
gbdt_params = GBDTParams(problem, length(data), gp_params, max_gbdt_depth, ...) 
result = induce_tree(gbdt_params)
```
