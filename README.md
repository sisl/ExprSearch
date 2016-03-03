# ExprSearch

Author: Ritchie Lee, Carnegie Mellon University Silicon Valley, ritchie.lee@sv.cmu.edu

In the face of big data, gaining insights by manually sifting through data is no longer practical.  Machine learning methods typically rely on statistical models.  Although these may provide good input/output behavior, the results are not conducive to human understanding.  We explore machine learning tasks guided by problem-specific grammar that a user provides.  We learn expressions derived from the provided grammar, making the results intuitive and interpretable to a human.

## Overview

ExprSearch is a collection of algorithms for solving grammar-guided expression discovery problems.  These problems are traditionally tackled using Genetic Programming.  Other methods also exist.  The following algorithms are currently available:

* GE - Grammatical Evolution (using the GrammaticalEvolution package)
* SA - Simulated Annealing
* MC - Monte Carlo
* MCTS/MCTS2 - Monte Carlo Tree Search and variants

## Problems

A problem extends ``ExprProblem`` and defines various specifics for a given grammar optimization problem, including:

* Grammar - From which expressions should be derived. Defines domain of search space.  The semantics of the grammar can be defined arbitrarily by the user.  For example, subsets of temporal logic can be used in time-series analysis.
* Fitness function - A function that maps an expression to a real number indicating the quality of the expression.  Lower is better.  For example, for a classification task, this may be misclassification rate.

The following problems are currently available, see GrammarExpts.jl package:

* ACAS X ("ACASXProblem") - Time-series classification task for encounter data from RLESCAS. Learn an expression for the decision boundary to separate NMACs vs. non-NMACs or discover rules to explain clusterings.
* Symbolic regression ("SymbolicProblem") - Reconstruct/rediscover the symbolic form of a mathematical expression from evaluation data only.

## Installation

Julia 0.4 is required.

* Pkg.clone("https://github.com/sisl/ExprSearch.jl.git", "ExprSearch")
* Pkg.build("ExprSearch") to automatically install dependencies
* Recommended, Pkg.test("ExprSearch")

Note: There is no need to call these if installing with GrammarExpts.jl.  Building GrammarExpts.jl automatically builds ExprSearch.jl.

### Main Dependencies

These packages are automatically fetched by the build script:

* RLESUtils.jl - Misc tools and utils

### Useful Locations to Know

* PKGDIR/ExprSearch/modules - Contains (mostly support) submodules
* PKGDIR/ExprSearch/src - Contains the source files for the main algorithms

## General Usage

The algorithms all follow the same form.  We will use Monte Carlo as an example.

```julia
using ExprSearch.MC #make MC algorithm available
problem = Symbolic(...) #problem is defined by a type that extends ExprProblem
p = MCESParams(...) #choose the algorithm by populating its input params object that extends SearchParams
result = exprsearch(p::SearchParams, problem::ExprProblem) #algorithm dispatched on p::SearchParams
#result is of type MCSearchResult that extends SearchResult
```

