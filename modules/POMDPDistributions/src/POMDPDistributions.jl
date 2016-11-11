module POMDPDistributions

using Compat

import Base: size, eltype, length, full, convert, show, getindex, scale, rand, rand!
import Base: +, -, .+, .-

export
     # generic types
    VariateForm,
    ValueSupport,
    Univariate,
    Multivariate,
    Matrixvariate,
    Discrete,
    Continuous,
    Sampleable,
    Distribution,
    UnivariateDistribution,
    MultivariateDistribution,
    MatrixDistribution,
    NoncentralHypergeometric,
    NonMatrixDistribution,
    DiscreteDistribution,
    ContinuousDistribution,
    DiscreteUnivariateDistribution,
    DiscreteMultivariateDistribution,
    DiscreteMatrixDistribution,
    ContinuousUnivariateDistribution,
    ContinuousMultivariateDistribution,
    ContinuousMatrixDistribution,
    SufficientStats,
    AbstractMvNormal,
    AbstractMixtureModel,
    UnivariateMixture,
    MultivariateMixture,
 
    # distribution types
    Categorical,

    # methods 
    rand,
    set_prob!


### source files

# type system
include("common.jl")

# implementation helpers
include("utils.jl")

# specific samplers and distributions
include("univariates.jl")

end # module
