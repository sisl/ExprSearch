rand(d::UnivariateDistribution) = quantile(d, rand())
rand(rng::AbstractRNG, d::UnivariateDistribution) = quantile(d, rand(rng))

##### specific distributions #####

const discrete_distributions = [
    "categorical"
]

for dname in discrete_distributions
    include(joinpath("univariate", "discrete", "$(dname).jl"))
end
