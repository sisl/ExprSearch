"""
Grammar-based Monte Carlo Tree Search algorithm
Modifications: max instead of mean. Number of actions varying with state. Deterministic assumed.
Subtree explorations are tracked.
"""
module GBMCTS

using POMDPs

export
    MCTSSolver,
    MCTSPolicy,
    action,
    simulate,
    rollout,
    # SPW
    StateNode


typealias Reward Float64

include("policies.jl")
include("vanilla.jl")

end # module
