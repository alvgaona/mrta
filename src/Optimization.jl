module Optimization

using Convex, SCS
using LinearAlgebra

export run_optimization_problem

"""
    run_optimization_problem(
        x::AbstractVector{T},
        p::AbstractMatrix{T};
        control_dim::Int=3,
        num_tasks::Int=3,
        num_robots::Int=2
    ) -> Tuple{Vector{T}, Vector{T}, Vector{T}} where T <: Real

Solves an optimization problem for multi robot task allocation.

# Arguments
- `x::AbstractVector{T}`: Current position vector of the robot
- `p::AbstractMatrix{T}`: Task positions matrix where each row is a task position
- `control_dim::Int`: Dimension of the control input (default: 3)
- `num_tasks::Int`: Number of tasks to be allocated (default: 3)
- `num_robots::Int`: Number of robots in the system (default: 2)

# Returns
- `u::Vector{T}`: Optimal control input
- `α::Vector{T}`: Task allocation vector
- `δ::Vector{T}`: Task deviation vector

# Description
This function formulates and solves a convex optimization problem for multi-robot
task allocation and control generation with specialization matrices.
"""
function run_optimization_problem(
    x::AbstractVector{T},
    p::AbstractMatrix{T};
    control_dim::Int=3,
    num_tasks::Int=3,
    num_robots::Int=2
)::Tuple{Vector{T},Vector{T},Vector{T}} where {T<:Real}

    S = diagm([1, 1, 1]) # specialization matrices
    P = S[1] * pinv(S[1])

    global_task_specification::Vector{T} = [0.5; 0.5; 0.5]

    # Parameters
    C::T = 1.0
    L::T = 1.0
    DELTA_MAX::T = 10.0
    KAPPA::T = 10.0

    u = Variable(control_dim)
    α = Variable(num_tasks)
    δ = Variable(num_tasks)

    objective = C * sumsquares(global_task_specification - 1 / num_robots * P * α) + sumsquares(u) + L * sumsquares(δ)

    constraints = [
        -2 * (x - p[1, :])' * u >= log(norm(x - p[1, :])^2) - δ[1],
        -2 * (x - p[2, :])' * u >= log(norm(x - p[2, :])^2) - δ[2],
        -2 * (x - p[3, :])' * u >= log(norm(x - p[3, :])^2) - δ[3],
        norm(δ, Inf) <= DELTA_MAX,
        ones(1, num_tasks) * α == 1.0,
        δ[1] <= KAPPA * (δ[1] - DELTA_MAX * (1 - α[1])),
        δ[1] <= KAPPA * (δ[3] - DELTA_MAX * (1 - α[3])),
        α <= 1,
        α >= 0,
        u <= 0.2,
        u >= -0.2
    ]

    problem = minimize(objective, constraints)
    solve!(problem, SCS.Optimizer; silent_solver=true)

    return evaluate(u), evaluate(α), evaluate(δ)
end

end # end module Optimization
