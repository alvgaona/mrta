module Kinematics

using ForwardDiff: jacobian, hessian

export forward_pose,
    inverse_pose,
    forward_kin_jacobian,
    inverse_kin_jacobian,
    dot_forward_jacobian_exact,
    dot_forward_jacobian

"""
    forward_pose(r::AbstractVector{<:Real})::Vector{<:Real}

Compute forward kinematics to transform from robots pose to cluster pose.

Arguments:
  r: robots position vector defined as (x₁, y₁, θ₁, x₂, y₂, θ₂) where:
     - x₁, y₁: position of robot 1 in global frame (meters)
     - θ₁: orientation of robot 1 in global frame (radians)
     - x₂, y₂: position of robot 2 in global frame (meters)
     - θ₂: orientation of robot 2 in global frame (radians)

Returns:
  cluster pose vector defined as (xᶜ, yᶜ, d, θᶜ, ϕ₁, ϕ₂) where:
     - xᶜ, yᶜ: center position of the cluster in global frame (meters)
     - d: half-distance between robots (meters)
     - θᶜ: orientation of the cluster in global frame (radians)
     - ϕ₁, ϕ₂: relative orientations of robots with respect to cluster (radians)
"""
function forward_pose(r::AbstractVector{<:Real})::Vector{<:Real}
    length(r) == 6 || throw(ArgumentError("Robot pose vector must have 6 elements"))

    x₁, y₁, θ₁, x₂, y₂, θ₂ = r

    a = x₁ - x₂
    b = y₁ - y₂

    xᶜ = (x₁ + x₂) / 2
    yᶜ = (y₁ + y₂) / 2
    d = (1 / 2) * sqrt(a^2 + b^2)

    θᶜ = atan(a, b)

    ϕ₁ = θ₁ - θᶜ
    ϕ₂ = θ₂ - θᶜ

    return [xᶜ, yᶜ, d, θᶜ, ϕ₁, ϕ₂]
end

"""
    inverse_pose(c::AbstractVector{<:Real})::Vector{<:Real}

Compute inverse kinematics to transform from cluster pose to robots pose.

Arguments:
  c: cluster position vector defined as (xᶜ, yᶜ, d, θᶜ, ϕ₁, ϕ₂) where:
     - xᶜ, yᶜ: center position of the cluster in global frame (meters)
     - d: half-distance between robots (meters)
     - θᶜ: orientation of the cluster in global frame (radians)
     - ϕ₁, ϕ₂: relative orientations of robots with respect to cluster (radians)

Returns:
  robots pose vector defined as (x₁, y₁, θ₁, x₂, y₂, θ₂) where:
     - x₁, y₁: position of robot 1 in global frame (meters)
     - θ₁: orientation of robot 1 in global frame (radians)
     - x₂, y₂: position of robot 2 in global frame (meters)
     - θ₂: orientation of robot 2 in global frame (radians)
"""
function inverse_pose(c::AbstractVector{<:Real})::Vector{<:Real}
    length(c) == 6 || throw(ArgumentError("Cluster pose vector must have 6 elements"))

    xᶜ, yᶜ, d, θᶜ, ϕ₁, ϕ₂ = c

    d >= 0 || throw(ArgumentError("Distance between robots must be non-negative"))

    x₁ = xᶜ + d * sin(θᶜ)
    x₂ = xᶜ - d * sin(θᶜ)

    y₁ = yᶜ + d * cos(θᶜ)
    y₂ = yᶜ - d * cos(θᶜ)

    θ₁ = θᶜ + ϕ₁
    θ₂ = θᶜ + ϕ₂

    return [x₁, y₁, θ₁, x₂, y₂, θ₂]
end

"""
    forward_kin_jacobian(robots_pose::AbstractVector{<:Real})::Matrix{<:Real}

Compute jacobian matrix based on the forward kinematics relationship.
This is the derivative of the cluster pose with respect to the robot pose.

Arguments:
  robots_pose: robots position vector defined as (x₁, y₁, θ₁, x₂, y₂, θ₂)
Returns:
  J: jacobian matrix of size 6×6
"""
function forward_kin_jacobian(robots_pose::AbstractVector{<:Real})::Matrix{<:Real}
    length(robots_pose) == 6 || throw(ArgumentError("Robot pose vector must have 6 elements"))

    f(x) = [
        (x[1] + x[4]) / 2;
        (x[2] + x[5]) / 2;
        1 / 2 * sqrt((x[1] - x[4])^2 + (x[2] - x[5])^2);
        atan(x[2] - x[5], x[1] - x[4]);
        x[3] - atan(x[2] - x[5], x[1] - x[4]);
        x[6] - atan(x[2] - x[5], x[1] - x[4])
    ]

    return jacobian(f, robots_pose)
end

"""
    inverse_kin_jacobian(cluster_pose::AbstractVector{<:Real})::Matrix{<:Real}

Compute jacobian matrix based on the inverse kinematics relationship.
This is the derivative of the robot pose with respect to the cluster pose.

Arguments:
  cluster_pose: cluster position vector defined as (xᶜ, yᶜ, d, θᶜ, ϕ₁, ϕ₂)
Returns:
  J⁻¹: jacobian matrix of size 6×6
"""
function inverse_kin_jacobian(cluster_pose::AbstractVector{<:Real})::Matrix{<:Real}
    length(cluster_pose) == 6 || throw(ArgumentError("Cluster pose vector must have 6 elements"))

    f⁻¹(x) = inverse_pose(cluster_pose)

    return jacobian(f⁻¹, cluster_pose)
end

"""
    dot_forward_jacobian(r::AbstractVector{<:Real}, dr::AbstractVector{<:Real})::Matrix{<:Real}

Compute the time derivative of the Jacobian matrix for the forward kinematics relationship.

This function calculates how the Jacobian matrix changes with respect to time given the
current robot positions and their velocities.

Arguments:
  r: robots position vector defined as (x₁, y₁, θ₁, x₂, y₂, θ₂) where:
     - x₁, y₁: position of robot 1 in global frame (meters)
     - θ₁: orientation of robot 1 in global frame (radians)
     - x₂, y₂: position of robot 2 in global frame (meters)
     - θ₂: orientation of robot 2 in global frame (radians)
  dr: robots velocity vector defined as (dx₁, dy₁, dθ₁, dx₂, dy₂, dθ₂) where:
     - dx₁, dy₁: linear velocity of robot 1 (meters/second)
     - dθ₁: angular velocity of robot 1 (radians/second)
     - dx₂, dy₂: linear velocity of robot 2 (meters/second)
     - dθ₂: angular velocity of robot 2 (radians/second)

Returns:
  Jₜ: time derivative of the Jacobian matrix (6×6 matrix)
"""
function dot_forward_jacobian_exact(r::AbstractVector{<:Real}, dr::AbstractVector{<:Real})::Matrix{<:Real}
    length(r) == 6 || throw(ArgumentError("Robot pose vector must have 6 elements"))
    length(dr) == 6 || throw(ArgumentError("Robot velocity vector must have 6 elements"))

    x₁, y₁, _, x₂, y₂, _ = r
    dx₁, dy₁, _, dx₂, dy₂, _ = dr

    B = (x₁ - x₂)^2 + (y₁ - y₂)^2

    Jₜ = zeros(6, 6)

    Jₜ[3, :] = 1 / (2 * B^(3 / 2)) * [
        (y₁ - y₂) * ((y₁ - y₂) * (dx₁ - dx₂) - (x₁ - x₂) * (dy₁ - dy₂)),
        (x₁ - x₂) * (-(y₁ - y₂) * (dx₁ - dx₂) + (x₁ - x₂) * (dy₁ - dy₂)),
        0,
        (y₁ - y₂) * (-(y₁ - y₂) * (dx₁ - dx₂) + (x₁ - x₂) * (dy₁ - dy₂)),
        -(x₁ - x₂) * (-(y₁ - y₂) * (dx₁ - dx₂) + (x₁ - x₂) * (dy₁ - dy₂)),
        0
    ]

    Jₜ[4, :] = (1 / B^2) * [
        2 * (x₁ - x₂) * (y₁ - y₂) * dx₁ - 2 * (x₁ - x₂) * (y₁ - y₂) * dx₂ - (x₁ - x₂ + y₁ - y₂) * (x₁ - x₂ - y₁ + y₂) * (dy₁ - dy₂),
        B * (dx₁ - dx₂) - (x₁ - x₂) * (2 * (x₁ - x₂) * (dx₁ - dx₂) + 2 * (y₁ - y₂) * (dy₁ - dy₂)),
        0,
        -2 * (x₁ - x₂) * (y₁ - y₂) * dx₁ + 2 * (x₁ - x₂) * (y₁ - y₂) * dx₂ + (x₁ - x₂ + y₁ - y₂) * (x₁ - x₂ - y₁ + y₂) * (dy₁ - dy₂),
        B * (-dx₁ + dx₂) - (-x₁ + x₂) * (2 * (x₁ - x₂) * (dx₁ - dx₂) + 2 * (y₁ - y₂) * (dy₁ - dy₂)),
        0
    ]

    Jₜ[5, :] = (1 / B^2) * [
        -2 * (x₁ - x₂) * (y₁ - y₂) * dx₁ + 2 * (x₁ - x₂) * (y₁ - y₂) * dx₂ + (x₁ - x₂ + y₁ - y₂) * (x₁ - x₂ - y₁ + y₂) * (dy₁ - dy₂),
        B * (-dx₁ + dx₂) - (-x₁ + x₂) * (2 * (x₁ - x₂) * (dx₁ - dx₂) + 2 * (y₁ - y₂) * (dy₁ - dy₂)),
        0,
        2 * (x₁ - x₂) * (y₁ - y₂) * dx₁ - 2 * (x₁ - x₂) * (y₁ - y₂) * dx₂ - (x₁ - x₂ + y₁ - y₂) * (x₁ - x₂ - y₁ + y₂) * (dy₁ - dy₂),
        B * (dx₁ - dx₂) - (x₁ - x₂) * (2 * (x₁ - x₂) * (dx₁ - dx₂) + 2 * (y₁ - y₂) * (dy₁ - dy₂)),
        0
    ]

    Jₜ[6, :] = Jₜ[5, :]

    return Jₜ
end

function dot_forward_jacobian(r::AbstractVector{<:Real}, dr::AbstractVector{<:Real})::Matrix{<:Real}
    length(r) == 6 || throw(ArgumentError("Robot pose vector must have 6 elements"))
    length(dr) == 6 || throw(ArgumentError("Robot velocity vector must have 6 elements"))

    f(x) = forward_pose(x)

    # Calculate each Hessian in a loop and stack them vertically
    H = zeros(0, length(r))  # Initialize an empty matrix to store Hessians
    for i in 1:length(f(r))
        Hi = hessian(x -> f(x)[i], r)
        H = vcat(H, Hi)
    end

    # Calculate dJ by applying both Hessians to dx and reshaping
    dJ_vec = H * dr

    # Reshape into a square matrix with dimensions matching the Jacobian
    dJ = Matrix(reshape(dJ_vec, 6, 6)')

    return dJ
end

end # module Kinematics
