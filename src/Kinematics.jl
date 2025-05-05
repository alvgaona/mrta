module Kinematics

using ForwardDiff: jacobian

# Export the functions
export forward_pose, inverse_pose, forward_kin_jacobian, inverse_kin_jacobian

"""
    forward_pose(r::AbstractVector{<:Real})::Vector{<:Real}

Compute forward kinematics to transform from robots pose to cluster pose.

Arguments:
  r: robots position vector defined as (x₁, y₁, θ₁, x₂, y₂, θ₂)
Returns:
  cluster pose vector defined as (xᶜ, yᶜ, d, θᶜ, ϕ₁, ϕ₂)
"""
function forward_pose(r::AbstractVector{<:Real})::Vector{<:Real}
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
  c: cluster position vector defined as (xᶜ, yᶜ, d, θᶜ, ϕ₁, ϕ₂)
Returns:
  robots pose vector defined as (x₁, y₁, θ₁, x₂, y₂, θ₂)
"""
function inverse_pose(c::AbstractVector{<:Real})::Vector{<:Real}
  xᶜ, yᶜ, d, θᶜ, ϕ₁, ϕ₂ = c

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

Compute jacobian matrix based on the forward kinematics relationship

Arguments:
  robots_pose: robots position vector defined as (x₁, y₁, θ₁, x₂, y₂, θ₂)
Returns:
  J: jacobian matrix
"""
function forward_kin_jacobian(robots_pose::AbstractVector{<:Real})::Matrix{<:Real}
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

Compute jacobian matrix based on the inverse kinematics relationship

Arguments:
  cluster_pose: cluster position vector defined as (x_c, y_c, d, θ, φ₁, φ₂)
Returns:
  J⁻¹: jacobian matrix
"""
function inverse_kin_jacobian(cluster_pose::AbstractVector{<:Real})::Matrix{<:Real}
  f⁻¹(x) = [
    x[1] + x[3] * sin(x[4]);
    x[2] + x[3] * cos(x[4]);
    x[4] + x[5];
    x[1] - x[3] * sin(x[4]);
    x[2] - x[3] * cos(x[4]);
    x[4] + x[6]
  ]

  return jacobian(f⁻¹, cluster_pose)
end

end # module Kinematics
