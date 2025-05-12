module Cluster

using ControlSystems
using LinearAlgebra
using SparseArrays
using ControlSystems

export cluster_matrices, cluster_state_space

"""
    cluster_matrices(m, Izz, bx, by, bθ)

Create block diagonal mass and damping matrices for a two-agent system.

# Arguments
- `m`: Mass of each agent
- `Izz`: Moment of inertia about z-axis
- `bx`: Damping coefficient in x direction
- `by`: Damping coefficient in y direction
- `bθ`: Damping coefficient for rotation

# Returns
- Tuple of two sparse block diagonal matrices (A, B)
"""
function cluster_matrices(m::T, Izz::T, bx::T, by::T, bθ::T) where {T<:Real}
    # Create mass matrix for a single agent
    Aᵢ = sparse(diagm([m, m, Izz]))

    # Create damping matrix for a single agent
    Bᵢ = sparse(diagm([bx, by, bθ]))

    # Create block diagonal matrices for two agents
    return blockdiag(Aᵢ, Aᵢ), blockdiag(Bᵢ, Bᵢ)
end

"""
    cluster_state_space(m, Izz, bx, by, bθ)

Create a state-space representation for a two-agent system.

# Arguments
- `m`: Mass of each agent
- `Izz`: Moment of inertia about z-axis
- `bx`: Damping coefficient in x direction
- `by`: Damping coefficient in y direction
- `bθ`: Damping coefficient for rotation

# Returns
- State-space system object representing the two-agent system dynamics
"""
function cluster_state_space(m::T, Izz::T, bx::T, by::T, bθ::T) where {T<:Real}
    Ai = sparse([
        0 0 0 1 0 0;
        0 0 0 0 1 0;
        0 0 0 0 0 1;
        0 0 0 -bx/m 0 0;
        0 0 0 0 -by/m 0;
        0 0 0 0 0 -bθ/Izz
    ])

    Bi = sparse([
        0 0 0;
        0 0 0;
        0 0 0;
        1/m 0 0;
        0 1/m 0;
        0 0 1/Izz
    ])

    A = blockdiag(Ai, Ai)
    B = blockdiag(Bi, Bi)

    C = sparse(I(12))
    D = sparse(zeros(12, 6))

    return ss(A, B, C, D)
end

end # end module Cluster
