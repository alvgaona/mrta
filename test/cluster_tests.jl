using Test
using MRTA
using LinearAlgebra
using SparseArrays

@testset "Cluster" begin
    @testset "Cluster Matrices" begin
        Izz = 1
        m = 10
        bx = 5
        by = 5
        bθ = 1

        A_matrix, B_matrix = cluster_matrices(m, Izz, bx, by, bθ) # Renamed A, B to avoid conflict if other tests use these names

        Aᵢ = sparse(diagm([10, 10, 1]))
        Bᵢ = sparse(diagm([5, 5, 1]))

        @test all(isapprox.(A_matrix, blockdiag(Aᵢ, Aᵢ), atol=1e-3))
        @test all(isapprox.(B_matrix, blockdiag(Bᵢ, Bᵢ), atol=1e-3))
    end

    @testset "Cluster State Space" begin
        Izz = 1
        m = 10
        bx = 5
        by = 5
        bθ = 1

        sys = cluster_state_space(m, Izz, bx, by, bθ)

        Aᵢ_ss = sparse([ # Renamed to avoid potential conflict
            0 0 0 1 0 0;
            0 0 0 0 1 0;
            0 0 0 0 0 1;
            0 0 0 -0.5 0 0;
            0 0 0 0 -0.5 0;
            0 0 0 0 0 -1
        ])

        Bᵢ_ss = sparse([ # Renamed to avoid potential conflict
            0 0 0;
            0 0 0;
            0 0 0;
            0.1 0 0;
            0 0.1 0;
            0 0 1
        ])

        @test all(isapprox.(sys.A, blockdiag(Aᵢ_ss, Aᵢ_ss), atol=1e-3))
        @test all(isapprox.(sys.B, blockdiag(Bᵢ_ss, Bᵢ_ss), atol=1e-3))
        @test all(isapprox.(sys.C, I(12), atol=1e-3))
        @test all(isapprox.(sys.D, zeros(12, 6), atol=1e-3))
    end
end