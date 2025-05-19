using Test
using MRTA.Kinematics
using LinearAlgebra

@testset "Kinematics" begin
    @testset "Forward Pose" begin
        robots_pose = [3.5, -2.2, 1.0, 0.0, 3.4, 33.3]
        cluster_pose = forward_pose(robots_pose)

        @test all(isapprox.(cluster_pose, [1.75, 0.6, 3.3018, 2.5829, -1.5829, 30.7170], atol=1e-3))
    end

    @testset "Inverse Pose" begin
        cluster_pose = [1.75, 0.6, 3.3018, 2.5829, -1.5829, 30.7170]
        robots_pose = inverse_pose(cluster_pose)

        @test all(isapprox.(robots_pose, [3.5, -2.2, 1.0, 0.0, 3.4, 33.3], atol=1e-3))
    end

    @testset "Forward Kinematics Jacobian" begin
        robots_pose = [3.5, -2.2, 1.0, 0, 3.4, 33.3]
        J = forward_kin_jacobian(robots_pose)

        expected = [
            0.5 0 0 0.5 0 0;
            0 0.5 0 0 0.5 0;
            0.26499947 -0.42399915 0 -0.26499947 0.42399915 0;
            0.12841091 0.08025682 0 -0.12841091 -0.08025682 0;
            -0.12841091 -0.08025682 1 0.12841091 0.08025682 0;
            -0.12841091 -0.08025682 0 0.12841091 0.08025682 1
        ]

        @test all(isapprox.(J, expected, atol=1e-3))
    end

    @testset "Inverse Kinematics Jacobian" begin
        cluster_pose = [1.8, 0.6, 3.3, 2.6, -1.6, 30.7]

        J⁻¹ = inverse_kin_jacobian(cluster_pose)

        expected = [
            1 0 0.51550137 -2.82773289 0 0;
            0 1 -0.85688875 -1.70115453 0 0;
            0 0 0 1 1 0;
            1 0 -0.51550137 2.82773289 0 0;
            0 1 0.85688875 1.70115453 0 0;
            0 0 0 1 0 1
        ]

        @test all(isapprox.(J⁻¹, expected, atol=1e-3))
    end

    @testset "Dot Forward Jacobian" begin
        robots_pose = [3.5, -2.2, 1.0, 0, 3.4, 33.3]
        robots_velocities = [-1.2, 2, 1.0, 2.2, -3.4, -1.0]

        Jₜ = dot_forward_jacobian(robots_pose, robots_velocities)

        expected = [
            0 0 0 0 0 0;
            0 0 0 0 0 0;
            -0.00136115 -0.00085072 0 0.00136115 0.00085072 0;
            0.1243401 0.0771393 0 -0.1243401 -0.0771393 0;
            -0.1243401 -0.0771393 0 0.1243401 0.0771393 0;
            -0.1243401 -0.0771393 0 0.1243401 0.0771393 0
        ]

        @test all(isapprox.(Jₜ, expected, atol=1e-3))
    end
end