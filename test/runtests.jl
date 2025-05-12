using Test

@testset "MRTA" begin
    include("kinematics_tests.jl")
    include("cluster_tests.jl")
end
