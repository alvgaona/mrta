module MRTA

include("Kinematics.jl")
include("Cluster.jl")
include("Optimization.jl")

using .Kinematics
using .Cluster
using .Optimization

# Export the functions from submodules
export cluster_matrices,
       cluster_state_space,
       run_optimization_problem,
       forward_kin_jacobian,
       inverse_kin_jacobian,
       dot_forward_jacobian,
       forward_pose

end # module MRTA
