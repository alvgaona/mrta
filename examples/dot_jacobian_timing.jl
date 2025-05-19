using MRTA: dot_forward_jacobian_exact,
    dot_forward_jacobian
using BenchmarkTools
using Random

# Generate a single set of random test vectors
Random.seed!(123)  # Set seed for reproducibility

# Generate random robot poses (smaller values)
r = rand(6) * 10 .- 5  # Values between -5 and 5

# Generate random robot velocities (smaller values)
dr = (rand(6) * 0.5) .- 0.25  # Values between -0.25 and 0.25

# Verify that both implementations produce the same result
@assert dot_forward_jacobian(r, dr) ≈ dot_forward_jacobian_exact(r, dr)

println("\nBenchmarking Numerical Implementation:")
numeric_benchmark = @benchmark dot_forward_jacobian($r, $dr)
display(numeric_benchmark)

println("\nBenchmarking Analytical Implementation:")
analytical_benchmark = @benchmark dot_forward_jacobian_exact($r, $dr)
display(analytical_benchmark)
