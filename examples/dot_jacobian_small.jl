using Distributed
@everywhere using ForwardDiff

f(x) = [
    x[1]^2 + x[2],
    x[1] * x[2]
]

x0 = [1.0, 1.0]
dx = [0.1, 0.2]

# Calculate Hessians for each component
H1 = ForwardDiff.hessian(x -> f(x)[1], x0)
H2 = ForwardDiff.hessian(x -> f(x)[2], x0)

hessians = @sync @distributed (vcat) for i in 1:2
    ForwardDiff.hessian(x -> f(x)[i], x0)
end

display(hessians)

# Vertically stack Hessians
H_stacked = vcat(H1, H2)

display(H_stacked)

# # Calculate dJ by applying both Hessians to dx and reshaping
# dJ_vec = H_stacked * dx
# # Reshape into a square matrix with dimensions matching the Jacobian
# dJ = reshape(dJ_vec, size(J)...)'

# display(dJ)
