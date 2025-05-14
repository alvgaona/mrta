using ForwardDiff
using MRTA

function main()
    x0 = [2.5, -1.3, 0.7, 4.2, -0.8, 3.1]
    dx = [0.2, 0.15, 0.05, 0.25, 0.1, 0.3]

    f(x) = [
        (x[1] + x[4]) / 2;
        (x[2] + x[5]) / 2;
        1 / 2 * sqrt((x[1] - x[4])^2 + (x[2] - x[5])^2);
        atan(x[2] - x[5], x[1] - x[4]);
        x[3] - atan(x[2] - x[5], x[1] - x[4]);
        x[6] - atan(x[2] - x[5], x[1] - x[4])
    ]

    # Calculate each Hessian in a loop and stack them vertically
    H = zeros(0, length(x0))  # Initialize an empty matrix to store Hessians
    for i in 1:length(f(x0))
        Hi = ForwardDiff.hessian(x -> f(x)[i], x0)
        H = vcat(H, Hi)
    end

    display(H)

    # Calculate dJ by applying both Hessians to dx and reshaping
    dJ_vec = H * dx
    # Reshape into a square matrix with dimensions matching the Jacobian
    dJ = reshape(dJ_vec, 6, 6)'

    @assert dJ ≈ dot_forward_jacobian(x0, dx)

    return dJ
end

main()
