using MRTA
using ControlSystems
using CairoMakie
using LaTeXStrings
using Logging

num_robots = 1  # Number of robots
num_tasks = 3  # Number of tasks
DOF = 3  # Number of Degrees of Freedom (DOF)
control_dim = 3  # Control vector dimension

T = 1000
t = LinRange(0, 30, T)
dt = abs(t[1] - t[2])

Izz = 1
m = 10
bx = 5
by = 5
bθ = 1

goals = [
    10 10 3*π/2;
    15 10 3*π/2;
    10 15 3*π/2
]

A, B = cluster_matrices(m, Izz, bx, by, bθ)
sys = cluster_state_space(m, Izz, bx, by, bθ)

r = zeros(T, 2 * DOF)
ṙ = zeros(T, 2 * DOF)

r[1, :] = [5 1 0 15 1 0]

c = zeros(T, 2 * DOF)
ċ = zeros(T, 2 * DOF)

c[1, :] = [10 1 0 5 1 0]

x = zeros(T, 12)
slack = zeros(T, num_tasks)

sysd = c2d(sys, dt)

Ad = sysd.A
Bd = sysd.B

for (i, tᵢ) in enumerate(t)
    u, α, δ = run_optimization_problem(c[i, 1:3], goals)
    control = [u; 0; 0; 0]

    # Log optimization results
    @info "Time step $i" control_input=u assignment_vars=α slack_vars=δ

    slack[i, :] = δ

    if i <= T - 1
        J = forward_kin_jacobian(r[i, :])

        J⁻¹ = inv(J)
        δ = J⁻¹' * A * J⁻¹
        υ = J⁻¹' * B * J⁻¹

        J_invkin = inverse_kin_jacobian(c[i, :])
        J̇ = dot_forward_jacobian(r[i, :], ṙ[i, :])
        μ = υ * (ċ[i, :] - δ * J̇ * J_invkin * ċ[i, :])

        F = δ * control + μ
        γ = J' * F

        x[i, 1:3] = r[i, 1:3]
        x[i, 4:6] = ṙ[i, 1:3]
        x[i, 7:9] = r[i, 4:end]
        x[i, 10:end] = ṙ[i, 4:end]

        x[i+1, :] = Ad * x[i, :] + Bd * γ

        r[i+1, 1:3] = x[i+1, 1:3]
        ṙ[i+1, 1:3] = x[i+1, 4:6]
        r[i+1, 4:6] = x[i+1, 7:9]
        ṙ[i+1, 4:6] = x[i+1, 10:end]

        c[i+1, :] = forward_pose(r[i+1, :])
    end
end

function plot_robots_trajectory_makie(c_plot, r_plot, goals_plot)
    fig = Figure(size=(800, 650)) # Increased resolution, adjusted height for legend
    ax = Axis(fig[1, 1],
        title="Planar Robot Trajectory",
        titlesize=20,
        xlabel=L"x_c / x_i",
        xlabelsize=18,
        xticklabelsize=14,
        ylabel=L"y_c / y_i",
        ylabelsize=18,
        yticklabelsize=14,
        leftspinevisible=true,
        rightspinevisible=false,
        topspinevisible=false,
        bottomspinevisible=true,
        xgridvisible=true,
        ygridvisible=true,
        xgridstyle=:dash,
        ygridstyle=:dash,
        xgridcolor=:gray85, # Lighter grid
        ygridcolor=:gray85  # Lighter grid
    )

    # Plot trajectories
    line_cluster = lines!(ax, c_plot[:, 1], c_plot[:, 2], linestyle=:dash, color=:magenta, label="cluster", linewidth=2.0)
    line_robot1 = lines!(ax, r_plot[:, 1], r_plot[:, 2], linestyle=:dash, color=:blue, label="robot 1", linewidth=2.0)
    line_robot2 = lines!(ax, r_plot[:, 4], r_plot[:, 5], linestyle=:dash, color=:red, label="robot 2", linewidth=2.0)

    # Plot goal points
    scatter!(ax, goals_plot[:, 1], goals_plot[:, 2], marker=:circle, color=:black, markersize=12)

    # Add annotations for goal points
    annotations_data = [
        (L"p_1", goals_plot[1, 1] + 0.5, goals_plot[1, 2] + 0.5),
        (L"p_2", goals_plot[2, 1] + 0.5, goals_plot[2, 2] - 0.5),
        (L"p_3", goals_plot[3, 1] - 0.5, goals_plot[3, 2] - 0.5)
    ]

    for (label_text, x_pos, y_pos) in annotations_data
        text!(ax, x_pos, y_pos, text=label_text, fontsize=14, align=(:left, :bottom))
    end

    # Add legend for line plots at the bottom, horizontal, no frame, no title
    Legend(fig[2, 1], # Position below the axis
        [line_cluster, line_robot1, line_robot2], # Plot elements
        ["cluster", "robot 1", "robot 2"], # Labels
        orientation=:horizontal,
        labelsize=14,
        framevisible=false,
        titlevisible=false, # Explicitly hide title
        tellheight=true, # Allow legend to take vertical space
        padding=(0, 0, 0, 0) # Reduce padding around legend entries
    )

    return fig
end

# After the simulation loop, you can call the plotting function:
fig = plot_robots_trajectory_makie(c, r, goals)
display(fig) # or save("trajectory_makie.png", fig)
# To save the plot, you might need to ensure the output directory exists or provide a full path.
