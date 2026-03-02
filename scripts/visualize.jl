using JLD2
using Plots
using LaTeXStrings

if length(ARGS) != 1
    println(stderr, "Usage: julia visualize.jl <path_to_checkpoint.jld2>")
    exit(1)
end

checkpoint_path = ARGS[1]

if !isfile(checkpoint_path)
    println(stderr, "Error: File not found at $checkpoint_path")
    exit(1)
end

data = jldopen(checkpoint_path, "r") do file
    Dict(k => file[k] for k in keys(file))
end

trace = data["complete_trace"]

iterations = [t.i for t in trace]
loss = [t.L for t in trace]

σ_matrix = hcat([t.σ for t in trace]...)'
num_layers = size(σ_matrix, 2)

p1 = plot(
    iterations, loss,
    title = "Objective Loss & Test Accuracy",
    xlabel = "Iteration",
    ylabel = "Loss",
    color = :red,
    legend = :topleft,
    label = "Training Loss"
)

if haskey(data, "accuracy_trace")
    acc_trace = data["accuracy_trace"]
    if !isempty(acc_trace)
        acc_iters = [t[1] for t in acc_trace]
        acc_vals = [t[2] <= 1.0 ? t[2] * 100.0 : t[2] for t in acc_trace]

        p1_acc = twinx(p1)
        plot!(
            p1_acc, acc_iters, acc_vals,
            color = :green,
            linewidth = 2.0,
            ylabel = "Test Accuracy (%)",
            legend = :topleft,
            label = "Test Accuracy",
            ylims = (0, 100)
        )
    end
end

layer_labels = ["Conv 1 (Spatial)", "Conv 2 (Spatial)", "Dense 1 (Global)", "Dense 2 (Classification)"]

if num_layers != 4
    layer_labels = ["Layer $i" for i in 1:num_layers]
end

p2 = plot(
    iterations, σ_matrix,
    title = "Layer-wise Strategy Parameter Dynamics",
    xlabel = "Iteration",
    ylabel = L"\sigma_l",
    labels = reshape(layer_labels, 1, :),
    legend = :topright,
    linewidth = 1.5,
    alpha = 0.8
)

fig = plot(p1, p2, layout = (2, 1), size = (1200, 900), margin = 5Plots.mm)

display(fig)

println("Plot rendered. Press Enter to close the window.")
readline()
