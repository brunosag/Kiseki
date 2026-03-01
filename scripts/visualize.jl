using JLD2
using Plots
using LaTeXStrings
using Statistics

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
σ_mean = [t.σ_mean for t in trace]
σ_var = [t.σ_var for t in trace]

σ_std = sqrt.(σ_var)

p1 = plot(
    iterations, loss,
    title = "Objective Loss",
    xlabel = "Iteration",
    ylabel = "Loss",
    color = :red,
    legend = false,
)

p2 = plot(
    iterations, σ_mean,
    ribbon = σ_std,
    fillalpha = 0.3,
    title = "Strategy Parameter Dynamics",
    xlabel = "Iteration",
    ylabel = L"\mathrm{E}[\sigma] \pm \sqrt{\mathrm{Var}(\sigma)}",
    color = :blue,
    legend = false,
)

fig = plot(p1, p2, layout = (2, 1), size = (1200, 900), margin = 5Plots.mm)

display(fig)

println("Plot rendered. Press Enter to close the window.")
readline()
