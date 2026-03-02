module Training

using Lux
using Random
using Printf
using Polyester
using ComponentArrays
using Statistics
using ..Data: load_MNIST
using ..Evaluation: accuracy
using ..Checkpoints: load_checkpoint
using ..Callbacks: CheckpointCallback

export train_evolution, ESConfig

Base.@kwdef struct ESConfig
    μ::Int = 100
    λ::Int = 500
    ema_decay::Float32 = 0.99f0
end

function train_evolution(
        model; I = 40000, batchsize = 1024, checkpoint_Δi = 10,
        resume_file = nothing, lossfn = CrossEntropyLoss(; logits = Val(true)),
        es_config = ESConfig(), save_dir = pwd(), rng = Random.default_rng()
    )
    train_dataloader, test_dataloader = load_MNIST(rng; batchsize)
    θ, s = Lux.setup(rng, model)
    s = Lux.testmode(s)

    θ_flat = ComponentArray(θ)
    N = length(θ_flat)
    axes_flat = getaxes(θ_flat)

    λ, μ = es_config.λ, es_config.μ
    τ = Float32(1.0 / sqrt(N))

    checkpoint_data = isnothing(resume_file) ? nothing : load_checkpoint(resume_file)

    i₀ = isnothing(checkpoint_data) ? 0 : checkpoint_data["i"]
    best_test_acc = isnothing(checkpoint_data) ? 0.0 : get(checkpoint_data, "test_acc", 0.0)

    TraceType = NamedTuple{(:i, :L, :σ), Tuple{Int, Float32, Float32}}
    complete_trace = isnothing(checkpoint_data) ?
        TraceType[] :
        get(checkpoint_data, "complete_trace", TraceType[])
    accuracy_trace = isnothing(checkpoint_data) ?
        Tuple{Int, Float64}[] :
        get(checkpoint_data, "accuracy_trace", Tuple{Int, Float64}[])

    θ_latest = isnothing(checkpoint_data) ? Vector{Float32}(θ_flat) : checkpoint_data["θ"]
    σ_latest = isnothing(checkpoint_data) ? 0.01f0 : Float32(checkpoint_data["σ"])
    θ_ema = isnothing(checkpoint_data) ?
        copy(θ_latest) : get(checkpoint_data, "θ_ema", copy(θ_latest))

    pop_parents = repeat(θ_latest, 1, μ)
    str_parents = fill(σ_latest, μ)
    fitness_parents = fill(Inf32, μ)

    pop_offspring = Matrix{Float32}(undef, N, λ)
    str_offspring = Vector{Float32}(undef, λ)
    fitness_offspring = zeros(Float32, λ)
    sort_idx = collect(1:λ)

    prev_checkpoint = Ref{String}(isnothing(resume_file) ? "" : resume_file)
    cb_state = CheckpointCallback(
        checkpoint_Δi, time(), complete_trace, accuracy_trace,
        model, s, axes_flat, test_dataloader,
        best_test_acc, prev_checkpoint, save_dir
    )

    data_iter = Iterators.Stateful(Iterators.cycle(train_dataloader))
    (X, y) = popfirst!(data_iter)

    for i in (i₀ + 1):I
        inner_i = mod1(i - i₀, checkpoint_Δi)
        block = cld(i - i₀, checkpoint_Δi)

        if inner_i == 1 && i > (i₀ + 1)
            (X, y) = popfirst!(data_iter)
        end

        θ_avg = dropdims(sum(pop_parents, dims = 2), dims = 2) ./ Float32(μ)
        σ_avg = mean(str_parents)

        ϵ_0 = randn(rng, Float32, λ)
        ϵ_θ = randn(rng, Float32, N, λ)

        str_offspring .= σ_avg .* exp.(τ .* ϵ_0)
        pop_offspring .= θ_avg .+ reshape(str_offspring, 1, λ) .* ϵ_θ

        @batch for j in 1:λ
            θ_ind = ComponentArray(@view(pop_offspring[:, j]), axes_flat)
            ŷ, _ = model(X, θ_ind, s)
            fitness_offspring[j] = lossfn(ŷ, y)
        end

        partialsortperm!(sort_idx, fitness_offspring, 1:μ)
        best_idx = @view sort_idx[1:μ]

        pop_parents .= pop_offspring[:, best_idx]
        str_parents .= str_offspring[best_idx]
        fitness_parents .= fitness_offspring[best_idx]

        θ_ema .= es_config.ema_decay .* θ_ema .+
            (1.0f0 - es_config.ema_decay) .* @view(pop_parents[:, 1])

        inner_i = mod1(i - i₀, checkpoint_Δi)
        block = cld(i - i₀, checkpoint_Δi)

        trace_record = (
            iteration = inner_i,
            metadata = (
                θ = copy(@view(pop_parents[:, 1])),
                σ = str_parents[1],
                L = fitness_parents[1],
                θ_ema = copy(θ_ema),
            ),
        )

        cb_state(i, copy(@view(pop_parents[:, 1])), copy(θ_ema), fitness_parents[1], str_parents[1])
    end

    return cb_state.complete_trace
end

end
