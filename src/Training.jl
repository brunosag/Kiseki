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

export train_evolution, ESConfig, AbstractBatchScheduler, ConstantScheduler, StepScheduler, LinearScheduler

abstract type AbstractBatchScheduler end

struct ConstantScheduler <: AbstractBatchScheduler
    interval::Int
end
(s::ConstantScheduler)(iteration::Int) = s.interval

struct StepScheduler <: AbstractBatchScheduler
    initial::Int
    step_size::Int
    multiplier::Float64
end
(s::StepScheduler)(iteration::Int) = max(1, round(Int, s.initial * s.multiplier^(iteration ÷ s.step_size)))

struct LinearScheduler <: AbstractBatchScheduler
    initial::Int
    final::Int
    total_steps::Int
end
function (s::LinearScheduler)(iteration::Int)
    progress = min(1.0, iteration / s.total_steps)
    return max(1, round(Int, s.initial + progress * (s.final - s.initial)))
end

Base.@kwdef struct ESConfig
    μ::Int = 100
    λ::Int = 500
    ema_decay::Float32 = 0.99f0
end

function select_population!(
        pop_parents, str_parents, fitness_parents,
        pop_offspring, str_offspring, fitness_offspring,
        buffers, config::ESConfig
    )
    μ = config.μ
    partialsortperm!(buffers.sort_idx, fitness_offspring, 1:μ)
    pop_parents .= pop_offspring[:, buffers.sort_idx[1:μ]]
    str_parents .= str_offspring[:, buffers.sort_idx[1:μ]]
    fitness_parents .= fitness_offspring[buffers.sort_idx[1:μ]]
    return
end

function train_evolution(
        model; I = 10000, batchsize = 1024, checkpoint_Δi = 50,
        batch_scheduler::AbstractBatchScheduler = ConstantScheduler(10),
        resume_file = nothing, lossfn = CrossEntropyLoss(; logits = Val(true)),
        es_config = ESConfig(), save_dir = pwd(), rng = Random.default_rng()
    )
    train_dataloader, test_dataloader = load_MNIST(rng; batchsize)
    θ, s = Lux.setup(rng, model)
    s = Lux.testmode(s)
    θ_flat = ComponentArray(θ)
    N = length(θ_flat)
    axes_flat = getaxes(θ_flat)

    layer_ranges = UnitRange{Int}[]
    offset = 1
    for k in keys(θ_flat)
        len = length(θ_flat[k])
        if len > 0
            push!(layer_ranges, offset:(offset + len - 1))
            offset += len
        end
    end
    L = length(layer_ranges)

    TraceType = NamedTuple{(:i, :L, :σ_var, :σ_mean, :batch_interval, :σ), Tuple{Int, Float32, Float32, Float32, Int, Vector{Float32}}}
    complete_trace = TraceType[]
    best_test_acc = 0.0
    i₀ = 0

    checkpoint_data = isnothing(resume_file) ? nothing : load_checkpoint(resume_file)
    if !isnothing(checkpoint_data)
        θ_latest = checkpoint_data["θ"]
        σ_latest = checkpoint_data["σ"]
        if length(σ_latest) == N || length(σ_latest) != L
            σ_latest = fill(mean(σ_latest), L)
        end
        i₀ = checkpoint_data["i"]
        best_test_acc = get(checkpoint_data, "test_acc", 0.0)
        complete_trace = get(checkpoint_data, "complete_trace", complete_trace)
        θ_ema = get(checkpoint_data, "θ_ema", copy(θ_latest))
    else
        θ_latest = Vector{Float32}(θ_flat)
        σ_latest = fill(0.01f0, L)
        θ_ema = copy(θ_latest)
    end

    λ, μ = es_config.λ, es_config.μ

    pop_parents = repeat(θ_latest, 1, μ)
    str_parents = repeat(σ_latest, 1, μ)
    fitness_parents = fill(Inf32, μ)

    pop_offspring = Matrix{Float32}(undef, N, λ)
    str_offspring = Matrix{Float32}(undef, L, λ)
    fitness_offspring = zeros(Float32, λ)

    buffers = (
        sort_idx = collect(1:λ),
        σ_expanded = Matrix{Float32}(undef, N, λ),
    )

    τ = Float32(1.0 / sqrt(2.0 * sqrt(L)))
    τ′ = Float32(1.0 / sqrt(2.0 * L))

    prev_checkpoint = Ref{String}(isnothing(resume_file) ? "" : resume_file)
    cb_state = CheckpointCallback(
        i₀, 0, checkpoint_Δi, time(), complete_trace,
        model, s, axes_flat, test_dataloader,
        best_test_acc, prev_checkpoint, save_dir, i₀
    )

    train_iter = iterate(train_dataloader)

    generations_on_current_batch = 0
    current_batch_interval = batch_scheduler(i₀ + 1)
    X, y = nothing, nothing

    for global_i in (i₀ + 1):I
        batch_changed = false
        interval_changed = false

        new_interval = batch_scheduler(global_i)
        if new_interval != current_batch_interval
            current_batch_interval = new_interval
            interval_changed = true
        end

        if generations_on_current_batch == 0 || generations_on_current_batch >= current_batch_interval
            if train_iter === nothing
                train_iter = iterate(train_dataloader)
            end
            (X, y), dl_state = train_iter
            train_iter = iterate(train_dataloader, dl_state)
            generations_on_current_batch = 0
            batch_changed = true
        end

        θ_avg = dropdims(sum(pop_parents, dims = 2), dims = 2) ./ Float32(μ)
        σ_avg = dropdims(sum(str_parents, dims = 2), dims = 2) ./ Float32(μ)

        ϵ_0 = randn(rng, Float32, 1, λ)
        ϵ_l = randn(rng, Float32, L, λ)
        ϵ_θ = randn(rng, Float32, N, λ)

        str_offspring .= σ_avg .* exp.(τ′ .* ϵ_0 .+ τ .* ϵ_l)

        for l in 1:L
            buffers.σ_expanded[layer_ranges[l], :] .= view(str_offspring, l:l, :)
        end

        pop_offspring .= θ_avg .+ buffers.σ_expanded .* ϵ_θ

        @batch for j in 1:λ
            θ_ind = ComponentArray(@view(pop_offspring[:, j]), axes_flat)
            ŷ, _ = model(X, θ_ind, s)
            fitness_offspring[j] = lossfn(ŷ, y)
        end

        select_population!(
            pop_parents, str_parents, fitness_parents,
            pop_offspring, str_offspring, fitness_offspring,
            buffers, es_config
        )

        θ_ema .= es_config.ema_decay .* θ_ema .+ (1.0f0 - es_config.ema_decay) .* pop_parents[:, 1]

        current_iter_in_block = (global_i - i₀ - 1) % checkpoint_Δi + 1
        current_block = (global_i - i₀ - 1) ÷ checkpoint_Δi + 1

        trace_record = (
            iteration = current_iter_in_block,
            metadata = (
                θ = copy(pop_parents[:, 1]),
                σ = copy(str_parents[:, 1]),
                L = fitness_parents[1],
                θ_ema = copy(θ_ema),
                batch_interval = current_batch_interval,
                batch_changed = batch_changed,
                interval_changed = interval_changed,
            ),
        )

        cb_state.block = current_block
        cb_state(trace_record)

        generations_on_current_batch += 1
    end

    return cb_state.complete_trace
end

end
