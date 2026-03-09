module LEEA

import Lux
using Random: Xoshiro, rand!
using Polyester: @batch
using StatsBase: Weights, sample
using Base.Threads: nthreads, threadid
using ..Data: load_MNIST
using ..Models: CNN_2C2D_MNIST

export train_LEEA

const logitcrossentropy = Lux.CrossEntropyLoss(; logits = Val(true))

@kwdef struct LEEAConfig
    N::Int = 1000       # population size
    r::Float32 = 0.04   # mutation rate
    m::Float32 = 0.03   # mutation power
    γₘ::Float32 = 0.99  # mutation power decay
    pₛ::Float32 = 0.4   # selection proportion
    s::Float32 = 0.5    # sexual reproduction proportion
    γᵢ::Float32 = 0.2   # fitness inheritance decay
end

function mutate!(O, P, as_idx, U₁, U₂, rngs, r, m, γₘ)
    return @batch for i in 1:length(as_idx)
        tid = threadid()
        u₁ = @view U₁[:, tid]
        u₂ = @view U₂[:, tid]

        rand!(rngs[tid], u₁)
        rand!(rngs[tid], u₂)

        p_idx = as_idx[i]

        @. O[:, i] = P[:, p_idx] + (u₁ < r) * m * (2.0f0 * u₂ - 1.0f0)
    end
end

function train_LEEA(; seed::Int, batchsize::Int, generations::Int)
    n_threads = nthreads()
    rng = Xoshiro(seed)
    rngs = [Xoshiro(seed + t) for t in 1:n_threads]
    (; N, r, m, γₘ, pₛ, s, γᵢ) = LEEAConfig()

    train_dataloader, _ = load_MNIST(; rng, batchsize, balanced = true)
    model = CNN_2C2D_MNIST
    _, st = Lux.setup(rng, model)
    st = Lux.testmode(st)
    θ_len = Lux.parameterlength(model)

    P = stack(Vector(Lux.initialparameters(rng, model).params) for _ in 1:N)
    O = similar(P)

    f = Vector{Float32}(undef, N)

    num_as = round(Int, (1 - s) * N)
    U₁ = Matrix{Float32}(undef, θ_len, num_as)
    U₂ = Matrix{Float32}(undef, θ_len, num_as)

    for i in 1:generations
        X, Y = popfirst!(train_dataloader)

        @batch for j in 1:N
            θⱼ = NamedTuple{(:params,)}((@view(P[:, j]),))
            Ŷ, _ = model(X, θⱼ, st)
            f[j] = 1.0f0 / (1.0f0 + logitcrossentropy(Ŷ, Y))
        end

        wheel_idx = partialsortperm(f, 1:round(Int, pₛ * N), rev = true)
        wheel_w = Weights(@view f[wheel_idx])

        as_idx = sample(rng, wheel_idx, wheel_w, num_as)

        mutate!(O, P, as_idx, U₁, U₂, rngs, r, m, γₘ)
        m *= γₘ

        @show O[1:10, 1]
    end

    return
end

end
