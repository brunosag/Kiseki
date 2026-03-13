module LEEA

import Lux
using CUDA, LuxCUDA, Printf, Dates
using Random: Xoshiro, rand!
using StatsBase: Weights, sample, sample!
using OneHotArrays: onecold
using Optimisers: destructure
using Serialization: serialize
using ..Data: load_MNIST
using ..Models: CNN_2C2D_MNIST

export train_LEEA

const logitcrossentropy = Lux.CrossEntropyLoss(; logits = Val(true))

@kwdef struct LEEAConfig
    N::Int = 1000       # population size
    r::Float32 = 0.04   # mutation rate
    m::Float32 = 0.03   # mutation power
    γ::Float32 = 0.99   # mutation power decay
    p::Float32 = 0.4    # selection proportion
    s::Float32 = 0.5    # sexual reproduction proportion
    d::Float32 = 0.2    # fitness inheritance decay
    pat_lim::Int = 50   # generations to wait before decaying m
end

function evaluate_population!(fₚ, fₒ, model, P, X, Y, st, pₐ, p₁, p₂, N, Nₐ, d, gen, re)
    d′ = 1.0f0 - d
    half_d′ = 0.5f0 * d′
    is_first_gen = gen == 1

    for j in 1:N
        θ = re(@view(P[:, j]))
        Ŷ, _ = model(X, θ, st)
        loss = logitcrossentropy(Ŷ, Y)
        f = 1.0f0 / (1.0f0 + Float32(loss))

        if is_first_gen
            fₒ[j] = f
        elseif j <= Nₐ
            fₒ[j] = fₚ[pₐ[j]] * d′ + f
        else
            fₒ[j] = (fₚ[p₁[j - Nₐ]] + fₚ[p₂[j - Nₐ]]) * half_d′ + f
        end
    end
    return
end

function select_parents!(fₒ, pₐ, p₁, p₂, rng, N, p)
    wheel = partialsortperm(fₒ, 1:round(Int, p * N), rev = true)
    weights = Weights(@view fₒ[wheel])

    sample!(rng, wheel, weights, pₐ)
    sample!(rng, wheel, weights, p₁)
    sample!(rng, wheel, weights, p₂)

    collisions = p₁ .== p₂
    while any(collisions)
        n_collisions = sum(collisions)
        p₂[collisions] .= sample(rng, wheel, weights, n_collisions)
        collisions = p₁ .== p₂
    end
    return
end

function reproduce_assexual!(O, P, pₐ, Nₐ, r, m)
    pₐ_gpu = CuArray(pₐ)

    u₁ = CUDA.rand(Float32, size(P, 1), Nₐ)
    u₂ = CUDA.rand(Float32, size(P, 1), Nₐ)

    @views O[:, 1:Nₐ] .= P[:, pₐ_gpu] .+ (u₁ .< r) .* m .* (2.0f0 .* u₂ .- 1.0f0)
    return
end

function reproduce_sexual!(O, P, p₁, p₂, Nₛ, Nₐ)
    p₁_gpu = CuArray(p₁)
    p₂_gpu = CuArray(p₂)

    u = CUDA.rand(Float32, size(P, 1), Nₛ)

    @views O[:, (Nₐ + 1):end] .= ifelse.(u .< 0.5f0, P[:, p₁_gpu], P[:, p₂_gpu])
    return
end

function evaluate_best(fₒ, P, st, model, dataloader, dev, gen, re, t₀, best_acc, pat, pat_lim)
    best_idx = argmax(fₒ)
    θ = re(@view(P[:, best_idx]))

    correct, total = 0, 0

    for (X, Y) in dataloader
        X_gpu = X |> dev
        Y_gpu = Y |> dev

        Ŷ, _ = model(X_gpu, θ, st)

        correct += sum(onecold(Array(Ŷ)) .== onecold(Array(Y_gpu)))
        total += size(Y, 2)
    end

    acc = (correct / total) * 100.0
    Δt = time() - t₀
    base_str = @sprintf "i = %i      Δt = %.2fs      Accuracy = %.2f%%" gen Δt acc

    if acc > best_acc
        println(base_str)
        time_str = Dates.format(Dates.now(), "yyyy-mm-ddTHHMMSS")
        filename = @sprintf "LEEA_%ia_%ii_%s.jls" round(Int, acc * 100.0) gen time_str
        serialize(filename, θ |> Lux.cpu_device())
        return acc
    else
        println(base_str, @sprintf " [Best: %.2f%% %i/%i]" best_acc (pat + 1) pat_lim)
    end

    return best_acc
end

function train_LEEA(; seed::Int = 42, batchsize::Int = 1000, generations::Int = 900000)
    (; N, r, m, γ, p, s, d, pat_lim) = LEEAConfig()

    rng = Xoshiro(seed)
    dev = Lux.gpu_device()

    train_dataloader, test_dataloader = load_MNIST(; rng, batchsize, balanced = true)
    model = CNN_2C2D_MNIST

    θ, st = Lux.setup(rng, model)
    st = st |> dev
    st = Lux.testmode(st)
    θ_flat, re = destructure(θ)
    θ_len = length(θ_flat)

    P = stack([destructure(Lux.initialparameters(rng, model))[1] for _ in 1:N]) |> dev
    O = similar(P)

    Nₛ = round(Int, s * N)
    Nₐ = N - Nₛ

    alloc(T, n) = Vector{T}(undef, n)
    fₚ, fₒ = alloc(Float32, N), alloc(Float32, N)
    pₐ, p₁, p₂ = alloc(Int, Nₐ), alloc(Int, Nₛ), alloc(Int, Nₛ)

    best_acc = 0.0
    pat = 0

    for i in 1:generations
        t₀ = time()
        X, Y = popfirst!(train_dataloader) |> dev

        evaluate_population!(fₚ, fₒ, model, P, X, Y, st, pₐ, p₁, p₂, N, Nₐ, d, i, re)
        select_parents!(fₒ, pₐ, p₁, p₂, rng, N, p)
        reproduce_assexual!(O, P, pₐ, Nₐ, r, m)
        reproduce_sexual!(O, P, p₁, p₂, Nₛ, Nₐ)

        acc = evaluate_best(fₒ, P, st, model, test_dataloader, dev, i, re, t₀, best_acc, pat, pat_lim)

        if acc > best_acc
            best_acc = acc
            pat = 0
        else
            pat += 1
        end

        if pat >= pat_lim
            println("m: $m -γ→ $(m * γ)")
            m *= γ
            pat = 0
        end

        P, O = O, P
        fₚ, fₒ = fₒ, fₚ
    end
    return
end

end
