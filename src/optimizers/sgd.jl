@kwdef struct SGD <: AbstractOptimizer
    η::Float32 = 0.01f0
end

mutable struct SGDState <: AbstractOptimizerState
    θ
    ts
end

function init(opt::SGD, model, rng, dev)
    θ, st = Lux.setup(rng, model) |> dev
    ts = Lux.Training.TrainState(model, θ, st, Descent(opt.η))

    return SGDState(θ, ts)
end

function step!(
        opt::SGD, ops::SGDState, re, model, st, X, Y, rng, best_acc, val_set, evaluate
    )
    grads, loss, stats, ops.ts = Lux.Training.single_train_step!(
        AutoZygote(), logitcrossentropy, (X, Y), ops.ts
    )

    acc = evaluate(ops.θ, model, st, val_set)

    return loss, acc, nothing
end
