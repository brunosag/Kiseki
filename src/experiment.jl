const logitcrossentropy = Lux.CrossEntropyLoss(; logits = Val(true))

@kwdef struct Experiment
    seed::Int = 42
    batchsize::Int = 500
    max_i::Int = 500000
    target_acc::Float64 = 100.0
    opt::AbstractOptimizer = LEEA()
end

mutable struct ExperimentState
    rng
    ops
    train_loader
    val_set
    test_loader
    best_acc
    i
end

function init(exp::Experiment, model, dev)
    LuxCUDA.CUDA.seed!(exp.seed)

    rng = Xoshiro(exp.seed)
    ops = init(exp.opt, model, rng, dev)
    train_loader, val_set, test_loader = load_MNIST(rng, exp.batchsize, dev, val_size = 10_000)
    best_acc = 0.0
    i = 1

    return ExperimentState(rng, ops, train_loader, val_set, test_loader, best_acc, i)
end

function evaluate(θ, model, st, val_set)
    X, Y = val_set
    Ŷ, _ = model(X, θ, st)

    correct = sum(onecold(Array(Ŷ), 0:9) .== Y)
    total = length(Y)

    return (correct / total) * 100.0
end

function run(exp::Experiment; est::Union{ExperimentState, Nothing} = nothing)
    dev = Lux.gpu_device()
    model = CNN_2C2D_MNIST
    st = Lux.testmode(Lux.initialstates(TaskLocalRNG(), model))
    re = destructure(Lux.initialparameters(TaskLocalRNG(), model))[2]

    if isnothing(est)
        est = init(exp, model, dev)
    end

    while est.i <= exp.max_i && est.best_acc < exp.target_acc
        t₀ = time()
        X, Y = popfirst!(est.train_loader) |> dev

        L, acc, etc = step!(exp.opt, est.ops, re, model, st, X, Y, est.rng, est.best_acc, est.val_set, evaluate)

        Δt = time() - t₀
        base_log = @sprintf "i = %-*d      Δt = %.2fs      L = %.4f      %sAcc. = %-*.2f%%" ndigits(exp.max_i) est.i Δt L (!isnothing(etc) ? ("$etc      ") : "") 5 acc

        if acc > est.best_acc
            println(base_log)
            est.best_acc = acc
        else
            @printf "%s [Best: %-*.2f%%]\n" base_log 5 est.best_acc
        end

        est.i += 1
    end
    return
end
