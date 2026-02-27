using Lux, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Printf
using MLDatasets: MNIST
using SimpleChains: SimpleChains

function load_MNIST(rng; batchsize::Int)
    preprocess(data) = (
        reshape(data.features, 28, 28, 1, :),
        onehotbatch(data.targets, 0:9),
    )

    X_train, y_train = preprocess(MNIST(:train))
    X_test, y_test = preprocess(MNIST(:test))

    return (
        DataLoader((X_train, y_train); batchsize, shuffle = true, rng),
        DataLoader((X_test, y_test); batchsize, shuffle = false),
    )
end

function accuracy(model, θ, s, dataloader)
    s_test = Lux.testmode(s)
    correct, total = 0, 0

    for (X, y) in dataloader
        pred = Array(first(model(X, θ, s_test)))
        correct += sum(onecold(pred) .== onecold(y))
        total += size(y, 2)
    end

    return (correct / total) * 100
end

function train(
        model;
        rng = Random.default_rng(),
        batchsize = 128,
        n_epochs = 10,
        optimizer = Adam(3.0f-4),
        lossfn = CrossEntropyLoss(; logits = Val(true)),
        ad_backend = AutoZygote(),
    )
    train_dataloader, test_dataloader = load_MNIST(rng; batchsize)
    θ, s = Lux.setup(rng, model)
    ts = Training.TrainState(model, θ, s, optimizer)

    for epoch in 1:n_epochs
        Δt = @elapsed begin
            for (X, y) in train_dataloader
                grads, loss, stats, ts = Training.single_train_step!(
                    ad_backend, lossfn, (X, y), ts
                )
            end
        end

        if epoch == 1 || epoch % 5 == 0 || epoch == n_epochs
            train_acc = accuracy(
                model, ts.parameters, ts.states, train_dataloader
            )
            test_acc = accuracy(
                model, ts.parameters, ts.states, test_dataloader
            )
            @printf "[%2d/%2d] \t Time %.2fs \t Training Accuracy: %.2f%% \t Test Accuracy: %.2f%%\n" epoch n_epochs Δt train_acc test_acc
        end
    end

    return ts
end

model = Chain(
    Conv((3, 3), 1 => 8, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 8 => 16, relu),
    MaxPool((2, 2)),
    FlattenLayer(3),
    Dense(5 * 5 * 16 => 120, relu),
    Dense(120 => 10)
)

simple_chains_model = ToSimpleChainsAdaptor((28, 28, 1))(model)
trained_state = train(simple_chains_model; rng = Xoshiro(42))
