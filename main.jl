using Lux, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Statistics, Printf, Reactant
using MLDatasets: MNIST
using SimpleChains: SimpleChains

Reactant.set_default_backend("cpu")

function load_MNIST(; batch_size, train_split)
    dataset = MNIST(:train)
    imgs = dataset.features
    labels_raw = dataset.targets

    x_data = Float32.(
        reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3))
    )
    y_data = onehotbatch(labels_raw, 0:9)
    (x_train, y_train), (x_test, y_test) = splitobs(
        (x_data, y_data); at = train_split
    )

    return (
        DataLoader(
            collect.((x_train, y_train));
            batchsize = batch_size, shuffle = true, partial = false
        ),
        DataLoader(
            collect.((x_test, y_test));
            batchsize = batch_size, shuffle = false, partial = false
        ),
    )
end

lux_model = Chain(
    MaxPool((2, 2)),
    FlattenLayer(3),
    Dense(196 => 20, relu),
    Dense(20 => 10)
)

adaptor = ToSimpleChainsAdaptor((28, 28, 1))
simple_chains_model = adaptor(lux_model)

const lossfn = CrossEntropyLoss(; logits = Val(true))

function accuracy(model, θ, s, dataloader, device)
    total_correct, total = 0, 0
    s = Lux.testmode(s)

    for (x_raw, y_raw) in dataloader
        x, y = (x_raw, y_raw) |> device
        target_class = onecold(Array(y))
        predicted_class = onecold(Array(first(model(x, θ, s))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end

    return total_correct / total
end

function train(model, device = cpu_device(); rng = Random.default_rng(), kwargs...)
    train_dataloader, test_dataloader = load_MNIST(
        batch_size = 128,
        train_split = 0.9
    )
    θ, s = Lux.setup(rng, model) |> device

    ts = Training.TrainState(model, θ, s, Adam(3.0f-4))

    if device isa ReactantDevice
        vjp = AutoEnzyme()
        x_ra = first(test_dataloader)[1] |> device
        model_compiled = @compile model(x_ra, θ, Lux.testmode(s))
    else
        vjp = AutoZygote()
        model_compiled = model
    end

    n_epochs = 150
    train_acc, test_acc = 0.0, 0.0

    for epoch in 1:n_epochs
        stime = time()
        for (x_raw, y_raw) in train_dataloader
            x, y = (x_raw, y_raw) |> device
            _, _, _, ts = Training.single_train_step!(
                vjp, lossfn, (x, y), ts
            )
        end
        ttime = time() - stime

        train_acc = accuracy(
            model_compiled, ts.parameters, ts.states, train_dataloader, device
        ) * 100
        test_acc = accuracy(
            model_compiled, ts.parameters, ts.states, test_dataloader, device
        ) * 100

        if epoch == 1 || epoch % 10 == 0 || epoch == n_epochs
            @printf "[%2d/%2d] \t Time %.2fs \t Training Accuracy: %.2f%% \t Test Accuracy: %.2f%%\n" epoch n_epochs ttime train_acc test_acc
        end
    end

    return train_acc, test_acc
end

train_acc, test_acc = train(simple_chains_model)
