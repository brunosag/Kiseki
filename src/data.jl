struct BalancedDataLoader{T <: AbstractArray, V <: AbstractArray, R <: AbstractRNG}
    X::T
    Y::V
    batchsize::Int
    class_groups::Vector{Vector{Int}}
    samples_per_class::Int
    n_batches::Int
    rng::R

    function BalancedDataLoader(X::T, Y::V, y_cpu::AbstractVector, batchsize::Int, rng::R) where {T <: AbstractArray, V <: AbstractArray, R <: AbstractRNG}
        classes = sort(unique(y_cpu))
        n_classes = length(classes)

        if batchsize % n_classes != 0
            throw(ArgumentError("Batch size must be a multiple of $n_classes"))
        end

        samples_per_class = batchsize ÷ n_classes
        class_groups = [findall(==(c), y_cpu) for c in classes]
        n_batches = minimum(length.(class_groups)) ÷ samples_per_class

        return new{T, V, R}(X, Y, batchsize, class_groups, samples_per_class, n_batches, rng)
    end
end

Base.length(d::BalancedDataLoader) = d.n_batches

function Base.iterate(d::BalancedDataLoader)
    shuffled_groups = [shuffle(d.rng, group) for group in d.class_groups]
    return iterate(d, (shuffled_groups, 1))
end

function Base.iterate(d::BalancedDataLoader, state)
    shuffled_groups, b = state
    if b > d.n_batches
        return nothing
    end

    batch_idx = Int[]
    sizehint!(batch_idx, d.batchsize)

    for group in shuffled_groups
        start_idx = (b - 1) * d.samples_per_class + 1
        end_idx = b * d.samples_per_class
        append!(batch_idx, @view group[start_idx:end_idx])
    end

    shuffle!(d.rng, batch_idx)

    Xᵢ = d.X[:, :, :, batch_idx]
    Yᵢ = d.Y[:, batch_idx]

    return (Xᵢ, Yᵢ), (shuffled_groups, b + 1)
end

function load_MNIST(rng, batchsize, dev; balanced = true, val_size = 0)
    train_data = MNIST(:train)
    test_data = MNIST(:test)

    X_train_cpu = reshape(train_data.features, 28, 28, 1, :)
    y_train_cpu = train_data.targets
    Y_train_cpu = onehotbatch(y_train_cpu, 0:9)

    X_test_cpu = reshape(test_data.features, 28, 28, 1, :)
    Y_test_cpu = onehotbatch(test_data.targets, 0:9)

    if val_size > 0
        classes = sort(unique(y_train_cpu))
        n_classes = length(classes)

        if val_size % n_classes != 0
            throw(ArgumentError("Validation size must be a multiple of $n_classes for exact balancing."))
        end

        val_per_class = val_size ÷ n_classes

        train_idx = Int[]
        val_idx = Int[]

        for c in classes
            c_idx = findall(==(c), y_train_cpu)
            shuffle!(rng, c_idx)
            append!(val_idx, @view c_idx[1:val_per_class])
            append!(train_idx, @view c_idx[(val_per_class + 1):end])
        end

        shuffle!(rng, train_idx)
        shuffle!(rng, val_idx)

        X_val = dev(X_train_cpu[:, :, :, val_idx])
        Y_val_cold = y_train_cpu[val_idx]

        X_train = dev(X_train_cpu[:, :, :, train_idx])
        Y_train = dev(Y_train_cpu[:, train_idx])
        y_train = y_train_cpu[train_idx]
    else
        X_train = dev(X_train_cpu)
        Y_train = dev(Y_train_cpu)
        y_train = y_train_cpu
    end

    train_loader = balanced ?
        BalancedDataLoader(X_train, Y_train, y_train, batchsize, rng) :
        DataLoader((X_train, Y_train); batchsize, rng, shuffle = true)

    train_loader = Iterators.Stateful(Iterators.cycle(train_loader))
    test_loader = DataLoader((dev(X_test_cpu), dev(Y_test_cpu)); batchsize)

    if val_size > 0
        return train_loader, (X_val, Y_val_cold), test_loader
    else
        return train_loader, test_loader
    end
end
