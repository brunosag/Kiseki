module Models

using Lux: Chain, Conv, MaxPool, FlattenLayer, Dense, relu, ToSimpleChainsAdaptor
import SimpleChains

export create_mnist_model

function create_mnist_model()
    return ToSimpleChainsAdaptor((28, 28, 1))(
        Chain(
            Conv((3, 3), 1 => 8, relu),
            MaxPool((2, 2)),
            Conv((3, 3), 8 => 16, relu),
            MaxPool((2, 2)),
            FlattenLayer(3),
            Dense(5 * 5 * 16 => 120, relu),
            Dense(120 => 10)
        )
    )
end

end
