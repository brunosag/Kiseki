module Kiseki

import Lux, Zygote, SimpleChains
import Base: run
using LuxCUDA
using Printf
using Random: AbstractRNG, Xoshiro, TaskLocalRNG, rand!, shuffle, shuffle!
using Optimisers: destructure, Descent, Restructure
using OneHotArrays: onehotbatch, onecold
using ADTypes: AutoZygote
using StatsBase: Weights, sample, sample!
using MLDatasets: MNIST
using MLUtils: DataLoader, getobs, batch
using Lux: Chain, Conv, MaxPool, FlattenLayer, Dense, relu
using WeightInitializers: kaiming_normal
using MLDataDevices: AbstractDevice, AbstractCPUDevice, AbstractGPUDevice
using Polyester: @batch

export Experiment, CNN_2C2D_MNIST, LEEA, SGD, run

include("data.jl")
include("models.jl")
include("optimizers.jl")
include("optimizers/leea.jl")
include("optimizers/sgd.jl")
include("experiment.jl")

end
