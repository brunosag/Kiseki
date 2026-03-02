using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Kiseki
using Random

function main()
    rng = Xoshiro(42)
    resume_target = isempty(ARGS) ? nothing : ARGS[1]

    checkpoint_dir = abspath(joinpath(@__DIR__, "..", "checkpoints"))
    mkpath(checkpoint_dir)

    model = create_mnist_model()

    return train_evolution(
        model;
        rng = rng,
        resume_file = resume_target,
        save_dir = checkpoint_dir,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
