module Callbacks

using Printf
using ComponentArrays
using ..Evaluation: accuracy
using ..Checkpoints: save_checkpoint

export CheckpointCallback

mutable struct CheckpointCallback{M, S, DL, T} <: Function
    checkpoint_Δi::Int
    i_timer::Float64
    complete_trace::Vector{T}
    accuracy_trace::Vector{Tuple{Int, Float64}}
    model::M
    st::S
    axes::Any
    test_dataloader::DL
    best_test_acc::Float64
    prev_checkpoint::Ref{String}
    save_dir::String
end

function (cb::CheckpointCallback)(
        global_i::Int, θ::Vector{Float32}, θ_ema::Vector{Float32}, L::Float32, σ::Float32
    )
    current_time = time()
    elapsed = current_time - cb.i_timer
    cb.i_timer = current_time

    @printf("i = %d \t L(𝛉) = %.6f \t Δt = %.2fs \t σ = %.5f\n", global_i, L, elapsed, σ)

    push!(cb.complete_trace, (i = global_i, L = L, σ = σ))

    if global_i % cb.checkpoint_Δi == 0
        θ_current = ComponentArray(θ_ema, cb.axes)

        raw_acc = accuracy(cb.model, θ_current, cb.st, cb.test_dataloader)
        test_acc = raw_acc > 1.0 ? raw_acc / 100.0 : raw_acc

        push!(cb.accuracy_trace, (global_i, test_acc))

        if test_acc > cb.best_test_acc
            cb.best_test_acc = test_acc

            checkpoint_data = Dict(
                "i" => global_i,
                "L" => L,
                "θ" => θ,
                "σ" => σ,
                "θ_ema" => θ_ema,
                "test_acc" => test_acc,
                "complete_trace" => cb.complete_trace,
                "accuracy_trace" => cb.accuracy_trace
            )

            save_checkpoint(checkpoint_data, test_acc, global_i, cb.prev_checkpoint, cb.save_dir)
            @printf("\n[Checkpoint Saved] i = %d \t L(𝛉) = %.6f \t Accuracy = %.2f%%\n\n", global_i, L, test_acc * 100.0)
        else
            @printf("\n[Skipped Checkpoint] i = %d \t L(𝛉) = %.6f \t Accuracy = %.2f%% (Best: %.2f%%)\n\n", global_i, L, test_acc * 100.0, cb.best_test_acc * 100.0)
        end
    end
    return false
end

end
