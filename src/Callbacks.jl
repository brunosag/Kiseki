module Callbacks

using Printf
using ComponentArrays
using Statistics
using ..Evaluation: accuracy
using ..Checkpoints: save_checkpoint

export CheckpointCallback

mutable struct CheckpointCallback{M, S, DL, T} <: Function
    i₀::Int
    block::Int
    checkpoint_Δi::Int
    i_timer::Float64
    complete_trace::Vector{T}
    model::M
    st::S
    axes::Any
    test_dataloader::DL
    best_test_acc::Float64
    prev_checkpoint::Ref{String}
    save_dir::String
    current_global_i::Int
end

function (cb::CheckpointCallback)(trace_record)
    trace_record.iteration == 0 && return false

    global_i = cb.i₀ + (cb.block - 1) * cb.checkpoint_Δi + trace_record.iteration
    cb.current_global_i = global_i

    current_time = time()
    elapsed = current_time - cb.i_timer
    cb.i_timer = current_time

    meta = trace_record.metadata

    if meta.interval_changed
        @printf("[Scheduler] Batch interval updated to: %d\n", meta.batch_interval)
    end

    if meta.batch_changed
        @printf("[DataLoader] Sampled new data batch.\n")
    end

    σ_str = join([@sprintf("%.5f", s) for s in meta.σ], ", ")

    @printf("i = %d \t L(𝛉) = %.6f \t Δt = %.2fs \t 𝛔 = [%s]\n", global_i, meta.L, elapsed, σ_str)

    push!(
        cb.complete_trace, (
            i = global_i,
            L = Float32(meta.L),
            σ_var = Float32(var(meta.σ)),
            σ_mean = Float32(mean(meta.σ)),
            batch_interval = meta.batch_interval,
            σ = copy(meta.σ),
        )
    )

    if trace_record.iteration == cb.checkpoint_Δi
        θ_current = ComponentArray(meta.θ_ema, cb.axes)
        recent_traces = @view cb.complete_trace[max(1, end - cb.checkpoint_Δi + 1):end]
        rolling_loss = sum(t.L for t in recent_traces) / length(recent_traces)

        test_acc = accuracy(cb.model, θ_current, cb.st, cb.test_dataloader)

        if test_acc > cb.best_test_acc
            cb.best_test_acc = test_acc

            checkpoint_data = Dict(
                "i" => global_i,
                "L" => meta.L,
                "θ" => meta.θ,
                "σ" => meta.σ,
                "θ_ema" => meta.θ_ema,
                "rolling_loss" => rolling_loss,
                "test_acc" => test_acc,
                "complete_trace" => cb.complete_trace
            )

            save_checkpoint(checkpoint_data, test_acc, global_i, cb.prev_checkpoint, cb.save_dir)

            @printf("\n[Checkpoint Saved] Iter: %d \t Current Loss: %.4f \t Rolling Loss: %.4f \t Test Acc: %.2f%%\n\n", global_i, meta.L, rolling_loss, test_acc)
        else
            @printf("\n[Skipped Checkpoint] Iter: %d \t Current Loss: %.4f \t Rolling Loss: %.4f \t Test Acc: %.2f%% (Best: %.2f%%)\n\n", global_i, meta.L, rolling_loss, test_acc, cb.best_test_acc)
        end
    end
    return false
end

end
