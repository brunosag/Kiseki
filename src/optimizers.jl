abstract type AbstractOptimizer end
abstract type AbstractOptimizerState end

init_workspace(opt::AbstractOptimizer, ops) = nothing
update_scheduler!(opt::AbstractOptimizer, ops, acc, best_acc) = nothing
