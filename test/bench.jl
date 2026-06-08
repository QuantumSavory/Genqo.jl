using Genqo
using BenchmarkTools

# Get optional function filter and output directory from command line arguments
func_filter = length(ARGS) > 0 ? ARGS[1] : ""
bench_dir   = length(ARGS) > 1 ? ARGS[2] : ".benchmarks"

uniform(min_val, max_val) = min_val + (max_val-min_val)*rand(Float64)
log_uniform(min_exp, max_exp) = 10^uniform(min_exp, max_exp)

const SUITE = BenchmarkGroup()


# TMSV benchmarks
rand_tmsv() = tmsv.TMSV(
    log_uniform(-5, 1),
    uniform(0.5, 1.0),
)
SUITE["tmsv.covariance_matrix"]      = @benchmarkable tmsv.covariance_matrix(t)           setup=(t=rand_tmsv())
SUITE["tmsv.loss_matrix_pgen"]       = @benchmarkable tmsv.loss_matrix_pgen(t)            setup=(t=rand_tmsv())
SUITE["tmsv.probability_success"]    = @benchmarkable tmsv.probability_success(t)         setup=(t=rand_tmsv())


# SPDC benchmarks
rand_spdc() = spdc.SPDC(
    log_uniform(-5, 1),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
)
nvec = [0,1,0,1]

SUITE["spdc.covariance_matrix"]      = @benchmarkable spdc.covariance_matrix(s)           setup=(s=rand_spdc())
SUITE["spdc.loss_bsm_matrix_fid"]    = @benchmarkable spdc.loss_bsm_matrix_fid(s)         setup=(s=rand_spdc())
SUITE["spdc.spin_density_matrix"]    = @benchmarkable spdc.spin_density_matrix(s, $nvec)  setup=(s=rand_spdc())
SUITE["spdc.fidelity"]               = @benchmarkable spdc.fidelity(s)                    setup=(s=rand_spdc())

# ZALM benchmarks
rand_zalm() = zalm.ZALM(
    log_uniform(-5, 1),
    #[one(Float64)],
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    zero(Float64),
    #one(Float64)
)
nvec = [1,0,1,1,0,0,1,0]

SUITE["zalm.covariance_matrix"]      = @benchmarkable zalm.covariance_matrix(z)           setup=(z=rand_zalm())
SUITE["zalm.loss_bsm_matrix_fid"]    = @benchmarkable zalm.loss_bsm_matrix_fid(z)         setup=(z=rand_zalm())
SUITE["zalm.spin_density_matrix"]    = @benchmarkable zalm.spin_density_matrix(z, $nvec)  setup=(z=rand_zalm())
SUITE["zalm.probability_success"]    = @benchmarkable zalm.probability_success(z)         setup=(z=rand_zalm())
SUITE["zalm.fidelity"]               = @benchmarkable zalm.fidelity(z)                    setup=(z=rand_zalm())


# SIGSAG benchmarks

rand_sigsag() = sigsag.SIGSAG(
    log_uniform(-5, 1),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
)

SUITE["sigsag.covariance_matrix"]      = @benchmarkable sigsag.covariance_matrix(s)           setup=(s=rand_sigsag())
SUITE["sigsag.loss_bsm_matrix_fid"]    = @benchmarkable sigsag.loss_bsm_matrix_fid(s)         setup=(s=rand_sigsag())
SUITE["sigsag.probability_success"]    = @benchmarkable sigsag.probability_success(s)         setup=(s=rand_sigsag())
SUITE["sigsag.fidelity"]               = @benchmarkable sigsag.fidelity(s)                    setup=(s=rand_sigsag())


# Perfect Matching Permutation (PMP) algorithm benchmarks (old vs new)
# The old algorithm (ported from genqo_old_pip.py:33-41) generates every n/2-sized
# combination of index pairs and rejects those that don't cover all indices exactly
# once. The new algorithm (tools._wick_partitions) builds pairings recursively, only
# emitting valid perfect matchings.
function _wick_partitions_old(n::Int)
    @assert iseven(n) "n must be even"
    all_partitions = Vector{Vector{Tuple{Int,Int}}}()

    # Mimic itertools.combinations(range(n), 2)
    pairs = Tuple{Int,Int}[]
    for i in 1:n-1, j in i+1:n
        push!(pairs, (i, j))
    end
    npairs = length(pairs)
    target = n ÷ 2

    # Mimic itertools.combinations(pairs, n//2), filtering for full coverage
    function inner(start::Int, current::Vector{Tuple{Int,Int}})
        if length(current) == target
            flat = Set{Int}()
            for (i, j) in current
                push!(flat, i); push!(flat, j)
            end
            if length(flat) == n
                push!(all_partitions, copy(current))
            end
            return
        end
        # Need at least (target - length(current)) more pairs
        for k in start:(npairs - (target - length(current)) + 1)
            push!(current, pairs[k])
            inner(k + 1, current)
            pop!(current)
        end
    end
    inner(1, Tuple{Int,Int}[])
    return all_partitions
end

for N in (2, 4, 6, 8)
    SUITE["pmp.new.N=$N"] = @benchmarkable tools._wick_partitions($N)
    SUITE["pmp.old.N=$N"] = @benchmarkable _wick_partitions_old($N)
end


# Other benchmarks
SUITE["tools.k_function_matrix"]     = @benchmarkable tools.k_function_matrix(cov) setup=(cov=zalm.covariance_matrix(rand_zalm()))
SUITE["linsweep_1d"]                 = @benchmarkable tmsv.probability_success.(range(1e-4, stop=1e-2, length=100), 0.7)
SUITE["linsweep_2d"]                 = @benchmarkable tmsv.probability_success.(range(1e-4, stop=1e-2, length=100), [0.2, 0.5, 0.6, 0.7, 0.9]')
SUITE["logsweep_1d"]                 = @benchmarkable tmsv.probability_success.(logrange(1e-4, 1e-2, length=100), 0.7)


# If running this file directly, run benchmarks
if abspath(PROGRAM_FILE) == @__FILE__
    # Filter suite based on func_filter if provided
    if !isempty(func_filter)
        filtered_suite = BenchmarkGroup()
        for (name, benchmark) in SUITE
            if occursin(func_filter, name)
                filtered_suite[name] = benchmark
            end
        end
        SUITE = filtered_suite
        if isempty(SUITE)
            @warn "No benchmarks matched filter: $func_filter"
        end
    end

    results = run(SUITE)
    for (func, trial) in results
        println("$func:")
        display(trial)
        println()
    end
    BenchmarkTools.save(joinpath(bench_dir, "jl-bench.json"), results)
end
