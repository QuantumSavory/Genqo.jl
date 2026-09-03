# Regression benchmark suite for the essentials: Wick contraction kernels and the
# probability / fidelity / density-matrix calculations built on them, in both the v2
# generalized framework and the v1 legacy sources.
#
# Used by AirspeedVelocity (`just asv <rev>`), which picks up `SUITE`, and runnable
# directly with an optional substring filter:
#
#     julia --project=benchmark benchmark/benchmarks.jl [filter]
#
# Parameters are fixed (no randomness) so timings are comparable across revisions.

using Genqo
using Genqo: _wick_partitions
using Gabs
using BenchmarkTools

const SUITE = BenchmarkGroup()

const μ = 1e-2
const ηᵗ, ηᵈ, ηᵇ = 0.9, 0.8, 0.85


# Wick contraction kernels

zalm_Ainv() = inv(tools.k_function_matrix(zalm.covariance_matrix(μ)) + zalm.loss_bsm_matrix_fid(ηᵗ, ηᵈ, ηᵇ))
bell_sum = zalm.moment_vector.bell_aa + zalm.moment_vector.bell_ab +
           zalm.moment_vector.bell_ba + zalm.moment_vector.bell_bb

SUITE["wick.partitions_N8"] = @benchmarkable _wick_partitions(Int8(8))
SUITE["wick.extract_W_terms"] = @benchmarkable extract_W_terms($bell_sum)
SUITE["wick.W_zalm_bell"] = @benchmarkable W(terms, Ainv) setup = (terms = extract_W_terms($bell_sum); Ainv = zalm_Ainv())


# v2 generalized framework: ZALM source built from Gabs circuits, projected with a
# pre-warmed HybridProjectionEngine so steady-state (cached-polynomial) evaluation is measured

function zalm_state(μ)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), Float64(π))
    ms = modeswap(QuadBlockBasis(2))
    bs = beamsplitter(QuadBlockBasis(2), 0.5)
    apply!(st, [2, 4], ms)
    apply!(st, [5, 7], ms)
    apply!(st, [3, 5], bs)
    apply!(st, [4, 6], bs)
    st
end

const ZALM_Π = projector([-1, -1, 1, 1, 0, 0, -1, -1])
const ZALM_η = [ηᵗ * ηᵈ, ηᵗ * ηᵈ, ηᵇ, ηᵇ, ηᵇ, ηᵇ, ηᵗ * ηᵈ, ηᵗ * ηᵈ]
const ENGINE8 = HybridProjectionEngine(8)
const ψ⁺ = (clicks([1, 0, 0, 1]) + clicks([0, 1, 1, 0])) / √2
const ψ⁺ᵈ = ψ⁺'

const ENGINE12 = HybridProjectionEngine(12) # emissive loading needs 2 extra modes per memory
const DK_d = [1, 0, 1, 0]

let ps = project(zalm_state(μ), ZALM_Π; η = ZALM_η) # warm the C-polynomial caches
    tr(ps; engine = ENGINE8)
    dot(ψ⁺ᵈ, ps, ψ⁺; engine = ENGINE8)
    Matrix(duankimble(ps, DK_d; engine = ENGINE8).data)
    Matrix(emissiveload(ps; engine = ENGINE12).data)
end

SUITE["project.zalm_state"] = @benchmarkable zalm_state($μ)
SUITE["project.tr"] = @benchmarkable tr(ps; engine = ENGINE8) setup = (ps = project(zalm_state($μ), ZALM_Π; η = ZALM_η))
SUITE["project.dot_bell"] = @benchmarkable dot(ψ⁺ᵈ, ps, ψ⁺; engine = ENGINE8) setup = (ps = project(zalm_state($μ), ZALM_Π; η = ZALM_η))
SUITE["project.to_fock"] = @benchmarkable to_fock(ps; engine = ENGINE8, cutoff = 1) setup = (ps = project(zalm_state($μ), ZALM_Π; η = ZALM_η))
# Memory loading returns a lazily-evaluated density matrix, so the access pattern is what costs:
# `tr` needs the M diagonal entries, materializing needs all M². Both are benchmarked so a
# regression in either the deferred path or the bulk path shows up.
SUITE["project.duankimble.construct"] = @benchmarkable duankimble(ps, DK_d; engine = ENGINE8) setup = (ps = project(zalm_state($μ), ZALM_Π; η = ZALM_η))
SUITE["project.duankimble.tr"] = @benchmarkable tr(duankimble(ps, DK_d; engine = ENGINE8)) setup = (ps = project(zalm_state($μ), ZALM_Π; η = ZALM_η))
SUITE["project.duankimble.materialize"] = @benchmarkable Matrix(duankimble(ps, DK_d; engine = ENGINE8).data) setup = (ps = project(zalm_state($μ), ZALM_Π; η = ZALM_η))
SUITE["project.emissiveload.tr"] = @benchmarkable tr(emissiveload(ps; engine = ENGINE12)) setup = (ps = project(zalm_state($μ), ZALM_Π; η = ZALM_η))
SUITE["project.emissiveload.materialize"] = @benchmarkable Matrix(emissiveload(ps; engine = ENGINE12).data) setup = (ps = project(zalm_state($μ), ZALM_Π; η = ZALM_η))



# v1 legacy sources: one headline calculation per source

nvec_spdc = [0, 1, 0, 1]

SUITE["legacy.tmsv.probability_success"] = @benchmarkable tmsv.probability_success($μ, $ηᵈ)
SUITE["legacy.spdc.spin_density_matrix"] = @benchmarkable spdc.spin_density_matrix($μ, $ηᵗ, $ηᵈ, $nvec_spdc)
SUITE["legacy.spdc.fidelity"] = @benchmarkable spdc.fidelity($μ, $ηᵗ, $ηᵈ)
SUITE["legacy.zalm.probability_success"] = @benchmarkable zalm.probability_success($μ, $ηᵗ, $ηᵈ, $ηᵇ, 0.0)
SUITE["legacy.zalm.fidelity"] = @benchmarkable zalm.fidelity($μ, $ηᵗ, $ηᵈ, $ηᵇ)
SUITE["legacy.sigsag.fidelity"] = @benchmarkable sigsag.fidelity($μ, $ηᵗ, $ηᵈ)


# If running this file directly, run (optionally filtered) benchmarks and print results
if abspath(PROGRAM_FILE) == @__FILE__
    func_filter = isempty(ARGS) ? "" : ARGS[1]
    suite = SUITE
    if !isempty(func_filter)
        suite = BenchmarkGroup()
        for (name, benchmark) in SUITE
            occursin(func_filter, name) && (suite[name] = benchmark)
        end
        isempty(suite) && @warn "No benchmarks matched filter: $func_filter"
    end

    tune!(suite)
    results = run(suite)
    for name in sort(collect(keys(results)))
        println("$name:")
        display(results[name])
        println()
    end
end
