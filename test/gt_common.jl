# Shared between generate_ground_truth.jl and the test suite: the fixed RNG seed,
# the parameter sets drawn from it, and the legacy-covariance → GaussianState bridge.
#
# The test suite re-draws the parameters with the same seed and checks them against
# the values stored in the JLD2 file, so any drift between this file and the committed
# ground truth is caught before the physics comparisons run.

using StableRNGs
using Gabs
using Genqo

const GT_SEED = 0x67656e716f # "genqo"
const GT_FILE = joinpath(@__DIR__, "data", "ground_truth.jld2")
const GT_SOURCES = ("tmsv", "spdc", "zalm", "sigsag")
const GT_NCASES = 5 # one lossless set + 4 random sets per source

# Photon-number outcomes for the spin-density-matrix / Duan-Kimble comparisons.
# Entries stay in {0, 1}: extract_W_terms is multilinear-only.
const GT_NVEC_SPDC = [1, 0, 1, 0]
const GT_NVEC_ZALM = [1, 0, 1, 1, 0, 0, 1, 0] # modes 3-6 are the BSM pattern; 1, 2, 7, 8 are the loaded modes

_gt_loguniform(rng, lo, hi) = exp10(log10(lo) + (log10(hi) - log10(lo)) * rand(rng))
_gt_eta(rng) = 0.5 + 0.5 * rand(rng)

"""
    gt_params()::Dict{String, Matrix{Float64}}

Parameter sets for each source, keyed by source name. Each matrix is `GT_NCASES × 4`
with columns `(μ, ηᵗ, ηᵈ, ηᵇ)`; row 1 is the lossless case. Sources that take fewer
efficiencies simply ignore the extra columns. Deterministic: always drawn from
`StableRNG(GT_SEED)` in a fixed order.
"""
function gt_params()
    rng = StableRNG(GT_SEED)
    params = Dict{String,Matrix{Float64}}()
    for src in GT_SOURCES
        m = Matrix{Float64}(undef, GT_NCASES, 4)
        m[1, :] .= (1e-2, 1.0, 1.0, 1.0)
        for i in 2:GT_NCASES
            m[i, :] .= (_gt_loguniform(rng, 1e-4, 1e-1), _gt_eta(rng), _gt_eta(rng), _gt_eta(rng))
        end
        params[src] = m
    end
    params
end

"""
    legacy_state(cov_qqpp::Matrix{Float64})

Wrap a legacy (v1) covariance matrix — ħ=1 convention (vacuum variance 0.5), qqpp
ordering — as a Gabs `GaussianState` suitable for `project`, which rescales by `st.ħ`
so the projection machinery sees exactly this matrix.
"""
legacy_state(cov_qqpp::Matrix{Float64}) =
    GaussianState(QuadBlockBasis(size(cov_qqpp, 1) ÷ 2), zeros(size(cov_qqpp, 1)), cov_qqpp; ħ = 1)
