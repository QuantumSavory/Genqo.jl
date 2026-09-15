module zalm

using DocStringExtensions
using LinearAlgebra: tr
using Gabs: QuadBlockBasis, eprstate, apply!, beamsplitter

import ..Genqo
using ..Genqo: modeswap, project, projector, clicks, duankimble


"""
$(TYPEDEF)

Parameters for a Zero-Added-Loss Multiplexing (ZALM) cascaded entanglement source.

The ZALM architecture uses two SPDC sources, interfering half of the modes from each source on a pair of 50/50 beamsplitters to perform a Bell-state measurement (BSM). A heralding click pattern signifies a probabilistic photon-photon Bell state between the output modes. Dark counts and all three efficiency channels can be modeled.

# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct ZALM
    "Mean photon number per mode per SPDC source (default `1e-2`)"
    mean_photon::Real = 1e-2
    #schmidt_coeffs::Vector{Float64}
    "Signal detector efficiency, ∈ [0, 1] (default `1.0`)"
    detection_efficiency::Real = 1.0
    "BSM detector efficiency, ∈ [0, 1] (default `1.0`)"
    bsm_efficiency::Real = 1.0
    "Photon outcoupling / transmission efficiency, ∈ [0, 1] (default `1.0`)"
    outcoupling_efficiency::Real = 1.0
    "Dark-count probability per BSM detector gate, ≥ 0 (default `0.0`)"
    dark_counts::Real = 0.0
    #visibility::Real = 1.0
end

const mds = 8 # Number of modes for our system

# The Gaussian circuit: two SPDC sources (EPR pairs plus polarization swaps) whose inner modes
# are interfered on the BSM beamsplitters (3,5) and (4,6). Signal modes are 1, 2, 7, 8. The
# squeezing phase θ = π reproduces the legacy sign convention for the qq correlation; Gabs works
# in ħ=2 and the legacy covariance matrices in ħ=1, hence the rescaling by `st.ħ` below.
function _state(μ::Real)
    st = eprstate(QuadBlockBasis(mds), asinh(√μ), Float64(π))
    apply!(st, [2, 4, 5, 7], modeswap(QuadBlockBasis(4)))
    apply!(st, [3, 5, 4, 6], beamsplitter(QuadBlockBasis(4), 0.5))
    st
end

# Per-mode efficiencies: the signal modes see transmission and signal detection, the BSM modes
# see the BSM detector efficiency.
_η(ηᵗ::Real, ηᵈ::Real, ηᵇ::Real) = Float64[ηᵗ*ηᵈ, ηᵗ*ηᵈ, ηᵇ, ηᵇ, ηᵇ, ηᵇ, ηᵗ*ηᵈ, ηᵗ*ηᵈ]

# The accepted herald: a coincidence across BSM modes 3 and 4 with nothing in 5 and 6. The signal
# modes are traced out, leaving the photon-photon state free.
const _HERALD = projector([:, :, 1, 1, 0, 0, :, :])

# The dual-rail Bell state (|1001⟩ + |0110⟩)/√2 heralded across the signal modes 1, 2, 7, 8.
const ψ⁺ = (clicks([1, 0, 0, 1]) + clicks([0, 1, 1, 0])) / √2

"""
$(TYPEDSIGNATURES)

Construct the pre-heralding covariance matrix for a ZALM source.

# Parameters
- μ: Mean photon number per mode

# Returns
16×16 `Float64` covariance matrix in qqpp ordering and the ħ=1 convention, after the BSM beamsplitter transforms.
"""
function covariance_matrix(μ::Real)::Matrix{Float64}
    st = _state(μ)
    st.covar ./ st.ħ
end
covariance_matrix(zalm::ZALM) = covariance_matrix(zalm.mean_photon)

"""
$(TYPEDSIGNATURES)

Calculate the density operator of the single-mode ZALM source on the spin-spin state.

Evaluated as [`Genqo.duankimble`](@ref) loading of the heralded photon-photon state into two spin
memories, pairing the signal modes (1,2) and (7,8). The heralding pattern measured on the BSM
modes is taken from `nvec[3:6]`, each of which must be 0 or 1.

# Parameters
- μ : Mean photon number
- ηᵗ : Outcoupling efficiency
- ηᵈ : Detection efficiency
- ηᵇ : Bell state measurement efficiency
- nvec : The vector of nᵢ's for the system, where nᵢ is the number of photons in mode i

# Returns
4×4 `ComplexF64` spin-spin density matrix
"""
function spin_density_matrix(μ::Real, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real, nvec::Vector{Int})::Matrix{ComplexF64}
    length(nvec) == mds || throw(ArgumentError("ZALM takes one photon-number outcome per mode, expected $mds but got $(length(nvec))"))
    Π = projector([-1, -1, nvec[3], nvec[4], nvec[5], nvec[6], -1, -1])
    ps = project(_state(μ), Π; η = _η(ηᵗ, ηᵈ, ηᵇ))
    # `duankimble` normalizes by the 4 accepted BSM click patterns; the legacy convention reports
    # the density matrix for the single pattern given by `nvec`, hence the factor of 4.
    Matrix(duankimble(ps, nvec[[1, 2, 7, 8]]).data) .* 4
end
spin_density_matrix(zalm::ZALM, nvec::Vector{Int}) = spin_density_matrix(zalm.mean_photon, zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency, nvec)

"""
$(TYPEDSIGNATURES)

Calculate the probability of photon-photon state generation with the given parameters.

A herald is accepted when exactly BSM modes 3 and 4 register a click. With dark counts present
that can happen four ways, so the probability is a weighted sum over the heralding patterns that
a real photon can be missing from: both clicks genuine, one genuine and one dark, or both dark.

# Parameters
- μ : Mean photon number
- ηᵗ : Outcoupling efficiency
- ηᵈ : Detection efficiency
- ηᵇ : Bell state measurement efficiency
- dark_counts : Probability of click with no photon present

# Returns
Probability of successful photon-photon state generation
"""
function probability_success(μ::Real, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real, dark_counts::Real)::Real
    st = _state(μ)
    η = _η(ηᵗ, ηᵈ, ηᵇ)
    p(pattern) = tr(project(st, projector(pattern); η = η))

    both = p([:, :, 1, 1, 0, 0, :, :]) # both heralding clicks are real photons
    only3 = p([:, :, 1, 0, 0, 0, :, :]) # mode 4's click is a dark count
    only4 = p([:, :, 0, 1, 0, 0, :, :]) # mode 3's click is a dark count
    neither = p([:, :, 0, 0, 0, 0, :, :]) # both clicks are dark counts

    dc = dark_counts
    return (1-dc)^4 * both +
           dc * (1-dc)^3 * (only3 + only4) +
           dc^2 * (1-dc)^2 * neither
end
probability_success(zalm::ZALM) = probability_success(zalm.mean_photon, zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency, zalm.dark_counts)

"""
$(TYPEDSIGNATURES)

Calculate the Bell-state fidelity of the single-mode ZALM source under loss.

This computes the overlap ⟨Φ|ρ|Φ⟩ of the post-heralding photon-photon state produced by the ZALM source with an ideal Bell state, normalized by the probability of accepting the herald. Efficiencies are applied through transmission/outcoupling (ηᵗ), detection (ηᵈ), and Bell-state-measurement (ηᵇ) models. Dark counts are not modeled here.

# Parameters
- μ  : Mean photon number
- ηᵗ : Outcoupling / transmission efficiency
- ηᵈ : Detection efficiency
- ηᵇ : Bell-state measurement efficiency

# Returns
Real-valued Bell-state fidelity of the ZALM source for the given parameters.
"""
function fidelity(μ::Real, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real)::Real
    ps = project(_state(μ), _HERALD; η = _η(ηᵗ, ηᵈ, ηᵇ))
    value = Genqo.fidelity(ψ⁺, ps)
    if abs(imag(value)) > 1e-10
        @warn "fidelity has nontrivial imaginary part" imag=imag(value) value=value
    end
    return real(value)
end
fidelity(zalm::ZALM) = fidelity(zalm.mean_photon, zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency)

end # module
