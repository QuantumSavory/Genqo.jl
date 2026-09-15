module sigsag

using DocStringExtensions
using LinearAlgebra: tr
using Gabs: QuadBlockBasis, eprstate, vacuumstate, apply!, beamsplitter, ⊗

import ..Genqo
using ..Genqo: modeswap, project, projector, clicks


"""
$(TYPEDEF)

Parameters for the heralded entanglement source architecture proposed by Chahine et al.

The SIGSAG source is a heralded dual-rail entanglement source architecture proposed by Yousef Chahine et al. as an alternative to the cascaded source architecture. It can be realized with a single Sagnac configured entanglement source, hence the nomenclature of "SIGSAG" for short.

# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct SIGSAG
    "Mean photon number per mode (default `1e-2`)"
    mean_photon::Real = 1e-2
    "Signal detector efficiency, ∈ [0, 1] (default `1.0`)"
    detection_efficiency::Real = 1.0
    "BSM detector efficiency, ∈ [0, 1] (default `1.0`)"
    bsm_efficiency::Real = 1.0
    "Photon outcoupling / transmission efficiency, ∈ [0, 1] (default `1.0`)"
    outcoupling_efficiency::Real = 1.0
end

const mds = 6 # Number of modes for our system

# The Gaussian circuit: one SPDC source (an EPR pair plus a polarization swap) padded with two
# vacuum modes, interfered on the beamsplitters (3,5) and (4,6). Modes 1 and 2 carry the herald
# and modes 3-6 the photon-photon state. The squeezing phase θ = π reproduces the legacy sign
# convention for the qq correlation; Gabs works in ħ=2 and the legacy covariance matrices in
# ħ=1, hence the rescaling by `st.ħ` below.
function _state(μ::Real)
    st4 = eprstate(QuadBlockBasis(4), asinh(√μ), Float64(π))
    apply!(st4, [2, 4], modeswap(QuadBlockBasis(2)))
    st = st4 ⊗ vacuumstate(QuadBlockBasis(2))
    apply!(st, [3, 5, 4, 6], beamsplitter(QuadBlockBasis(4), 0.5))
    st
end

# Heralding clicks land on modes 1 and 2 with the signal detector efficiency; the photon-photon
# state lives in the traced-out measured modes 3-6, which see the outcoupling efficiency.
_projected_state(μ::Real, ηᵗ::Real, ηᵈ::Real) =
    project(_state(μ), projector([1, 1, :, :, :, :]); η = Float64[ηᵈ, ηᵈ, ηᵗ, ηᵗ, ηᵗ, ηᵗ])

# The dual-rail Bell state (|1001⟩ + |0110⟩)/√2 heralded across the measured modes 3-6.
const ψ⁺ = (clicks([1, 0, 0, 1]) + clicks([0, 1, 1, 0])) / √2

"""
$(TYPEDSIGNATURES)

Construct the covariance matrix for a SIGSAG source.

# Parameters
- μ: Mean photon number per mode

# Returns
12×12 `Float64` covariance matrix in qqpp ordering and the ħ=1 convention, after the beamsplitter transforms.
"""
function covariance_matrix(μ::Real)::Matrix{Float64}
    st = _state(μ)
    st.covar ./ st.ħ
end
covariance_matrix(sigsag::SIGSAG) = covariance_matrix(sigsag.mean_photon)

"""
$(TYPEDSIGNATURES)

Calculate the probability of photon-photon state generation for the SIGSAG source.

Evaluated as the trace of the [`Genqo.project`](@ref)ion of the source state onto the heralding
coincidence across modes 1 and 2.

# Parameters
- μ  : Mean photon number per mode
- ηᵗ : Outcoupling / transmission efficiency
- ηᵈ : Detection efficiency

# Returns
Real-valued probability of successful photon-photon state generation.
"""
probability_success(μ::Real, ηᵗ::Real, ηᵈ::Real)::Real = tr(_projected_state(μ, ηᵗ, ηᵈ))
probability_success(sigsag::SIGSAG) = probability_success(sigsag.mean_photon, sigsag.outcoupling_efficiency, sigsag.detection_efficiency)

"""
$(TYPEDSIGNATURES)

Calculate the Bell-state fidelity of the SIGSAG source under loss.

Computes the Bell-state overlap ⟨Φ|ρ|Φ⟩, where ρ is the photon-photon density matrix following heralding, normalized by the probability of the herald.

# Parameters
- μ  : Mean photon number per mode
- ηᵗ : Outcoupling / transmission efficiency
- ηᵈ : Detection efficiency

# Returns
Real-valued Bell-state fidelity of the SIGSAG source for the given parameters.
"""
function fidelity(μ::Real, ηᵗ::Real, ηᵈ::Real)::Real
    value = Genqo.fidelity(ψ⁺, _projected_state(μ, ηᵗ, ηᵈ))
    if abs(imag(value)) > 1e-10
        @warn "fidelity has nontrivial imaginary part" imag=imag(value) value=value
    end
    return real(value)
end
fidelity(sigsag::SIGSAG) = fidelity(sigsag.mean_photon, sigsag.outcoupling_efficiency, sigsag.detection_efficiency)

end # module
