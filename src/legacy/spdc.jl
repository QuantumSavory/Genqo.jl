module spdc

using DocStringExtensions
using LinearAlgebra: dot
using Gabs: QuadBlockBasis, eprstate, apply!

using ..tools: _unreorder
using ..Genqo: modeswap, project, projector, clicks, duankimble


"""
$(TYPEDEF)

Parameters for a Spontaneous Parametric Down-Conversion (SPDC) entanglement source.

An SPDC source is composed of two Two-Mode Squeezed Vacuum (TMSV) states whose idler modes are swapped. This is not to be confused with the fact that there are two Spontaneous Parametric Down-Conversion (SPDC) processes occuring. This is a standard unheralded source of dual-rail entangled photon pairs.


# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct SPDC
    "Mean photon number per mode (default `1e-2`)"
    mean_photon::Real = 1e-2
    "Detector efficiency, ∈ [0, 1] (default `1.0`)"
    detection_efficiency::Real = 1.0
    "Bell-state measurement efficiency, ∈ [0, 1] (default `1.0`)"
    bsm_efficiency::Real = 1.0
    "Photon outcoupling / transmission efficiency, ∈ [0, 1] (default `1.0`)"
    outcoupling_efficiency::Real = 1.0
end

const mds = 4 # Number of modes for our system

# The Gaussian circuit: two EPR pairs on modes (1,2) and (3,4), then a swap of modes 2 and 4
# to give the SPDC polarization pairing (1,4)(2,3). The squeezing phase θ = π reproduces the
# legacy sign convention for the qq correlation; Gabs works in ħ=2 and the legacy covariance
# matrices in ħ=1, hence the rescaling by `st.ħ` wherever the matrix is reported.
function _state(μ::Real)
    st = eprstate(QuadBlockBasis(mds), asinh(√μ), Float64(π))
    apply!(st, [2, 4], modeswap(QuadBlockBasis(2)))
    st
end

# The source is unheralded, so every mode is left free and the photon-photon state is read out
# directly. A single transmission-detection efficiency η = ηᵗηᵈ applies to all four modes.
_projected_state(μ::Real, ηᵗ::Real, ηᵈ::Real) =
    project(_state(μ), projector([:, :, :, :]); η = fill(Float64(ηᵗ * ηᵈ), mds))

# The dual-rail Bell state (|1001⟩ + |0110⟩)/√2 that the source ideally emits.
const ψ⁺ = (clicks([1, 0, 0, 1]) + clicks([0, 1, 1, 0])) / √2

"""
$(TYPEDSIGNATURES)

Construct the covariance matrix for an SPDC source.

# Parameters
- μ: Mean photon number per mode

# Returns
8×8 `Float64` covariance matrix in qpqp ordering and the ħ=1 convention.
"""
function covariance_matrix(μ::Real)::Matrix{Float64}
    st = _state(μ)
    _unreorder(st.covar ./ st.ħ)
end
covariance_matrix(spdc::SPDC) = covariance_matrix(spdc.mean_photon)

"""
$(TYPEDSIGNATURES)

Calculate the spin-spin density matrix for the SPDC source conditioned on photon-number measurement outcome `nvec` after simulated mode-memory interaction.

Evaluated as [`Genqo.duankimble`](@ref) loading of the raw source into two spin memories, pairing
the dual-rail modes (1,2) and (3,4).

# Parameters
- μ    : Mean photon number per mode
- ηᵗ   : Outcoupling / transmission efficiency
- ηᵈ   : Detection efficiency
- nvec : Photon-number vector `[n₁, n₂, n₃, n₄]` for the four modes

# Returns
4×4 `ComplexF64` spin-spin density matrix.
"""
function spin_density_matrix(μ::Real, ηᵗ::Real, ηᵈ::Real, nvec::Vector{Int})::Matrix{ComplexF64}
    length(nvec) == mds || throw(ArgumentError("SPDC takes one photon-number outcome per mode, expected $mds but got $(length(nvec))"))
    Matrix(duankimble(_projected_state(μ, ηᵗ, ηᵈ), nvec).data)
end
spin_density_matrix(spdc::SPDC, nvec::Vector{Int}) = spin_density_matrix(spdc.mean_photon, spdc.outcoupling_efficiency, spdc.detection_efficiency, nvec)

"""
$(TYPEDSIGNATURES)

Calculate the Bell-state fidelity of the single-mode SPDC source under loss.

This computes the overlap ⟨Φ|ρ|Φ⟩ of the photon-photon state produced by the SPDC source with an ideal Bell state. The source is unheralded, so ρ is *not* renormalized by a success probability and the result falls off with μ — divide by the coincidence probability to condition on a detected pair.

# Parameters
- μ  : Mean photon number
- ηᵗ : Outcoupling / transmission efficiency
- ηᵈ : Detection efficiency

# Returns
Real-valued Bell-state overlap of the SPDC source for the given parameters.
"""
function fidelity(μ::Real, ηᵗ::Real, ηᵈ::Real)::Real
    value = dot(ψ⁺', _projected_state(μ, ηᵗ, ηᵈ), ψ⁺)
    if abs(imag(value)) > 1e-10
        @warn "fidelity has nontrivial imaginary part" imag=imag(value) value=value
    end
    return real(value)
end
fidelity(spdc::SPDC) = fidelity(spdc.mean_photon, spdc.outcoupling_efficiency, spdc.detection_efficiency)

end # module
