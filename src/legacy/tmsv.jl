module tmsv

using DocStringExtensions
using LinearAlgebra: tr
using Gabs: QuadBlockBasis, eprstate

using ..tools: _unreorder
using ..Genqo: project, projector


"""
$(TYPEDEF)

Parameters for a Two-Mode Squeezed Vacuum (TMSV) entanglement source.

The TMSV is the simplest Gaussian entangled state. Both modes are produced by a
single parametric interaction, and the entanglement is characterized by the mean photon number μ.

# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct TMSV
    "Mean photon number per mode (default `1e-2`)"
    mean_photon::Real = 1e-2
    "Detector efficiency, ∈ [0, 1] (default `1.0`)"
    detection_efficiency::Real = 1.0
end

const mds = 2 # Number of modes for our system

# The Gaussian circuit: a single EPR pair. The squeezing phase θ = π reproduces the legacy
# sign convention for the qq correlation (+√(μ(μ+1))); Gabs works in ħ=2, the legacy
# covariance matrices in ħ=1, hence the rescaling by `st.ħ` wherever the matrix is reported.
_state(μ::Real) = eprstate(QuadBlockBasis(mds), asinh(√μ), Float64(π))

# Both modes are detected, so a success is a coincidence click across the pair.
_projected_state(μ::Real, ηᵈ::Real) =
    project(_state(μ), projector([1, 1]); η = fill(Float64(ηᵈ), mds))

"""
$(TYPEDSIGNATURES)

Construct the covariance matrix for a TMSV state.

# Parameters
- μ : The mean photon number of the TMSV state

# Returns
4×4 `Float64` covariance matrix for the TMSV state, in the qpqp ordering and the ħ=1 convention.
"""
function covariance_matrix(μ::Real)::Matrix{Float64}
    st = _state(μ)
    _unreorder(st.covar ./ st.ħ)
end
covariance_matrix(tmsv::TMSV) = covariance_matrix(tmsv.mean_photon)

"""
$(TYPEDSIGNATURES)

Calculate the probability of photon-photon state generation with the given parameters.

Evaluated as the trace of the [`Genqo.project`](@ref)ion of the source state onto a coincidence
click in both modes.

# Parameters
- μ : The mean photon number of the TMSV state
- ηᵈ : Detection efficiency

# Returns
Probability of successful photon-photon state generation
"""
probability_success(μ::Real, ηᵈ::Real)::Real = tr(_projected_state(μ, ηᵈ))
probability_success(tmsv::TMSV) = probability_success(tmsv.mean_photon, tmsv.detection_efficiency)

end # module
