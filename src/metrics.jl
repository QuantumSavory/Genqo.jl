"""
metrics.jl: This module contains functions for computing performance metrics such as probability of success and fidelity for quantum circuits.
"""
module metrics

using LinearAlgebra

using ..tools: W, extract_W_terms, k_function_matrix
using ..gates
using ..states
using ..registers
using ..detectors

export Metric, Probability, Fidelity, compute!


abstract type ComputeStep end

# The A matrix only needs to know which modes are being measured vs traced out, not the specific detection outcome.
# That information is removed to improve cache hits.
struct AMatrix <: ComputeStep
    detectors::Vector{Union{Detector, Nothing}}
end
function compute!(amat::AMatrix, register::QuantumRegister, cache::Dict{ComputeStep, Any} = [])::Matrix{ComplexF64}
    get!(cache, amat) do
        # Compute the A⁻¹ matrix from the Gaussian state covariance matrix
        A = k_function_matrix(register.state.covariance) + G_matrix(amat.detectors, register.mds)
        return inv(A)
    end
end

# The C polynomial depends on the specific detection outcome.
struct CPoly <: ComputeStep
    detection_outcome::MeasurementOutcome
end
function compute!(cpoly::CPoly, register::QuantumRegister, cache::Dict{ComputeStep, Any} = [])::Vector{Tuple{ComplexF64, Vector{Int}}}
    get!(cache, cpoly) do
        # Compute the C polynomial from the measurement spec
        C = one(register.R)
        for (measurement, α, β) in zip(cpoly.detection_outcome.measurements, register.α, register.β)
            if measurement isa PhotonNumMeasurement && measurement.n == 1
                C *= (α * β)
            # TODO: support higher photon number outcomes as well, which will involve including the appropriate Fock term (αβ*)ⁿ/n! in the C polynomial
            # tools.W() will need to be generalized
            end
        end

        return extract_W_terms(C)
    end
end

struct DensityOperatorElem <: ComputeStep
    d::Int
    g::Int
end
function compute!(elem::DensityOperatorElem, register::QuantumRegister, cache::Dict{ComputeStep, Any} = [])::ComplexF64
    get!(cache, elem) do
        # TODO
    end
end

abstract type Metric end

# Compute probability of a given detection outcome
struct Probability <: Metric
    detection_outcome::MeasurementOutcome
end
function compute!(probability::Probability, register::QuantumRegister, cache::Dict{ComputeStep, Any} = [])::Float64
    # Compute the probability of the given detection outcome by building the appropriate moment polynomial and performing the necessary Wick contractions
    Ainv = compute!(AMatrix(register.detectors), register, cache)
    C = compute!(CPoly(probability.detection_outcome), register, cache)
    Γ = register.state.covariance + 0.5*I
    detΓ = det(Γ)
    return (det(Ainv) / (detΓ^0.25 * conj(detΓ)^0.25)) * W(C, Ainv)
end

# Compute fidelity of post-selected state with respect to ideal target state
struct Fidelity <: Metric
    ideal_state::QuantumState
    detection_outcome::Vector{MeasurementOutcome}
end
function compute!(fidelity::Fidelity, register::QuantumRegister, cache::Dict{ComputeStep, Any} = [])::Float64
    # Compute the fidelity of the post-selected state with respect to the ideal target state by computing the density matrix of the post-selected state and then computing the appropriate overlap with the ideal state
    # TODO
end

end # module
