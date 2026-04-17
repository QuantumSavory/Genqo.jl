"""
metrics.jl: This module contains functions for computing performance metrics such as probability of success and fidelity for quantum circuits.
"""
module metrics

using LinearAlgebra

using ..tools: W, k_function_matrix
using ..gates
using ..states
using ..registers
using ..detectors: DetectionOutcome

export Metric, Probability, Fidelity, compute!


abstract type ComputeStep end

struct AMatrix <: ComputeStep end
function compute_Ainv(state::GaussianState)::Matrix{ComplexF64}
    # Compute the A⁻¹ matrix from the Gaussian state covariance matrix
    A = k_function_matrix(state.covariance) # A = K + G in the paper, but we treat loss as operation on Gaussian state so G=0.
    return inv(A)
end
function compute!(::AMatrix, register::QuantumRegister, cache::Dict{ComputeStep, Any} = [])::Matrix{ComplexF64}
    get!(cache, AMatrix()) do
        # Compute the A⁻¹ matrix from the Gaussian state covariance matrix
        return compute_Ainv(register.state)
    end
end

struct ApgenMatrix <: ComputeStep end

struct DensityOperatorElem <: ComputeStep
    d::Int
    g::Int
end
function compute!(elem::DensityOperatorElem, register::QuantumRegister, cache::Dict{ComputeStep, Any} = [])::ComplexF64
    get!(cache, elem) do
        Ainv = compute!(AMatrix(), register, cache)
        # TODO
    end
end

abstract type Metric end

# Compute probability of a given detection outcome
struct Probability <: Metric
    detection_outcome::Vector{DetectionOutcome}
end
function compute!(probability::Probability, register::QuantumRegister, cache::Dict{ComputeStep, Any} = [])::Float64
    get!(cache, probability) do
        # Compute the probability of the given detection outcome by building the appropriate moment polynomial and performing the necessary Wick contractions
        # TODO: this is just an example for a particular detection outcome; need to build the appropriate polynomial based on the detection outcome
        Ainv = compute!(AMatrix(), register, cache)
        Γ = register.state.covariance + 0.5*I
        detΓ = det(Γ)
        # C = build_moment_polynomial(probability.detection_outcome, register.α, register.β, register.R)
        C = register.α[3]*register.α[4]*register.β[3]*register.β[4]
        return (det(Ainv) / (detΓ^0.25 * conj(detΓ)^0.25)) * W(C, Ainv)
    end
end

# Compute fidelity of post-selected state with respect to ideal target state
struct Fidelity <: Metric
    ideal_state::QuantumState
    detection_outcome::Vector{DetectionOutcome}
end
function compute!(fidelity::Fidelity, register::QuantumRegister, cache::Dict{ComputeStep, Any} = [])::Float64
    get!(cache, fidelity) do
        # Compute the fidelity of the post-selected state with respect to the ideal target state by computing the density matrix of the post-selected state and then computing the appropriate overlap with the ideal state
        # TODO
    end
end

end # module
