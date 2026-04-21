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
function compute!(::AMatrix, register::QuantumRegister, cache::Dict{ComputeStep, Any} = [])::Matrix{ComplexF64}
    get!(cache, AMatrix()) do
        # Compute the A⁻¹ matrix from the Gaussian state covariance matrix
        K = k_function_matrix(register.state.covariance)
        return inv(K)
    end
end

# TODO: generalize so we just compute Ainv from the detection spec, instead of hardcoding for pgen (here we assume 1,2,7,8 are traced out)
struct ApgenMatrix <: ComputeStep end
function compute!(::ApgenMatrix, register::QuantumRegister, cache::Dict{ComputeStep, Any} = [])::Matrix{ComplexF64}
    get!(cache, ApgenMatrix()) do
        # Compute the A⁻¹ matrix from the Gaussian state covariance matrix
        K = k_function_matrix(register.state.covariance)
        G = zeros(ComplexF64, size(K))
        mds = register.mds
        for i in [1,2,7,8]
            G[i,      i+2mds] = -1
            G[i,      i+3mds] = im
            G[i+mds,  i+2mds] = -im
            G[i+mds,  i+3mds] = -1
        end
        return inv(K + (G + transpose(G) + I) / 2)
    end
end

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
    # Compute the probability of the given detection outcome by building the appropriate moment polynomial and performing the necessary Wick contractions
    # TODO: this is just an example for a particular detection outcome; need to build the appropriate polynomial based on the detection outcome
    Ainv = compute!(ApgenMatrix(), register, cache)
    Γ = register.state.covariance + 0.5*I
    detΓ = det(Γ)
    # C = build_moment_polynomial(probability.detection_outcome, register.α, register.β, register.R)
    C = register.α[3]*register.α[4]*register.β[3]*register.β[4]
    return (det(Ainv) / (detΓ^0.25 * conj(detΓ)^0.25)) * W(C, Ainv)
end

# Compute fidelity of post-selected state with respect to ideal target state
struct Fidelity <: Metric
    ideal_state::QuantumState
    detection_outcome::Vector{DetectionOutcome}
end
function compute!(fidelity::Fidelity, register::QuantumRegister, cache::Dict{ComputeStep, Any} = [])::Float64
    # Compute the fidelity of the post-selected state with respect to the ideal target state by computing the density matrix of the post-selected state and then computing the appropriate overlap with the ideal state
    # TODO
end

end # module
