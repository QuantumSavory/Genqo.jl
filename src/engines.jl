module engines

using Nemo
using LinearAlgebra
using Gabs: GaussianState, vacuumstate, changebasis

using ..circuits
using ..gates
using ..gates: default_basis
using ..tools

export AbstractEngine, HybridGaussianCoherentEngine
export run!


abstract type AbstractEngine end

mutable struct HybridGaussianCoherentEngine <: AbstractEngine
    mds::Int

    # Pure Gaussian state before loss and measurement
    gaussian_state::GaussianState

    # Phase-space ring and variables for symbolic calculations
    R::Generic.MPolyRing{ComplexFieldElem}
    qai::Vector{Generic.MPoly{ComplexFieldElem}}
    pai::Vector{Generic.MPoly{ComplexFieldElem}}
    qbi::Vector{Generic.MPoly{ComplexFieldElem}}
    pbi::Vector{Generic.MPoly{ComplexFieldElem}}
    α::Vector{Generic.MPoly{ComplexFieldElem}}
    β::Vector{Generic.MPoly{ComplexFieldElem}}

    function HybridGaussianCoherentEngine(mds::Int)
        # Define canonical phase-space variables for the circuit
        _qai = ["qa$i" for i in 1:mds]
        _pai = ["pa$i" for i in 1:mds]
        _qbi = ["qb$i" for i in 1:mds]
        _pbi = ["pb$i" for i in 1:mds]
        all_qps = hcat(_qai, _pai, _qbi, _pbi)
        CC = ComplexField()
        i = onei(CC) # Imaginary unit in CC ring
        R, generators = polynomial_ring(CC, all_qps)
        (qai, pai, qbi, pbi) = (generators[:,i] for i in 1:4)

        # Define the α and β* vectors (note that we pre-conjugate β)
        α = (qai + i .* pai) / sqrt(2)
        β = (qbi - i .* pbi) / sqrt(2)

        return new(mds, vacuumstate(default_basis(mds)), R, qai, pai, qbi, pbi, α, β)
    end
end

function run!(engine::AbstractEngine, circuit::QCircuit)
    error("run! not implemented for engine of type $(typeof(engine)) on circuit of type $(typeof(circuit))")
end

function run!(engine::HybridGaussianCoherentEngine, circuit::HeraldedGaussianCircuit, initial_state::GaussianState = vacuumstate(default_basis(engine.mds)))::ComplexF64
    @assert engine.mds == circuit.mds "Engine and circuit must have the same number of modes"

    # Apply fused gates in order
    engine.gaussian_state = initial_state
    for gate in circuit.gates
        engine.gaussian_state = apply!(engine.gaussian_state, gate)
    end

    # Convert to K-function representation and compute probabilities.
    # Gabs uses the ħ=2 convention (vacuum variance 1), but the K-function machinery
    # assumes ħ=1 (vacuum variance 1/2, hence the `Γ = σ + ½I` below), so rescale by 1/ħ.
    gstate = changebasis(default_basis, engine.gaussian_state)
    σ = gstate.covar ./ gstate.ħ
    A = k_function_matrix(σ) + G_matrix(circuit.losses, circuit.detectors, circuit.mds)

    invA, detA = inv(A), det(A)
    C = C_poly(engine, circuit)
    Γ = σ + 0.5*I
    detΓ = det(Γ)

    prob = W(C, invA) / (sqrt(detA)*detΓ^0.25*conj(detΓ)^0.25)

    # TODO: return the post-measurement state as well as probability
    return prob
end


# G matrix only cares about the type of measurement (photon number vs trace out) and not the specific outcome, so we can compute it directly from the detector layout
function G_matrix(η::Vector{Float64}, detector::Detector, mds::Int)::Matrix{ComplexF64}
    # Expansion of (|α|² + |β|²)(1 - η) / 2
    # G = diagm(ComplexF64.(vcat(η, η, η, η) / 2))
    G = 0.5 * Matrix{ComplexF64}(I, 4mds, 4mds)

    if detector isa PhotonNumDetector
        for (i, n) in enumerate(detector.outcomes)
            if n == -1
                # No detector means we trace out the mode
                # Expansion of αβ*
                G[i,      i+2mds] = -0.5
                G[i,      i+3mds] = 0.5im
                G[i+mds,  i+2mds] = -0.5im
                G[i+mds,  i+3mds] = -0.5
                G[i+2mds, i     ] = -0.5
                G[i+2mds, i+mds ] = -0.5im
                G[i+3mds, i     ] = 0.5im
                G[i+3mds, i+mds ] = -0.5
            else
                # Fock term (αβ*)ⁿ/n! is handled in the moment polynomial C, as it is not inside an exp()
                # Expansion of αβ*(1 - η)
                G[i,      i+2mds] = -0.5 * (1 - η[i])
                G[i,      i+3mds] = 0.5im * (1 - η[i])
                G[i+mds,  i+2mds] = -0.5im * (1 - η[i])
                G[i+mds,  i+3mds] = -0.5 * (1 - η[i])
                G[i+2mds, i     ] = -0.5 * (1 - η[i])
                G[i+2mds, i+mds ] = -0.5im * (1 - η[i])
                G[i+3mds, i     ] = 0.5im * (1 - η[i])
                G[i+3mds, i+mds ] = -0.5 * (1 - η[i])
            end
        end
    
    elseif detector isa PhotonThresholdDetector
        # TODO
        # Imagining we will have a term like trace out here, and -(αβ*)ⁿ/n! terms in the C polynomial
    end

    return G
end

function C_poly(engine::HybridGaussianCoherentEngine, circuit::HeraldedGaussianCircuit)::WTerms
    # TODO: cache and measure speed improvement
    # get!(engine.cache, CPoly(detection_outcome)) do
        C = one(engine.R)
        if circuit.detectors isa PhotonNumDetector
            for (n, η, α, β) in zip(circuit.detectors.outcomes, circuit.losses, engine.α, engine.β)
                if n == 1
                    C *= (α * β * η)
                # TODO: support higher photon number outcomes as well, which will involve including the appropriate Fock term (αβ*)ⁿ/n! in the C polynomial
                # tools.W() will need to be generalized
                end
            end
        end

        return extract_W_terms(C)
    # end
end

end # module
