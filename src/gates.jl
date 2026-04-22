module gates

using Nemo
using LinearAlgebra

using ..states: GaussianState
using ..tools: W, k_function_matrix

export Gate, num_qubits, SymplecticGate, getTransformMatrix, expand, apply!, apply, BeamSplitter, Squeeze2Mode, ModeSwap, LossChannel
export ModeProjection, FockProjection, TraceOut, Remaining, Detector, PhotonNumDetector, PhotonThresholdDetector, FormalismTransition, GaussiantoCoherent


abstract type Gate end
num_qubits(::Gate) = -1 # Any number of qubits

struct SymplecticGate <: Gate
    S::Matrix{Float64}
end
num_qubits(::SymplecticGate) = 2

function getTransformMatrix(gate::SymplecticGate, indices::Vector{Int}, mds::Int)::Matrix{Float64}
    # Set rows/columns corresponding to qi, pi, qj, pj using row/column 1, 2, 3, 4 from `gate`. qqpp ordering.
    S = Matrix{Float64}(I, 2*mds, 2*mds)
    idx = [indices; (indices .+ mds)]
    @views S[idx,idx] .= gate.S

    return S
end

function expand(gate::SymplecticGate, indices::Vector{Int}, mds::Int)::SymplecticGate
    return SymplecticGate(getTransformMatrix(gate, indices, mds))
end

function apply!(gate::SymplecticGate, st::GaussianState)
    st.covariance .= gate.S * st.covariance * gate.S'
end

# All gates in qqpp representation

BeamSplitter(t::Real=0.5) = SymplecticGate(
    [
        √(t)     √(1-t)  0        0      ;
        -√(1-t)  √(t)    0        0      ;
        0        0       √(t)     √(1-t) ;
        0        0       -√(1-t)  √(t)   ;
    ]
)

Squeeze2Mode(r::Real, φ::Real=0) = begin
    coshr = cosh(r)
    sinhr = sinh(r)
    cosφ = cos(φ)
    sinφ = sin(φ)
    SymplecticGate(
        1/sqrt(2) * 
        [
            coshr       cosφ*sinhr  0            sinφ*sinhr  ;
            cosφ*sinhr  coshr       sinφ*sinhr   0           ;
            0           sinφ*sinhr  coshr        -cosφ*sinhr ;
            sinφ*sinhr  0           -cosφ*sinhr  coshr       ;
        ]
    )
end

ModeSwap() = SymplecticGate(
    [
        0   1   0   0 ;
        1   0   0   0 ;
        0   0   0   1 ;
        0   0   1   0 ;
    ]
)


abstract type Channel <: Gate end

struct LossChannel <: Channel
    η::Vector{Real}
end

function expand(ch::LossChannel, indices::Vector{Int}, mds::Int)::LossChannel
    expanded = ones(Float64, mds)
    if size(ch.η) == (1,)
        expanded[indices] .= ch.η[1]
    else
        @assert size(ch.η) == size(indices) "LossChannel η vector must have length equal to number of modes specified, but got length $(size(ch.η)) applied to $(size(indices)) modes"
        expanded[indices] .= ch.η
    end
    return LossChannel(expanded)
end

function apply!(ch::LossChannel, st::GaussianState)
    η = ch.η                # vector of per-mode transmissivities
    V = st.covariance
    mds = length(η)
    
    # Build the scaling vector: √ηᵢ for q and p of each mode
    d = vcat(sqrt.(η), sqrt.(η))   # length 2*mds, matches qqpp ordering
    
    # Apply V' = D V Dᵀ with D = diag(d)
    # This scales entire rows and columns, including off-diagonals
    V .= (d .* V) .* d'
    
    # Add the noise on the self-variance of each attenuated mode
    for i in 1:mds
        V[i,     i    ] += (1 - η[i]) / 2    # q-q noise
        V[i+mds, i+mds] += (1 - η[i]) / 2    # p-p noise
    end
    return st
end

function LossChannel(η::Real...)
    @assert all(0 .≤ η .≤ 1) "All loss values must be between 0 and 1"
    return LossChannel(collect(η))
end


end # module
