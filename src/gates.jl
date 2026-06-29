module gates

using LinearAlgebra

using Gabs: Gabs, SymplecticBasis, QuadBlockBasis, nmodes, GaussianState, GaussianUnitary, embed

export Gate, WrappedGaussianUnitary, modeswap, LossChannel, Detector, PhotonNumDetector, PhotonThresholdDetector
export apply!


# TODO: make the k-function machinery capable of handling a different basis by responding to what this function specifies
const default_basis = QuadBlockBasis

abstract type Gate end

struct WrappedGaussianUnitary <: Gate
    gate::GaussianUnitary
end
Base.convert(::Type{WrappedGaussianUnitary}, gate::GaussianUnitary) = WrappedGaussianUnitary(gate)
Base.convert(::Type{GaussianUnitary}, gate::WrappedGaussianUnitary) = gate.gate
apply!(state::GaussianState, gate::WrappedGaussianUnitary, indices::Vector{Int}) = Gabs.apply!(state, gate.gate, indices)
function Base.:*(gate1::GaussianUnitary, gate2::GaussianUnitary)
    mds = nmodes(gate1.gabs_gate)
    GaussianUnitary(default_basis(mds), zeros(2mds), gate1.symplectic * gate2.symplectic)
end

modeswap(basis::SymplecticBasis) = GaussianUnitary(
    basis,
    zeros(2*2),
    [
        0   1   0   0 ;
        1   0   0   0 ;
        0   0   0   1 ;
        0   0   1   0 ;
    ]
)

struct LossChannel <: Gate
    η::Vector{Float64}
end
LossChannel(η::Real) = LossChannel([η])


abstract type Detector <: Gate end

struct PhotonNumDetector <: Detector
    outcomes::Vector{Int}
end
function expand(gate::PhotonNumDetector, indices::Vector{Int}, mds::Int)
    outcomes = -ones(Int, mds)
    outcomes[indices] .= gate.outcomes
    return PhotonNumDetector(outcomes)
end

# TODO: coming soon
# struct PhotonThresholdDetector <: Detector
#     outcomes::Vector{Bool}
# end

end # module
