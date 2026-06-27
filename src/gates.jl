module gates

using LinearAlgebra

using Gabs: QuadBlockBasis, nmodes, GaussianState, GaussianUnitary as GabsGaussianUnitary, beamsplitter as Gabsbeamsplitter

export Gate, GaussianUnitary, modeswap, beamsplitter, LossChannel, Detector, PhotonNumDetector, PhotonThresholdDetector
export apply!, expand


# TODO: make the k-function machinery capable of handling a different basis by responding to what this function specifies
const default_basis = QuadBlockBasis

abstract type Gate end

struct GaussianUnitary <: Gate
    gabs_gate::GabsGaussianUnitary
end
convert(::Type{GaussianUnitary}, gabs_gate::GabsGaussianUnitary) = GaussianUnitary(changebasis(default_basis(nmodes(gabs_gate)), gabs_gate))
apply!(gaussian_state::GaussianState, gate::GaussianUnitary) = gate.gabs_gate * gaussian_state
function expand(gate::GaussianUnitary, indices::Vector{Int}, mds::Int)
    # Set rows/columns corresponding to qi, pi, qj, pj using row/column 1, 2, 3, 4 from `gate`. qqpp ordering.
    S = Matrix{Float64}(I, 2mds, 2mds)
    idx = [indices; (indices .+ mds)]
    @views S[idx,idx] .= gate.gabs_gate.symplectic
    return GaussianUnitary(GabsGaussianUnitary(default_basis(mds), zeros(2mds), S))
end
function Base.:*(gate1::GaussianUnitary, gate2::GaussianUnitary)
    mds = nmodes(gate1.gabs_gate)
    GaussianUnitary(GabsGaussianUnitary(default_basis(mds), zeros(2mds), gate1.gabs_gate.symplectic * gate2.gabs_gate.symplectic))
end

const modeswap = GaussianUnitary(
    GabsGaussianUnitary(
        default_basis(2),
        zeros(2*2),
        [
            0   1   0   0 ;
            1   0   0   0 ;
            0   0   0   1 ;
            0   0   1   0 ;
        ]
    )
)

beamsplitter(r::Real = 0.5) = GaussianUnitary(
    Gabsbeamsplitter(
        default_basis(2),
        r
    )
)

struct LossChannel <: Gate
    η::Vector{Float64}
end
LossChannel(η::Real) = LossChannel([η])
function expand(gate::LossChannel, indices::Vector{Int}, mds::Int)
    if length(gate.η) == 1
        η = ones(mds)
        η[indices] .= gate.η[1]
    else
        @assert length(gate.η) == length(indices) "LossChannel has $(length(gate.η)) loss values, but got $(length(indices)) indices"
        η = ones(mds)
        η[indices] .= gate.η
    end
    return LossChannel(η)
end


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
