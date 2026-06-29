module circuits

using Nemo
using LinearAlgebra
using Gabs: GaussianUnitary, embed, nmodes

using ..gates
using ..gates: default_basis

export AbstractQCircuit, QCircuit, FusedQCircuit, HeraldedGaussianCircuit
export fuse


abstract type AbstractQCircuit end

mutable struct QCircuit <: AbstractQCircuit
    mds::Int
    gates::Vector{Tuple{Gate, Vector{Int}}}

    QCircuit(mds::Int) = new(mds, [])
end

struct ModeRef{T<:AbstractQCircuit}
    circuit::Ref{T}
    indices::Vector{Int}
end

# Support indexing. Syntax: q[1], q[2,3]
Base.getindex(circuit::AbstractQCircuit, i::Int) = ModeRef(Ref(circuit), [i])
Base.getindex(circuit::AbstractQCircuit, is::Int...) = ModeRef(Ref(circuit), collect(is))
Base.getindex(circuit::AbstractQCircuit, is::AbstractUnitRange{Int}) = ModeRef(Ref(circuit), collect(is))

# Support applying a gate to modes. Syntax: gate(...) | q[1,3]
function Base.:(|)(gate::Gate, modes::ModeRef)
    # Expand gate to the full number of modes in the quantum register
    # For instance, this could take a 2x2 beamsplitter matrix and produce a 16x16 (8-mode) qqpp matrix applied to modes 3,5
    push!(modes.circuit[].gates, (gate, modes.indices)) # Push expanded gate to the circuit
end
function Base.:(|)(gate::GaussianUnitary, modes::ModeRef)
    # @assert gate.basis == default_basis(modes.circuit[].mds) "Gate basis $(gate.basis) does not match circuit basis $(default_basis(modes.circuit[].mds))"
    @assert length(modes.indices) == nmodes(gate) "Gate has $(nmodes(gate)) modes, but got $(length(modes.indices)) indices"
    push!(modes.circuit[].gates, (WrappedGaussianUnitary(gate), modes.indices))
end


abstract type FusedQCircuit <: AbstractQCircuit end

struct HeraldedGaussianCircuit <: FusedQCircuit
    mds::Int
    gates::Vector{Tuple{WrappedGaussianUnitary, Vector{Int}}}
    losses::Vector{Float64}
    detectors::Detector
end

# TODO Future: allow fuser to choose which kind of FusedQCircuit to make, depending on the circuit layout and available engines
# For right now, this only ever fuses into HeraldedGaussianCircuit
function fuse(circuit::QCircuit)::FusedQCircuit
    gates = []
    detectors = PhotonNumDetector(repeat([-1], circuit.mds))

    # TODO: eventually we want to check that the circuit can be decomposed into A and B, where A is a Gaussian circuit and B is a circuit of Fock measurements and losses. For now we expect gates to come in the order of Gaussian gates followed by losses followed by measurements.

    # Expand gates and fuse where possible
    losses = ones(circuit.mds)
    for (gate, indices) in circuit.gates
        # Fusion rules
        # Fuse with previous gate if both gates are GaussianUnitary
        if gate isa GaussianUnitary && !isempty(gates)
            # prev_gate = gates[end]
            # if prev_gate isa GaussianUnitary
            #     gates[end] = gate * prev_gate # left-multiply by new gate
            # else
                push!(gates, (gate, indices))
            # end

        # Collect all losses and apply at the end of the circuit, since losses commute with all gates
        # TODO: this can't be right. Losses need to go through the gates that come after them
        elseif gate isa LossChannel
            losses[indices] .*= gate.η # update cumulative losses

        elseif gate isa Detector
            # TODO: make this |= operator to combine detectors
            detectors = expand(gate, indices, circuit.mds) # expand detector to full number of modes

        # Default: push gate with no fusion
        else
            push!(gates, (gate, indices))
        end
    end

    # TODO Future: add more optimizations
    
    return HeraldedGaussianCircuit(circuit.mds, gates, losses, detectors)
end

end # module
