module registers

using Nemo
using LinearAlgebra

using ..states
using ..gates
using ..detectors

export CircuitBuilder, QuantumRegister, ModeRef


Base.@kwdef mutable struct CircuitBuilder
    ops::Vector{Tuple{Gate, Vector{Int}}} = []
    ops_fused::Vector{Gate} = [] # for storing optimized operations after fusion
    fused::Bool = false
end

# QuantumRegister: holds the covariance matrix representing the state of the quantum system and circuit builder to hold a list of gate operations.
Base.@kwdef mutable struct QuantumRegister
    mds::Int
    state::QuantumState
    builder::CircuitBuilder

    R::Generic.MPolyRing{ComplexFieldElem}
    qai::Vector{Generic.MPoly{ComplexFieldElem}}
    pai::Vector{Generic.MPoly{ComplexFieldElem}}
    qbi::Vector{Generic.MPoly{ComplexFieldElem}}
    pbi::Vector{Generic.MPoly{ComplexFieldElem}}
    α::Vector{Generic.MPoly{ComplexFieldElem}}
    β::Vector{Generic.MPoly{ComplexFieldElem}}
end

function QuantumRegister(mds::Int)
    # TODO: can we infer the best engine to start with? or at least have user specify
    state = VacuumState(mds)
    builder = CircuitBuilder()

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

    # Define the alpha and beta vectors
    α = (qai + i .* pai) / sqrt(2)
    β = (qbi - i .* pbi) / sqrt(2)

    return QuantumRegister(mds, state, builder, R, qai, pai, qbi, pbi, α, β)
end

struct ModeRef
    register::QuantumRegister
    indices::Vector{Int}
end

# Support indexing (syntax: q[1], q[2], ...)
Base.getindex(register::QuantumRegister, i::Int) = ModeRef(register, [i])

# Support multiple indexing (syntax: q[1,3], q[2,3,7,8], ...)
Base.getindex(register::QuantumRegister, is::Int...) = ModeRef(register, collect(is))

# Support initializing mode(s) with a state. Syntax: state(...) >> q[1,2]
function Base.:(>>)(state::GaussianState, modes::ModeRef)
    # For now, we only support initializing Gaussian states, which can be done by directly setting the covariance matrix of the register.
    @assert size(state.covariance) == (2length(modes.indices), 2length(modes.indices)) "Cannot initialize $(length(modes.indices)) modes with a state that has $(size(state.covariance,1)÷2) modes"
    mds = modes.register.mds
    idx = [modes.indices; (mds .+ modes.indices)] # Get indices for both q and p quadratures of the modes being initialized
    @views modes.register.state.covariance[idx, idx] .= state.covariance
    return
end

# Support applying a gate to modes. Syntax: gate(...) | q[1,3]
function Base.:(|)(gate::Gate, modes::ModeRef)
    nq = num_qubits(gate)
    if nq != -1 # -1 means gate can be applied to any number of modes
        @assert length(modes.indices) == nq "$(typeof(gate)) can only be applied to $nq modes, but got $(length(modes.indices)) modes"
    end

    # Push gate and modes to the circuit builder, to be optimized/applied when the circuit is run.
    push!(modes.register.builder.ops, (gate, modes.indices))

    # Unset fused flag so the fuser knows to refuse the circuit with the new gate
    modes.register.builder.fused = false
    return # TODO: return the gate in some form for consistency
end

# Support applying a detector to modes. Syntax: dets = detector() << q[6,7]
# We cannot compute the moment polynomial until we know the detection outcome, so we simply have the detectors remember what modes they were applied to
function Base.:(<<)(::PhotonNumDetector, modes::ModeRef)
    if length(modes.indices) == 1
        return PhotonNumDetector(modes.indices[1])
    else
        return [PhotonNumDetector(mode) for mode in modes.indices]
    end
end

function Base.:(<<)(::PhotonThresholdDetector, modes::ModeRef)
    if length(modes.indices) == 1
        return PhotonThresholdDetector(modes.indices[1])
    else
        return [PhotonThresholdDetector(mode) for mode in modes.indices]
    end
end

end # module
