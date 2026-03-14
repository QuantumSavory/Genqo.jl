module registers

using LinearAlgebra

using ..states
using ..gates

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
end

function QuantumRegister(mds::Int)
    # TODO: can we infer the best engine to start with? or at least have user specify
    state = GaussianState(Matrix{Float64}(I, 2mds, 2mds)) # Start in vacuum state
    builder = CircuitBuilder()
    return QuantumRegister(mds, state, builder)
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
    return
end

end # module
