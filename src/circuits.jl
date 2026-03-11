module circuits

using ..gates
using ..registers

export Circuit, fuse, fuse!, run


mutable struct Circuit
    register::QuantumRegister
end
Circuit(mds::Int) = Circuit(QuantumRegister(mds))

function fuse(circuit::Circuit)::Circuit
    if circuit.register.builder.fused
        return circuit
    end
    _circuit = deepcopy(circuit)
    fuse!(_circuit)

    return _circuit
end

function fuse!(circuit::Circuit)
    # Skip if already marked as fused
    if circuit.register.builder.fused return end

    # Clear fused ops cache before re-fusing
    circuit.register.builder.ops_fused = []

    # Expand gates and fuse where possible
    for (gate, indices) in circuit.register.builder.ops
        # Expand gate to the full number of modes in the quantum register
        # For instance, this could take a 2x2 beamsplitter matrix and produce a 16x16 (8-mode) qqpp matrix applied to modes 3,5
        gate_expanded = expand(gate, indices, circuit.register.mds)

        # Fuse with previous gate if both gates are linear
        if gate_expanded isa Linear2QubitGate && !isempty(circuit.register.builder.ops_fused)
            prev_gate = circuit.register.builder.ops_fused[end]
            if prev_gate isa Linear2QubitGate
                fused_S = gate_expanded.S * prev_gate.S # left-multiply by new gate
                circuit.register.builder.ops_fused[end] = Linear2QubitGate(fused_S)
            else
                push!(circuit.register.builder.ops_fused, gate_expanded)
            end
        else
            push!(circuit.register.builder.ops_fused, gate_expanded)
        end
    end

    # TODO: have fuser handle qqpp vs qpqp representations
    # TODO Future: add more optimizations

    circuit.register.builder.fused = true
    return
end

function run!(circuit::Circuit)
    fuse!(circuit)

    for gate in circuit.register.builder.ops_fused
        apply!(gate, circuit.register.covariance)
    end
end

end # module
