module circuits

using ..gates
using ..registers
using ..metrics
using ..metrics: ComputeStep
using ..detectors

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

        # Fusion rules
        # Fuse with previous gate if both gates are symplectic
        if gate_expanded isa SymplecticGate && !isempty(circuit.register.builder.ops_fused)
            prev_gate = circuit.register.builder.ops_fused[end]
            if prev_gate isa SymplecticGate
                fused_S = gate_expanded.S * prev_gate.S # left-multiply by new gate
                circuit.register.builder.ops_fused[end] = SymplecticGate(fused_S)
            else
                push!(circuit.register.builder.ops_fused, gate_expanded)
            end

        # Fuse with previous gate if both gates are loss channels
        elseif gate_expanded isa LossChannel && !isempty(circuit.register.builder.ops_fused)
            prev_gate = circuit.register.builder.ops_fused[end]
            if prev_gate isa LossChannel
                fused_η = prev_gate.η .* gate_expanded.η # elementwise multiply loss vectors
                circuit.register.builder.ops_fused[end] = LossChannel(fused_η)
            else
                push!(circuit.register.builder.ops_fused, gate_expanded)
            end

        # Default: push gate with no fusion
        else
            push!(circuit.register.builder.ops_fused, gate_expanded)
        end
    end

    # TODO Future: add more optimizations

    circuit.register.builder.fused = true
    return
end

function run!(circuit::Circuit)
    # Run fuser to expand gates and optimize circuit
    fuse!(circuit)

    # Apply gates in order
    for gate in circuit.register.builder.ops_fused
        apply!(gate, circuit.register.state)
    end
end

function analyze!(circuit::Circuit, metrics::Vector{<:Metric})::Dict{<:Metric, Any}
    run!(circuit)

    # Compute each metric, caching intermediate results as needed for efficiency
    results = Dict{Metric, Any}()
    cache = Dict{ComputeStep, Any}()
    for metric in metrics
        results[metric] = compute!(metric, circuit.register, cache)
    end
    return results
end
analyze!(circuit::Circuit, metric::Metric) = analyze!(circuit, [metric])[metric]

end # module
