module gates

using LinearAlgebra

export Gate, num_qubits, Linear2QubitGate, getTransformMatrix, expand, apply!, BeamSplitter


abstract type Gate end
num_qubits(::Gate) = -1 # Any number of qubits

struct Linear2QubitGate <: Gate
    S::Matrix{Float64}
end
num_qubits(::Linear2QubitGate) = 2

function getTransformMatrix(gate::Linear2QubitGate, i::Int, j::Int, mds::Int)::Matrix{Float64}
    S = Matrix{Float64}(I, 2*mds, 2*mds)
    S[i,i] = S[j,j] = S[mds+i,mds+i] = S[mds+j,mds+j] = gate.S[1,1]
    S[i,j] = S[mds+i,mds+j] = gate.S[1,2]
    S[j,i] = S[mds+j,mds+i] = gate.S[2,1]
    S[j,j] = S[mds+j,mds+j] = gate.S[2,2]

    return S
end

function expand(gate::Linear2QubitGate, indices::Vector{Int}, mds::Int)::Linear2QubitGate
    Linear2QubitGate(getTransformMatrix(gate, indices[1], indices[2], mds))
end

function apply!(gate::Linear2QubitGate, V::Matrix{Float64})
    V = gate.S * V * gate.S'
end

BeamSplitter(t::Real=0.5)::Linear2QubitGate = Linear2QubitGate(
    [
        sqrt(t)  sqrt(1-t);
        -sqrt(1-t) sqrt(t);
    ]
)

end # module
   