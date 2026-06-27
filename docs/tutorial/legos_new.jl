using Genqo
using Gabs: eprstate, QuadBlockBasis, ⊗

using BenchmarkTools
using Plots


# Create empty circuit
q = QCircuit(8)

# Apply mode swaps
modeswap | q[2,4]
modeswap | q[5,7]

# Apply beamsplitters
beamsplitter() | q[3,5]
beamsplitter() | q[4,6]

# Incorporate losses
ηᵗ = 1.0
ηᵈ = 0.8
LossChannel(ηᵗ) | q[1,2,7,8]
LossChannel(ηᵈ) | q[3,4,5,6] # TODO: allow 3:6

# Heralding detection
dets = PhotonNumDetector([1,1,0,0]) | q[3,4,5,6]

q_fused = fuse(q)
engine = HybridGaussianCoherentEngine(8)

function zalm_probability(μ::Float64)::Float64
    # Initialize TMSV states
    # TODO: is there a slicker way to parametrize any part of the circuit? Like losses, transmissivities, etc? Maybe symbolics would help?
    tmsv = eprstate(QuadBlockBasis(2), asinh(√μ), 0.)
    run!(engine, q_fused, tmsv⊗tmsv⊗tmsv⊗tmsv)
end

μ = logrange(1e-4, 1e2, 100)
@time P = zalm_probability.(μ)

plot(μ, P, xscale=:log10, yscale=:log10)
