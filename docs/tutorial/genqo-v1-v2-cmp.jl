using Genqo
using Gabs: eprstate, QuadBlockBasis, ⊗

using Plots

## ZALM Probability of Success

# Create empty circuit
q = QCircuit(8)
basis = QuadBlockBasis(2)

# Apply mode swaps
modeswap(basis) | q[2,4]
modeswap(basis) | q[5,7]

# Apply beamsplitters
beamsplitter(basis, 0.5) | q[3,5]
beamsplitter(basis, 0.5) | q[4,6]

# Incorporate losses
ηᵗ = 0.8
ηᵈ = 0.4
LossChannel(ηᵗ) | q[1,2,7,8]
LossChannel(ηᵈ) | q[3:6]

# Heralding detection
dets = PhotonNumDetector([1,1,0,0]) | q[3:6]

q_fused = fuse(q)
engine = HybridGaussianCoherentEngine(8)

function zalm_probability(μ::Float64)::Float64
    # Initialize TMSV states
    # TODO: is there a slicker way to parametrize any part of the circuit? Like losses, transmissivities, etc? Maybe symbolics would help?
    tmsv = eprstate(QuadBlockBasis(2), asinh(√μ), 0.)
    run!(
        engine,
        q_fused, 
        tmsv ⊗ tmsv ⊗ tmsv ⊗ tmsv
    )
end

μ = logrange(1e-4, 10, 100)
# η = 10 .^ -([0, 1, 3, 5]/10)

zalm_Pg_ground = zalm.probability_success.(μ, ηᵗ, 1, ηᵈ, 0)
zalm_Pg_new = zalm_probability.(μ)

plot(μ, zalm_Pg_ground, label="Genqo v1 (ground truth)", xscale=:log10, yscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Probability of Success", legend=:bottomright, color=[1 2 3 4])
plot!(μ, zalm_Pg_new, label="Genqo v2", linestyle=:dash, color=[1 2 3 4])
