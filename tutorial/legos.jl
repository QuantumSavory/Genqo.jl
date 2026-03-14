using Revise, Genqo


circuit = circuits.Circuit(8)
q = circuit.register

# Prepare TMSV states
# r = 1.0
# gates.Squeeze2Mode(r) | q[1,2]
# gates.Squeeze2Mode(r) | q[3,4]
# gates.Squeeze2Mode(r) | q[5,6]
# gates.Squeeze2Mode(r) | q[7,8]

# Initialize TMSV states (equivalent to above)
μ = 0.2
states.TMSVState(μ) >> q[1,2]
states.TMSVState(μ) >> q[3,4]
states.TMSVState(μ) >> q[5,6]
states.TMSVState(μ) >> q[7,8]

# Apply mode swaps
gates.ModeSwap() | q[2,4]
gates.ModeSwap() | q[6,8]

# Apply beamsplitters
gates.BeamSplitter() | q[3,5]
gates.BeamSplitter() | q[4,6]

# Incorporate transmission loss
ηᵗ = 0.8
# gates.LossChannel(ηᵗ) | q[1,2,7,8]

# Heralding detection with loss
ηᵈ = 0.9
# measurements.PhotonNumDetector(ηᵈ) | q[3,4,5,6]

# Run and analyze covariance matrix
circuits.run!(circuit)

old = zalm.covariance_matrix(μ)
new = circuit.register.state.covariance

display(old)
display(new)

if all(abs.(old - new) .< 1e-8)
    println("Success! Covariance matrices match.")
else
    println("Error: Covariance matrices do not match.")
end
