using Genqo


circuit = circuits.Circuit(8)
q = circuit.register

# Prepare TMSV states
gates.Squeeze2Mode(1.0) | q[1,2]
gates.Squeeze2Mode(1.0) | q[3,4]
gates.Squeeze2Mode(1.0) | q[5,6]
gates.Squeeze2Mode(1.0) | q[7,8]

# Apply mode swaps
gates.ModeSwap() | q[2,4]
gates.ModeSwap() | q[5,7]

# Apply beamsplitters
gates.BeamSplitter() | q[3,5]
gates.BeamSplitter() | q[4,6]

# Heralding detection
detectors.PhotonNumDetector() | q[3,4,5,6]

# Run and analyze covariance matrix
circuits.run!(circuit)
