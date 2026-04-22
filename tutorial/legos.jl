using Revise, Genqo


circuit = circuits.Circuit(8)
q = circuit.register

# Initialize TMSV states
μ = 0.2
states.TMSVState(μ) >> q[1,2]
states.TMSVState(μ) >> q[3,4]
states.TMSVState(μ) >> q[5,6]
states.TMSVState(μ) >> q[7,8]

# Apply mode swaps
gates.ModeSwap() | q[2,4]
gates.ModeSwap() | q[5,7]

# Apply beamsplitters
gates.BeamSplitter() | q[3,5]
gates.BeamSplitter() | q[4,6]

# Incorporate losses
ηᵗ = 1.0
ηᵈ = 1.0
gates.LossChannel(ηᵗ) | q[1,2,7,8]
gates.LossChannel(ηᵈ) | q[3,4,5,6]

# Heralding detection
dets = detectors.PhotonNumDetector() << q[3,4,5,6]

# Find performance metrics
# Analyze just one metric at one point
success = registers.MeasurementOutcome(q[3,4,5,6] => [1,1,0,0]) # Example measurement outcome for heralding measurement counted as success
result = circuits.analyze!(
    circuit,
    metrics.Probability(success),
)

# outcome = registers.MeasurementOutcome(q[3,4,5,6] => [1,1,0,0]) # Example measurement outcome for heralding measurement counted as success
# result = metrics.probability(circuit, outcome)

# cache_per_point(circuit) do cache
#     Pg = metrics.probability(circuit, outcome; cache)
#     F = metrics.fidelity(circuit, ξ, outcome; cache)
# end

print(result)

## Analyze multiple metrics at one point (faster because certain Wick contractions can be reused across metrics)
success = registers.MeasurementOutcome([ # multiple measurement outcomes can also be used, and Genqo sums the probabilities across them
    dets => [1,1,0,0],
    dets => [0,0,1,1],
    dets => [0,1,1,0],
    dets => [1,0,0,1],
])
results = circuits.analyze(
    circuit,
    [
        metrics.Probability(success),
        metrics.Fidelity(ξ, success), # need to specify ideal state ξ as well as post-selection condition for fidelity calculation
        metrics.DensityMatrix(q[1,2,7,8], success),
    ],
)

print("Probability of Success: $(results[1].probability)")
print("Fidelity: $(results[2].fidelity)")
print("Density Matrix:")
display(results[3].density_matrix)

# Analyze one metric across multiple points (faster because parts of the circuit only need to be run once)
circuit = circuits.Circuit(8)
q = circuit.register

# Initialize TMSV states
# :μ is a Symbol, acting as a placeholder for the mean photon number that will be swept later.
states.TMSVState(:μ) >> q[1,2]
states.TMSVState(:μ) >> q[3,4]
states.TMSVState(:μ) >> q[5,6]
states.TMSVState(:μ) >> q[7,8]

# Apply mode swaps
gates.ModeSwap() | q[2,4]
gates.ModeSwap() | q[5,7]

# Apply beamsplitters
gates.BeamSplitter() | q[3,5]
gates.BeamSplitter() | q[4,6]

# Incorporate transmission loss
ηᵗ = 0.8
gates.LossChannel(ηᵗ) | q[1,2,7,8]

# Heralding detection with loss
ηᵈ = 0.9
# measurements.PhotonNumDetector(ηᵈ) << q[3,4,5,6]

results = circuits.analyze(
    circuit,
    MeasurementOutcome([q[3,4]=>1, q[5,6]=>0]), # Example measurement outcome for heralding measurement
    metrics.ProbabilitySuccess(),
    :μ => logrange(0.01, 10.0, 100), # Sweep mean photon number from 0.01 to 10 across 100 points
)
