using Genqo
using Plots

## ZALM Probability of Success

function zalm_probability(μ, ηᵈ)
    circuit = circuits.Circuit(8)
    q = circuit.register

    # Initialize TMSV states
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
    gates.LossChannel(ηᵗ) | q[1,2,7,8]
    gates.LossChannel(ηᵈ) | q[3,4,5,6]

    # Heralding detection
    detectors.PhotonNumDetector() << q[3,4,5,6]

    # Find performance metrics
    # Analyze just one metric at one point
    success = registers.MeasurementOutcome(q[3,4,5,6] => [1,1,0,0]) # Example measurement outcome for heralding measurement counted as success
    return circuits.analyze!(
        circuit,
        metrics.Probability(success),
    )
end

μ = logrange(1e-4, 10, 100)
η = 10 .^ -([0, 1, 3, 5]/10)

zalm_Pg_ground = zalm.probability_success.(μ, 1, 1, η', 0)
zalm_Pg_new = zalm_probability.(μ, η')

plot(μ, zalm_Pg_ground, label="Ground truth", xscale=:log10, yscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Probability of Success", legend=:bottomright, color=[1 2 3 4])
plot!(μ, zalm_Pg_new, label="Genqo v2", linestyle=:dash, color=[1 2 3 4])

## Single Sagnac Probability of Success

function sigsag_probability(μ, ηᵈ)
    circuit = circuits.Circuit(6)
    q = circuit.register

    # Initialize TMSV states
    states.TMSVState(μ) >> q[1,2]
    states.TMSVState(μ) >> q[3,4]

    # Apply mode swaps
    gates.ModeSwap() | q[2,4]

    # Find performance metrics
    # Analyze just one metric at one point
    success = registers.MeasurementOutcome()
    return circuits.analyze!(
        circuit,
        metrics.Probability(success),
    )
end

μ = logrange(1e-4, 10, 100)
η = 10 .^ -([0, 1, 3, 5]/10)

sigsag_Pg_ground = sigsag.probability_success.(μ, 1, 1, η', 0)
sigsag_Pg_new = sigsag_probability.(μ, η')

plot(μ, sigsag_Pg_ground, label="Ground truth", xscale=:log10, yscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Probability of Success", legend=:bottomright)
plot!(μ, sigsag_Pg_new, label="Genqo v2", linestyle=:dash)
