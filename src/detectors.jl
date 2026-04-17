module detectors

export Detector, PhotonNumDetector, PhotonThresholdDetector
export DetectionOutcome, PhotonNumOutcome, PhotonThresholdOutcome


abstract type Detector end

struct PhotonNumDetector <: Detector
    mode::Int
end

struct PhotonThresholdDetector <: Detector
    mode::Int
end


abstract type DetectionOutcome end

struct PhotonNumOutcome <: DetectionOutcome
    mode::Int = -1 # TODO: can these be smarter and perhaps store references to the qubits themselves instead of just the mode index? That way you can't use a detector from one circuit on a different circuit by mistake.
    outcome::Int
end

struct PhotonThresholdOutcome <: DetectionOutcome
    mode::Int = -1
    outcome::Bool
end

function DetectionOutcome(detector_outcome::Pair{PhotonNumDetector, Int})
    detector, outcome = detector_outcome
    @assert outcome >= 0 "Photon number outcome must be a non-negative integer"
    return PhotonNumOutcome(detector.mode, outcome)
end

function DetectionOutcome(detector_outcome::Pair{PhotonThresholdDetector, Bool})
    detector, outcome = detector_outcome
    return PhotonThresholdOutcome(detector.mode, outcome)
end
DetectionOutcome(detector_outcome::Pair{PhotonThresholdDetector, Int}) = DetectionOutcome(detector_outcome.first => Bool(detector_outcome.second))

function DetectionOutcome(detectors_outcomes::Pair{Vector{Detector}, Vector{Int}})
    detectors, outcomes = detectors_outcomes
    @assert length(detectors) == length(outcomes) "Number of detectors and outcomes must match"
    return [DetectionOutcome(detector => outcome) for (detector, outcome) in zip(detectors, outcomes)]
end

end # module
