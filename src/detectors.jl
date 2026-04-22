module detectors

using LinearAlgebra

export Detector, PhotonNumDetector, PhotonThresholdDetector
export G_matrix


abstract type Detector end

struct PhotonNumDetector <: Detector
    mode::Int
end
PhotonNumDetector() = PhotonNumDetector(0)

struct PhotonThresholdDetector <: Detector
    mode::Int
end
PhotonThresholdDetector() = PhotonThresholdDetector(0)


# G matrix only cares about the type of measurement (photon number vs trace out) and not the specific outcome, so we can compute it directly from the detector layout
function G_matrix(detectors::Vector{Union{Detector, Nothing}}, mds::Int)::Matrix{ComplexF64}
    # Expansion of |α|² + |β|²
    G = 0.5 * Matrix{ComplexF64}(I, 4mds, 4mds)

    for (i, detector) in enumerate(detectors)
        if detector isa PhotonNumDetector
            # Fock term (αβ*)ⁿ/n! is handled in the moment polynomial C, as it is not inside an exp()

        elseif detector isa PhotonThresholdDetector
            # TODO
            # Imagining we will have a term like trace out here, and -(αβ*)ⁿ/n! terms in the C polynomial

        elseif detector === nothing
            # No detector means we trace out the mode
            # Expansion of αβ*
            G[i,      i+2mds] = -0.5
            G[i,      i+3mds] = 0.5im
            G[i+mds,  i+2mds] = -0.5im
            G[i+mds,  i+3mds] = -0.5
            G[i+2mds, i     ] = -0.5
            G[i+2mds, i+mds ] = -0.5im
            G[i+3mds, i     ] = 0.5im
            G[i+3mds, i+mds ] = -0.5
        end
    end

    return G
end

end # module
