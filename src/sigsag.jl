module sigsag

using LinearAlgebra
using BlockDiagonals
using Nemo

import ..spdc
using ..tools


"""
    SIGSAG

Parameters for the heralded entanglement source architecture proposed by Chahine et al.

The SIGSAG source is a heralded dual-rail entanglement source architecture proposed by Yousef Chahine et al. as an alternative to the cascaded source architecture. It can be realized with a single Sagnac configured entanglement source, hence the nomenclature of "SIGSAG" for short.

# Fields
- `mean_photon::Real`           : Mean photon number per mode (default `1e-2`)
- `detection_efficiency::Real`  : Signal detector efficiency, ∈ [0, 1] (default `1.0`)
- `bsm_efficiency::Real`        : BSM detector efficiency, ∈ [0, 1] (default `1.0`)
- `outcoupling_efficiency::Real`: Photon outcoupling / transmission efficiency, ∈ [0, 1] (default `1.0`)
"""
Base.@kwdef mutable struct SIGSAG
    mean_photon::Real = 1e-2
    detection_efficiency::Real = 1.0
    bsm_efficiency::Real = 1.0
    outcoupling_efficiency::Real = 1.0
end

# Global canonical position and momentum variables
const mds = 6 # Number of modes for our system

_qai = ["qa$i" for i in 1:mds]
_pai = ["pa$i" for i in 1:mds]
_qbi = ["qb$i" for i in 1:mds]
_pbi = ["pb$i" for i in 1:mds]
all_qps = hcat(_qai, _pai, _qbi, _pbi)
CC = ComplexField()
i = onei(CC) # Imaginary unit in CC ring
R, generators = polynomial_ring(CC, all_qps)
(qai, pai, qbi, pbi) = (generators[:,i] for i in 1:4)

# Define the alpha and beta vectors
α = (qai + i .* pai) / sqrt(2)
β = (qbi - i .* pbi) / sqrt(2)

# Symplective matrices that represent 50/50 beamsplitters between the bell state modes
_S35 = begin
    Id2 = Matrix{Float64}(I, 2, 2)
    St35 = [
        1/sqrt(2)   0  1/sqrt(2)  0;
        0           1  0          0;
        -1/sqrt(2)  0  1/sqrt(2)  0;
        0           0  0          1;
    ]
    Matrix(BlockDiagonal([Id2, St35, Id2, St35]))
end
_S46 = begin
    Id2 = Matrix{Float64}(I, 2, 2)
    St46 = [
        1  0           0  0;
        0  1/sqrt(2)   0  1/sqrt(2);
        0  0           1  0;
        0  -1/sqrt(2)  0  1/sqrt(2);
    ]
    Matrix(BlockDiagonal([Id2, St46, Id2, St46]))
end

"""
    covariance_matrix(μ::Real)

Construct the 24×24 covariance matrix for a SIGSAG source.

# Parameters
- μ: Mean photon number per mode

# Returns
24×24 `Float64` covariance matrix in qqpp ordering after beamsplitter transforms.
"""
function covariance_matrix(μ::Real)
    # Expand SPDC covariance matrix to 6 modes by adding vacuum modes
    covar_qpqp = zeros(2*mds, 2*mds)
    covar_qpqp[1:8, 1:8] = spdc.covariance_matrix(μ)
    for i in 9:12
        covar_qpqp[i,i] = 1/2
    end
    
    # Reorder qpqp → qqpp and apply beamsplitters
    covar_qqpp = reorder(covar_qpqp) 
    return _S46 * _S35 * covar_qqpp * _S35' * _S46'
end
covariance_matrix(sigsag::SIGSAG) = covariance_matrix(sigsag.mean_photon)

"""
    loss_bsm_matrix_fid(ηᵗ::Real, ηᵈ::Real)

Construct the 24×24 loss matrix for SIGSAG fidelity calculations.

Encodes per-mode loss: signal/detection modes (1, 2) use ηᵈ; measured modes (3–6) use ηᵗ.
Added to the K-matrix before Wick evaluation of Bell-state overlap terms.

# Parameters
- ηᵗ: Outcoupling / transmission efficiency for measured modes, ∈ [0, 1]
- ηᵈ: Signal detection efficiency, ∈ [0, 1]

# Returns
24×24 `ComplexF64` loss matrix for fidelity calculations.
"""
function loss_bsm_matrix_fid(ηᵗ::Real, ηᵈ::Real)
    G = zeros(ComplexF64, 4*mds, 4*mds)
    η = [ηᵈ, ηᵈ, ηᵗ, ηᵗ, ηᵗ, ηᵗ]

    for i in 1:mds
        G[i,     i+2*mds] = (η[i] - 1)
        G[i,     i+3*mds] = -im*(η[i] - 1)
        G[i+mds, i+2*mds] = im*(η[i] - 1)
        G[i+mds, i+3*mds] = (η[i] - 1)
    end

    return (G + transpose(G) + I) / 2
end
loss_bsm_matrix_fid(sigsag::SIGSAG) = loss_bsm_matrix_fid(sigsag.outcoupling_efficiency, sigsag.detection_efficiency)

"""
    loss_bsm_matrix_pgen(ηᵗ::Real, ηᵈ::Real)

Construct the 24×24 loss matrix for SIGSAG probability-of-success calculations.

Similar to `loss_bsm_matrix_fid`, but measured modes (3–6) are traced out. Output modes (1, 2) use ηᵈ.

# Parameters
- ηᵗ: Outcoupling / transmission efficiency, ∈ [0, 1]
- ηᵈ: Signal detection efficiency, ∈ [0, 1]

# Returns
24×24 `ComplexF64` loss matrix for probability-of-success calculations.
"""
function loss_bsm_matrix_pgen(ηᵗ::Real, ηᵈ::Real)
    G = zeros(ComplexF64, 4*mds, 4*mds)
    η = [ηᵈ, ηᵈ, ηᵗ, ηᵗ, ηᵗ, ηᵗ]

    for i in 1:mds
        if i in (3,4,5,6)
            G[i,     i+2*mds] = -1
            G[i,     i+3*mds] = im
            G[i+mds, i+2*mds] = -im
            G[i+mds, i+3*mds] = -1
        else
            G[i,     i+2*mds] = (η[i] - 1)
            G[i,     i+3*mds] = -im*(η[i] - 1)
            G[i+mds, i+2*mds] = im*(η[i] - 1)
            G[i+mds, i+3*mds] = (η[i] - 1)
        end
    end

    return (G + transpose(G) + I) / 2
end
loss_bsm_matrix_pgen(sigsag::SIGSAG) = loss_bsm_matrix_pgen(sigsag.outcoupling_efficiency, sigsag.detection_efficiency)

"""
    moment_vector(n1::Vector{Int}, n2::Vector{Int}, ηᵗ::Real, ηᵈ::Real)

Construct the symbolic moment polynomial for a SIGSAG coincidence measurement.

Builds the Nemo polynomial encoding the joint detection event where signal modes (1, 2)
contribute through ηᵈ and BSM modes (3–6) contribute photon numbers `n1` and `n2` through ηᵗ.

# Parameters
- n1  : Photon-number vector for BSM modes on one side (length 4)
- n2  : Photon-number vector for BSM modes on the other side (length 4)
- ηᵗ  : Outcoupling / transmission efficiency
- ηᵈ  : Signal detection efficiency

# Returns
Nemo multivariate polynomial over `ComplexField`.
"""
function moment_vector(n1::Vector{Int}, n2::Vector{Int}, ηᵗ::Real, ηᵈ::Real)
    Ca12 = ηᵈ * (α[1]*α[2])
    Cb12 = ηᵈ * (β[1]*β[2])
    prod = one(R)
    for i in 3:mds
        prod *= (α[i]*sqrt(ηᵗ))^n1[i-2]/factorial(n1[i-2]) * (β[i]*sqrt(ηᵗ))^n2[i-2]/factorial(n2[i-2])
    end
    return Ca12 * Cb12 * prod
end


"""
    probability_success(μ::Real, ηᵗ::Real, ηᵈ::Real)

Calculate the probability of photon-photon state generation for the SIGSAG source.

# Parameters
- μ  : Mean photon number per mode
- ηᵗ : Outcoupling / transmission efficiency
- ηᵈ : Detection efficiency

# Returns
Real-valued probability of successful photon-photon state generation.
"""
function probability_success(μ::Real, ηᵗ::Real, ηᵈ::Real)
    cov = covariance_matrix(μ)
    A = k_function_matrix(cov) + loss_bsm_matrix_pgen(ηᵗ, ηᵈ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = 1/(D1*D2*D3)

    C = ηᵈ^2 * (α[1]*α[2]) * (β[1]*β[2]) # moment_vector([0,0,0,0], [0,0,0,0], ηᵗ, ηᵈ)

    return real(Coef * W(C, Ainv))
end
probability_success(sigsag::SIGSAG) = probability_success(sigsag.mean_photon, sigsag.outcoupling_efficiency, sigsag.detection_efficiency)

"""
    fidelity(μ::Real, ηᵗ::Real, ηᵈ::Real)

Calculate the Bell-state fidelity of the SIGSAG source under loss.

Computes the Bell-state overlap ⟨Φ|ρ|Φ⟩, where ρ is the normalized photon-photon density matrix following heralding.

# Parameters
- μ  : Mean photon number per mode
- ηᵗ : Outcoupling / transmission efficiency
- ηᵈ : Detection efficiency

# Returns
Real-valued Bell-state fidelity of the SIGSAG source for the given parameters.
"""
function fidelity(μ::Real, ηᵗ::Real, ηᵈ::Real)
    cov = covariance_matrix(μ)
    A = k_function_matrix(cov) + loss_bsm_matrix_fid(ηᵗ, ηᵈ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    # Wick terms (cached)
    Fsum =
        W(moment_vector([1,0,0,1], [1,0,0,1], ηᵗ, ηᵈ), Ainv) +
        W(moment_vector([0,1,1,0], [0,1,1,0], ηᵗ, ηᵈ), Ainv) +
        W(moment_vector([1,0,0,1], [0,1,1,0], ηᵗ, ηᵈ), Ainv) +
        W(moment_vector([0,1,1,0], [1,0,0,1], ηᵗ, ηᵈ), Ainv)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)

    pgen = probability_success(μ, ηᵗ, ηᵈ)

    coef = 1 / (2 * D1 * D2 * D3 * pgen)

    value = coef * Fsum
    if abs(imag(value)) > 1e-10
        @warn "fidelity has nontrivial imaginary part" imag=imag(value) value=value
    end
    return real(value)
end
fidelity(sigsag::SIGSAG) = fidelity(sigsag.mean_photon, sigsag.outcoupling_efficiency, sigsag.detection_efficiency)

end # module