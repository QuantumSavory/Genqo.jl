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
function covariance_matrix(μ::Real)::Matrix{Float64}
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
function loss_bsm_matrix_fid(ηᵗ::Real, ηᵈ::Real)::Matrix{ComplexF64}
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
function loss_bsm_matrix_pgen(ηᵗ::Real, ηᵈ::Real)::Matrix{ComplexF64}
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
    _moment_vector_sym(n1::Vector{Int}, n2::Vector{Int})

Construct the *purely symbolic* moment polynomial for a SIGSAG coincidence measurement.

This is the η-stripped version of the moment polynomial: the full physical moment is
`ηᵈ² · ηᵗ^((sum(n1)+sum(n2))/2) · _moment_vector_sym(n1, n2)`. Stripping the numeric prefactor
lets the symbolic part be cached and Wick-contracted via the precompiled `moment_terms` fast path,
with the prefactor applied as a scalar at call time.

# Parameters
- n1  : Photon-number vector for BSM modes on one side (length 4)
- n2  : Photon-number vector for BSM modes on the other side (length 4)

# Returns
Nemo multivariate polynomial over `ComplexField`.
"""
function _moment_vector_sym(n1::Vector{Int}, n2::Vector{Int})::Nemo.Generic.MPoly{Nemo.ComplexFieldElem}
    Ca12 = α[1]*α[2]
    Cb12 = β[1]*β[2]
    prod = one(R)
    for i in 3:mds
        prod *= α[i]^n1[i-2]/factorial(n1[i-2]) * β[i]^n2[i-2]/factorial(n2[i-2])
    end
    return Ca12 * Cb12 * prod
end

"""
    moment_vector::Dict{Int, Nemo.Generic.MPoly{Nemo.ComplexFieldElem}}

Symbolic moment polynomials used by SIGSAG. Each polynomial is the η-stripped part of a
specific Gaussian moment; the full physical moment recovers a `ηᵈ²·ηᵗ^k` prefactor at call time.

Keys:
- `0` — `α[1]α[2]·β[1]β[2]` — used by `probability_success` (prefactor `ηᵈ²`)
- `1`–`4` — Bell-state overlap moments for the four `(n1, n2)` patterns used by `fidelity`
  (prefactor `ηᵈ²·ηᵗ²`)
"""
const moment_vector::Dict{Int, Nemo.Generic.MPoly{Nemo.ComplexFieldElem}} = Dict(
    0 => _moment_vector_sym([0,0,0,0], [0,0,0,0]),
    1 => _moment_vector_sym([1,0,0,1], [1,0,0,1]),
    2 => _moment_vector_sym([0,1,1,0], [0,1,1,0]),
    3 => _moment_vector_sym([1,0,0,1], [0,1,1,0]),
    4 => _moment_vector_sym([0,1,1,0], [1,0,0,1]),
)

"""
    moment_terms::Dict{Int, tools.WTerms}

Precompiled Wick terms for SIGSAG moment polynomials.

- Keys match `moment_vector`.
- Values are `tools.WTerms` objects bundling per-degree monomial buckets — see `tools.WBucket`.
- Used by `tools.W(moment_terms[k], Ainv)` for fast Gaussian moment evaluation via Wick pairings.
- Exists to avoid repeated Nemo polynomial parsing during fidelity and probability_success.
"""
const moment_terms::Dict{Int, tools.WTerms} = Dict(
    k => extract_W_terms(v) for (k, v) in moment_vector
)


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
function probability_success(μ::Real, ηᵗ::Real, ηᵈ::Real)::Real
    cov = covariance_matrix(μ)
    A = k_function_matrix(cov) + loss_bsm_matrix_pgen(ηᵗ, ηᵈ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = 1/(D1*D2*D3)

    # Full moment is ηᵈ² · α[1]α[2]·β[1]β[2]; symbolic part cached as moment_terms[0].
    return real(Coef * ηᵈ^2 * W(moment_terms[0], Ainv))
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
function fidelity(μ::Real, ηᵗ::Real, ηᵈ::Real)::Real
    cov = covariance_matrix(μ)
    A = k_function_matrix(cov) + loss_bsm_matrix_fid(ηᵗ, ηᵈ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    # Wick terms (cached). Each fidelity moment carries a numeric prefactor ηᵈ²·ηᵗ²
    # (sum(n1)+sum(n2) = 4 ⇒ ηᵗ^(4/2) = ηᵗ²); the symbolic part is moment_terms[1..4].
    Fsum =
        W(moment_terms[1], Ainv) +
        W(moment_terms[2], Ainv) +
        W(moment_terms[3], Ainv) +
        W(moment_terms[4], Ainv)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)

    pgen = probability_success(μ, ηᵗ, ηᵈ)

    coef = ηᵈ^2 * ηᵗ^2 / (2 * D1 * D2 * D3 * pgen)

    value = coef * Fsum
    if abs(imag(value)) > 1e-10
        @warn "fidelity has nontrivial imaginary part" imag=imag(value) value=value
    end
    return real(value)
end
fidelity(sigsag::SIGSAG) = fidelity(sigsag.mean_photon, sigsag.outcoupling_efficiency, sigsag.detection_efficiency)

end # module