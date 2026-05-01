module spdc

using BlockDiagonals
using Nemo
using LinearAlgebra

import ..tmsv
using ..tools


"""
    SPDC

Parameters for a Spontaneous Parametric Down-Conversion (SPDC) entanglement source.

An SPDC source is composed of two Two-Mode Squeezed Vacuum (TMSV) states whose idler modes are swapped. This is not to be confused with the fact that there are two Spontaneous Parametric Down-Conversion (SPDC) processes occuring. This is a standard unheralded source of dual-rail entangled photon pairs.


# Fields
- `mean_photon::Real`           : Mean photon number per mode (default `1e-2`)
- `detection_efficiency::Real`  : Detector efficiency, ∈ [0, 1] (default `1.0`)
- `bsm_efficiency::Real`        : Bell-state measurement efficiency, ∈ [0, 1] (default `1.0`)
- `outcoupling_efficiency::Real`: Photon outcoupling / transmission efficiency, ∈ [0, 1] (default `1.0`)
"""
Base.@kwdef mutable struct SPDC
    mean_photon::Real = 1e-2
    detection_efficiency::Real = 1.0
    bsm_efficiency::Real = 1.0
    outcoupling_efficiency::Real = 1.0
end

# Global canonical position and momentum variables
const mds = 4 # Number of modes for our system

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

_perm_matrix_12785634 = permutation_matrix([1,2,7,8,5,6,3,4])

"""
    covariance_matrix(μ::Real)

Construct the 8×8 covariance matrix for an SPDC source.

# Parameters
- μ: Mean photon number per mode

# Returns
8×8 `Float64` covariance matrix in qpqp ordering.
"""
function covariance_matrix(μ::Real)::Matrix{Float64}
    tmsv_covar = tmsv.covariance_matrix(μ)
    covar = Matrix(BlockDiagonal([tmsv_covar, tmsv_covar]))
    covar_qpqp = _perm_matrix_12785634 * covar * _perm_matrix_12785634'
    return covar_qpqp
end
covariance_matrix(spdc::SPDC) = covariance_matrix(spdc.mean_photon)

"""
    loss_bsm_matrix_fid(ηᵗ::Real, ηᵈ::Real)

Construct the 16×16 loss matrix for SPDC fidelity calculations.

Encodes the combined transmission-detection loss η = ηᵗηᵈ for all four signal modes.
The resulting matrix is added to the K-matrix before Wick evaluation of Bell-state overlap terms.

# Parameters
- ηᵗ: Outcoupling / transmission efficiency, ∈ [0, 1]
- ηᵈ: Detection efficiency, ∈ [0, 1]

# Returns
16×16 `ComplexF64` loss matrix for fidelity: `A = k_function_matrix(cov) + loss_bsm_matrix_fid(ηᵗ, ηᵈ)`.
"""
function loss_bsm_matrix_fid(ηᵗ::Real, ηᵈ::Real)::Matrix{ComplexF64}
    G = zeros(ComplexF64, 16, 16)
    η = ηᵗ*ηᵈ

    for i in 1:4
        G[i,     i+2*mds] = (η - 1)
        G[i,     i+3*mds] = -im*(η - 1)
        G[i+mds, i+2*mds] = im*(η - 1)
        G[i+mds, i+3*mds] = (η - 1)
    end

    return (G + transpose(G) + I) / 2
end
loss_bsm_matrix_fid(spdc::SPDC) = loss_bsm_matrix_fid(spdc.outcoupling_efficiency, spdc.detection_efficiency)

"""
Calculating the portion of the A matrix that arises due to incorporating loss, specifically for the trace of the BSM matrix
"""
loss_bsm_matrix_trace::Matrix{ComplexF64} = begin
    G = zeros(ComplexF64, 16, 16)

    for i in 1:4
        G[i,     i+2*mds] = -1
        G[i,     i+3*mds] = im
        G[i+mds, i+2*mds] = -im
        G[i+mds, i+3*mds] = -1
    end

    (G + transpose(G) + I) / 2
end

"""
    dmijZ(dmi::Int, dmj::Int, Ainv::Matrix{ComplexF64}, nvec::Vector{Int}, ηᵗ::Real, ηᵈ::Real)

Calculate a single element of the unnormalized spin-spin density matrix for the SPDC source.

# Parameters
- dmi  : Row index of the density matrix element (1–4, indexing the four Bell states)
- dmj  : Column index of the density matrix element (1–4)
- Ainv : Inverse A-matrix (from `k_function_matrix` + `loss_bsm_matrix_fid`)
- nvec : Photon-number vector `[n₁, n₂, n₃, n₄]` for the four modes
- ηᵗ   : Outcoupling / transmission efficiency
- ηᵈ   : Detection efficiency

# Returns
Complex density matrix element ρ[dmi, dmj] (unnormalized).
"""
function dmijZ(dmi::Int, dmj::Int, Ainv::Matrix{ComplexF64}, nvec::Vector{Int}, ηᵗ::Real, ηᵈ::Real)::ComplexF64
    η = [ηᵗ*ηᵈ, ηᵗ*ηᵈ, ηᵗ*ηᵈ, ηᵗ*ηᵈ]

    # Calculate Ca based on dmi value
    if dmi == 1
        Ca₁ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[3]*sqrt(η[3]) - α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Ca₄ = ((α[3]*sqrt(η[3]) + α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 2
        Ca₁ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[3]*sqrt(η[3]) + α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Ca₄ = ((α[3]*sqrt(η[3]) - α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 3
        Ca₁ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[3]*sqrt(η[3]) - α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Ca₄ = ((α[3]*sqrt(η[3]) + α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 4
        Ca₁ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[3]*sqrt(η[3]) + α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Ca₄ = ((α[3]*sqrt(η[3]) - α[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    else
        Ca = 1
    end

    # Calculate Cb based on dmj value
    if dmj == 1
        Cb₁ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[3]*sqrt(η[3]) - β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Cb₄ = ((β[3]*sqrt(η[3]) + β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 2
        Cb₁ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[3]*sqrt(η[3]) + β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Cb₄ = ((β[3]*sqrt(η[3]) - β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 3
        Cb₁ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[3]*sqrt(η[3]) - β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Cb₄ = ((β[3]*sqrt(η[3]) + β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 4
        Cb₁ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[3]*sqrt(η[3]) + β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[3])
        Cb₄ = ((β[3]*sqrt(η[3]) - β[4]*sqrt(η[4])) * (1/sqrt(2)))^(nvec[4])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    else
        Cb = 1
    end

    C = Ca*Cb

    # Sum over wick partitions
    return W(C, Ainv)
end

"""
    spin_density_matrix(μ::Real, ηᵗ::Real, ηᵈ::Real, nvec::Vector{Int})

Calculate the 4×4 spin-spin density matrix for the SPDC source conditioned on photon-number measurement outcome `nvec` after simulated mode-memory interaction.

# Parameters
- μ    : Mean photon number per mode
- ηᵗ   : Outcoupling / transmission efficiency
- ηᵈ   : Detection efficiency
- nvec : Photon-number vector `[n₁, n₂, n₃, n₄]` for the four modes

# Returns
4×4 `ComplexF64` normalized spin-spin density matrix.
"""
function spin_density_matrix(μ::Real, ηᵗ::Real, ηᵈ::Real, nvec::Vector{Int})::Matrix{ComplexF64}
    lmat = 4
    mat = Matrix{ComplexF64}(undef, lmat, lmat)
    cov = reorder(covariance_matrix(μ))
    A = k_function_matrix(cov) + loss_bsm_matrix_fid(ηᵗ, ηᵈ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = 1/(4*D1*D2*D3)

    for i in 1:lmat
        for j in 1:lmat
            mat[i,j] = dmijZ(i, j, Ainv, nvec, ηᵗ, ηᵈ)
        end
    end

    return Coef * mat
end
spin_density_matrix(spdc::SPDC, nvec::Vector{Int}) = spin_density_matrix(spdc.mean_photon, spdc.outcoupling_efficiency, spdc.detection_efficiency, nvec)

"""
    moment_vector::Dict{Int, Nemo.Generic.MPoly{Nemo.ComplexFieldElem}}

Symbolic moment polynomials used by SPDC.

- Maps an integer key to a Nemo polynomial in the global phase-space variables (`q/p` → α, β).
- Each polynomial represents a specific Gaussian moment needed in SPDC calculations
  (e.g., Bell-overlap terms and normalization/trace-related terms).
- These are evaluated numerically by contracting against `Ainv` via Wick’s theorem:
  - either directly with `tools.W(moment_vector[k], Ainv)`, or
  - more efficiently via `moment_terms[k] = tools.extract_W_terms(moment_vector[k])`.
"""
const moment_vector::Dict{Int, Nemo.Generic.MPoly{Nemo.ComplexFieldElem}} = begin
    Ca1 = α[1] * α[4]
    Ca2 = α[2] * α[3]
    Cb1 = β[1] * β[4]
    Cb2 = β[2] * β[3]

    Dict(
        0 => α[3] * α[4] * β[3] * β[4], 
        1 => Ca1 * Cb1,
        2 => Ca1 * Cb2,
        3 => Ca2 * Cb1,
        4 => Ca2 * Cb2,
    )
end

"""
    moment_terms::Dict{Int, tools.WTerms}

Precompiled Wick terms for SPDC moment polynomials.

- Keys match `moment_vector` (each key corresponds to a specific moment polynomial used in SPDC formulas).
- Values are `tools.WTerms` objects bundling per-degree monomial buckets — see `tools.WBucket`.
- Used by `tools.W(moment_terms[k], Ainv)` for fast Gaussian moment evaluation via Wick pairings.
- Exists to avoid repeated Nemo polynomial parsing during fidelity calculation.
"""
const moment_terms::Dict{Int, tools.WTerms} = Dict(
    k => extract_W_terms(v) for (k, v) in moment_vector
)

"""
    fidelity(μ::Real, ηᵗ::Real, ηᵈ::Real)

Calculate the Bell-state fidelity of the single-mode SPDC source under loss.

This computes the overlap ⟨Φ|ρ|Φ⟩ of the photon-photon state produced by the SPDC source with an ideal Bell state.

# Parameters
- μ  : Mean photon number
- ηᵗ : Outcoupling / transmission efficiency
- ηᵈ : Detection efficiency

# Returns
Real-valued Bell-state fidelity of the SPDC source for the given parameters.
"""
function fidelity(μ::Real, ηᵗ::Real, ηᵈ::Real)::Real
    cov = reorder(covariance_matrix(μ))

    Γ = cov + (1/2) * I
    detΓ = det(Γ)
    K = k_function_matrix(cov)

    # The loss matrix will be unique for calculating the fidelity    
    # A1 (fidelity loss)
    A1 = K + loss_bsm_matrix_fid(ηᵗ, ηᵈ)

    # Factor + invers
    factA1 = lu(A1)
    Ainv1  = inv(factA1)
    D1     = sqrt(det(factA1))  # sqrt(det(A1))

    # Wick terms (cached)
    Fsum =
        W(moment_terms[1], Ainv1) +
        W(moment_terms[2], Ainv1) +
        W(moment_terms[3], Ainv1) +
        W(moment_terms[4], Ainv1)

    N1 = (ηᵗ * ηᵈ)^4

    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)

    coef = N1 / (2 * D1 * D2 * D3)

    value = coef * Fsum
    if abs(imag(value)) > 1e-10
        @warn "fidelity has nontrivial imaginary part" imag=imag(value) value=value
    end
    return real(value)
end

fidelity(spdc::SPDC) = fidelity(spdc.mean_photon, spdc.outcoupling_efficiency, spdc.detection_efficiency)



end # module
