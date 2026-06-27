module zalm

using BlockDiagonals
using Nemo
using LinearAlgebra

import ..spdc
using ..tools


"""
    ZALM

Parameters for a Zero-Added-Loss Multiplexing (ZALM) cascaded entanglement source.

The ZALM architecture uses two SPDC sources, interfering half of the modes from each source on a pair of 50/50 beamsplitters to perform a Bell-state measurement (BSM). A heralding click pattern signifies a probabilistic photon-photon Bell state between the output modes. Dark counts and all three efficiency channels can be modeled.

# Fields
- `mean_photon::Real`           : Mean photon number per mode per SPDC source (default `1e-2`)
- `detection_efficiency::Real`  : Signal detector efficiency, ∈ [0, 1] (default `1.0`)
- `bsm_efficiency::Real`        : BSM detector efficiency, ∈ [0, 1] (default `1.0`)
- `outcoupling_efficiency::Real`: Photon outcoupling / transmission efficiency, ∈ [0, 1] (default `1.0`)
- `dark_counts::Real`           : Dark-count probability per BSM detector gate, ≥ 0 (default `0.0`)
"""
Base.@kwdef mutable struct ZALM
    mean_photon::Real = 1e-2
    #schmidt_coeffs::Vector{Float64}
    detection_efficiency::Real = 1.0
    bsm_efficiency::Real = 1.0
    outcoupling_efficiency::Real = 1.0
    dark_counts::Real = 0.0
    #visibility::Real = 1.0
end

# Global canonical position and momentum variables
const mds = 8 # Number of modes for our system

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
    Matrix(BlockDiagonal([Id2, St35, Id2, Id2, St35, Id2]))
end
_S46 = begin
    Id2 = Matrix{Float64}(I, 2, 2)
    St46 = [
        1  0           0  0;
        0  1/sqrt(2)   0  1/sqrt(2);
        0  0           1  0;
        0  -1/sqrt(2)  0  1/sqrt(2);
    ]
    Matrix(BlockDiagonal([Id2, St46, Id2, Id2, St46, Id2]))
end

"""
    covariance_matrix(μ::Real)

Construct the 32×32 pre-heralding covariance matrix for a ZALM source.

# Parameters
- μ: Mean photon number per mode

# Returns
32×32 `Float64` covariance matrix in qqpp ordering after beamsplitter transforms.
"""
function covariance_matrix(μ::Real)::Matrix{Float64}
    # Initial ZALM covariance matrix in qpqp ordering
    spdc_covar = spdc.covariance_matrix(μ)
    covar_qpqp = Matrix(BlockDiagonal([spdc_covar, spdc_covar]))

    # Reorder qpqp → qqpp and apply beamsplitters
    covar_qqpp = reorder(covar_qpqp) 
    return _S46 * _S35 * covar_qqpp * _S35' * _S46'
end
covariance_matrix(zalm::ZALM) = covariance_matrix(zalm.mean_photon)

"""
    loss_bsm_matrix_fid(ηᵗ::Real, ηᵈ::Real, ηᵇ::Real)

Construct the 32×32 loss matrix for ZALM fidelity calculations.

Encodes per-mode loss: signal modes (1, 2, 7, 8) use η = ηᵗηᵈ; BSM modes (3–6) use ηᵇ.
Added to the K-matrix before Wick evaluation of Bell-state overlap terms.

# Parameters
- ηᵗ: Outcoupling / transmission efficiency, ∈ [0, 1]
- ηᵈ: Signal detection efficiency, ∈ [0, 1]
- ηᵇ: BSM detector efficiency, ∈ [0, 1]

# Returns
32×32 `ComplexF64` loss matrix for fidelity calculations.
"""
function loss_bsm_matrix_fid(ηᵗ::Real, ηᵈ::Real, ηᵇ::Real)::Matrix{ComplexF64}
    G = zeros(ComplexF64, 4*mds, 4*mds)
    η = [ηᵗ*ηᵈ, ηᵗ*ηᵈ, ηᵇ, ηᵇ, ηᵇ, ηᵇ, ηᵗ*ηᵈ, ηᵗ*ηᵈ]

    for i in 1:mds
        G[i,       i+2*mds] = (η[i] - 1)
        G[i,       i+3*mds] = -im*(η[i] - 1)
        G[i+2*mds, i+mds  ] = im*(η[i] - 1)
        G[i+3*mds, i+mds  ] = (η[i] - 1)
    end

    return (G + transpose(G) + I) / 2
end
loss_bsm_matrix_fid(zalm::ZALM) = loss_bsm_matrix_fid(zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency)

"""
    loss_bsm_matrix_pgen(ηᵗ::Real, ηᵈ::Real, ηᵇ::Real)

Construct the 32×32 loss matrix for ZALM probability-of-success calculations.

Similar to `loss_bsm_matrix_fid`, but the signal modes (1, 2, 7, 8) are traced out. BSM modes (3–6) use ηᵇ.

# Parameters
- ηᵗ: Outcoupling / transmission efficiency, ∈ [0, 1]
- ηᵈ: Signal detection efficiency, ∈ [0, 1]
- ηᵇ: BSM detector efficiency, ∈ [0, 1]

# Returns
32×32 `ComplexF64` loss matrix for probability-of-success calculations.
"""
function loss_bsm_matrix_pgen(ηᵗ::Real, ηᵈ::Real, ηᵇ::Real)::Matrix{ComplexF64}
    G = zeros(ComplexF64, 4*mds, 4*mds)
    η = [ηᵗ*ηᵈ, ηᵗ*ηᵈ, ηᵇ, ηᵇ, ηᵇ, ηᵇ, ηᵗ*ηᵈ, ηᵗ*ηᵈ]

    for i in 1:mds
        if i in (1,2,7,8)
            G[i,       i+2*mds] = -1
            G[i,       i+3*mds] = im
            G[i+2*mds, i+mds  ] = -im
            G[i+3*mds, i+mds  ] = -1
        else
            G[i,       i+2*mds] = (η[i] - 1)
            G[i,       i+3*mds] = -im*(η[i] - 1)
            G[i+2*mds, i+mds  ] = im*(η[i] - 1)
            G[i+3*mds, i+mds  ] = (η[i] - 1)
        end
    end

    return (G + transpose(G) + I) / 2
end
loss_bsm_matrix_pgen(zalm::ZALM) = loss_bsm_matrix_pgen(zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency)

"""
    dmijZ(dmi::Int, dmj::Int, Ainv::Matrix{ComplexF64}, nvec::Vector{Int}, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real)

Calculate a single element of the unnormalized spin-spin density matrix.

# Parameters
- dmi : Row number for the corresponding density matrix element
- dmj : Column number for the corresponding density matrix element
- Ainv : Numerical inverse of the A matrix
- nvec : The vector of nᵢ's for the system, where nᵢ is the number of photons in mode i
- ηᵗ : Transmission efficiency
- ηᵈ : Detection efficiency
- ηᵇ : Bell state measurement efficiency

# Returns
Density matrix element for the ZALM source
"""
function dmijZ(dmi::Int, dmj::Int, Ainv::Matrix{ComplexF64}, nvec::Vector{Int}, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real)::ComplexF64
    η = [ηᵗ*ηᵈ, ηᵗ*ηᵈ, ηᵇ, ηᵇ, ηᵇ, ηᵇ, ηᵗ*ηᵈ, ηᵗ*ηᵈ]

    # Calculate Ca based on dmi value
    if dmi == 1
        Ca₁ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[7]*sqrt(η[7]) - α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Ca₄ = ((α[7]*sqrt(η[7]) + α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 2
        Ca₁ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[7]*sqrt(η[7]) + α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Ca₄ = ((α[7]*sqrt(η[7]) - α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 3
        Ca₁ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[7]*sqrt(η[7]) - α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Ca₄ = ((α[7]*sqrt(η[7]) + α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    elseif dmi == 4
        Ca₁ = ((α[1]*sqrt(η[1]) + α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Ca₂ = ((α[1]*sqrt(η[1]) - α[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Ca₃ = ((α[7]*sqrt(η[7]) + α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Ca₄ = ((α[7]*sqrt(η[7]) - α[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Ca = Ca₁*Ca₂*Ca₃*Ca₄
    else
        Ca = 1
    end

    # Calculate Cb based on dmj value
    if dmj == 1
        Cb₁ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[7]*sqrt(η[7]) - β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Cb₄ = ((β[7]*sqrt(η[7]) + β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 2
        Cb₁ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[7]*sqrt(η[7]) + β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Cb₄ = ((β[7]*sqrt(η[7]) - β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 3
        Cb₁ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[7]*sqrt(η[7]) - β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Cb₄ = ((β[7]*sqrt(η[7]) + β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    elseif dmj == 4
        Cb₁ = ((β[1]*sqrt(η[1]) + β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[1])
        Cb₂ = ((β[1]*sqrt(η[1]) - β[2]*sqrt(η[2])) * (1/sqrt(2)))^(nvec[2])
        Cb₃ = ((β[7]*sqrt(η[7]) + β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[7])
        Cb₄ = ((β[7]*sqrt(η[7]) - β[8]*sqrt(η[8])) * (1/sqrt(2)))^(nvec[8])
        Cb = Cb₁*Cb₂*Cb₃*Cb₄
    else
        Cb = 1
    end

    # Calculate Cd terms
    Cd₃ = (α[3]*β[3]*η[3])^(nvec[3])/factorial(nvec[3])
    Cd₄ = (α[4]*β[4]*η[4])^(nvec[4])/factorial(nvec[4])
    Cd₅ = (α[5]*β[5]*η[5])^(nvec[5])/factorial(nvec[5])
    Cd₆ = (α[6]*β[6]*η[6])^(nvec[6])/factorial(nvec[6])
    C = Cd₃*Cd₄*Cd₅*Cd₆*Ca*Cb

    # Sum over wick partitions (compile polynomial into fast terms)
    return W(C, Ainv)
end

"""
    spin_density_matrix(μ::Real, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real, nvec::Vector{Int})

Calculate the density operator of the single-mode ZALM source on the spin-spin state.

# Parameters
- μ : Mean photon number
- ηᵗ : Outcoupling efficiency
- ηᵈ : Detection efficiency
- ηᵇ : Bell state measurement efficiency
- nvec : The vector of nᵢ's for the system, where nᵢ is the number of photons in mode i

# Returns
Numerical complete spin density matrix
"""
function spin_density_matrix(μ::Real, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real, nvec::Vector{Int})::Matrix{ComplexF64}
    lmat = 4
    mat = Matrix{ComplexF64}(undef, lmat, lmat)
    cov = covariance_matrix(μ)
    A = k_function_matrix(cov) + loss_bsm_matrix_fid(ηᵗ, ηᵈ, ηᵇ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = 1/(D1*D2*D3)

    for i in 1:lmat
        for j in 1:lmat
            mat[i,j] = dmijZ(i, j, Ainv, nvec, ηᵗ, ηᵈ, ηᵇ)
        end
    end

    return Coef * mat
end
spin_density_matrix(zalm::ZALM, nvec::Vector{Int}) = spin_density_matrix(zalm.mean_photon, zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency, nvec)

"""
    moment_vector::NamedTuple

Symbolic moment polynomials used by ZALM, as a `NamedTuple` of Nemo polynomials in the global
phase-space variables (`q/p` → α, β). Each entry is a specific Gaussian moment needed in ZALM
calculations: Bell-overlap, normalization, trace, and click/dark-count terms.

Fields:
- `bell_aa`, `bell_ab`, `bell_ba`, `bell_bb` — fidelity Bell-overlap moments
- `norm_aa`, `norm_ab`, `norm_ba`, `norm_bb` — normalization moments
- `trc` — trace moment used by both `fidelity` and `probability_success`
- `click_a`, `click_b` — single-detector click moments (dark-count contributions)
- `pair_a` — paired α[1]²β[1]² moment
- `vac` — vacuum (constant 1) moment

These are evaluated numerically by contracting against `Ainv` via Wick's theorem, either directly
with `tools.W(moment_vector.<name>, Ainv)` (slow path) or, more efficiently, through the
precompiled `moment_terms.<name>` cache.
"""
const moment_vector = let
    Ca₁ = α[1]*α[3]*α[4]*α[8]
    Ca₂ = α[2]*α[3]*α[4]*α[7]
    Cb₁ = β[1]*β[3]*β[4]*β[8]
    Cb₂ = β[2]*β[3]*β[4]*β[7]

    # For calculating the normalization constant
    Ca₃ = α[1]*α[3]*α[4]*α[7]
    Ca₄ = α[2]*α[3]*α[4]*α[8]
    Cb₃ = β[1]*β[3]*β[4]*β[7]
    Cb₄ = β[2]*β[3]*β[4]*β[8]

    (
        bell_aa = Ca₁*Cb₁,
        bell_ab = Ca₁*Cb₂,
        bell_ba = Ca₂*Cb₁,
        bell_bb = Ca₂*Cb₂,
        norm_aa = Ca₃*Cb₃,
        norm_ab = Ca₃*Cb₄,
        norm_ba = Ca₄*Cb₃,
        norm_bb = Ca₄*Cb₄,
        trc     = α[3]*α[4]*β[3]*β[4],
        click_a = α[3]*β[3],
        click_b = α[4]*β[4],
        pair_a  = α[1]*α[1]*β[1]*β[1],
        vac     = one(R),
    )
end

"""
    moment_terms::NamedTuple

Precompiled Wick terms for ZALM moment polynomials, mirroring the field names of `moment_vector`.

Each field is a concrete `tools.WTerms{<:Tuple}` whose type is fixed at module load — so call
sites like `tools.W(moment_terms.bell_aa, Ainv)` resolve to a fully type-stable specialized
method, with no runtime dispatch.
"""
const moment_terms = map(extract_W_terms, moment_vector)

"""
    probability_success(μ::Real, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real, dark_counts::Real)

Calculate the probability of photon-photon state generation with the given parameters.

# Parameters
- μ : Mean photon number
- ηᵗ : Outcoupling efficiency
- ηᵈ : Detection efficiency
- ηᵇ : Bell state measurement efficiency
- dark_counts : Probability of click with no photon present

# Returns
Probability of successful photon-photon state generation
"""
function probability_success(μ::Real, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real, dark_counts::Real)::Real
    cov = covariance_matrix(μ)
    A = k_function_matrix(cov) + loss_bsm_matrix_pgen(ηᵗ, ηᵈ, ηᵇ)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = 1/(D1*D2*D3)

    return real(Coef * (
        ηᵇ^2 * (1-dark_counts)^4 * W(moment_terms.trc, Ainv) +
        ηᵇ * dark_counts * (1-dark_counts)^3 * W(moment_terms.click_a, Ainv) +
        ηᵇ * dark_counts * (1-dark_counts)^3 * W(moment_terms.click_b, Ainv) +
        dark_counts^2 * (1-dark_counts)^2 * W(moment_terms.vac, Ainv)
    ))
end
probability_success(zalm::ZALM) = probability_success(zalm.mean_photon, zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency, zalm.dark_counts)

"""
    fidelity(μ::Real, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real)

Calculate the Bell-state fidelity of the single-mode ZALM source under loss.

This computes the overlap ⟨Φ|ρ|Φ⟩ of the post-heralding photon-photon state produced by the ZALM source with an ideal Bell state. The evaluation is performed via Gaussian moment (Wick) contractions using the ZALM covariance matrix, with efficiencies applied through transmission/outcoupling (ηᵗ), detection (ηᵈ), and Bell-state-measurement (ηᵇ) models.

# Parameters
- μ  : Mean photon number
- ηᵗ : Outcoupling / transmission efficiency
- ηᵈ : Detection efficiency
- ηᵇ : Bell-state measurement efficiency

# Returns
Real-valued Bell-state fidelity of the ZALM source for the given parameters.
"""
function fidelity(μ::Real, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real)::Real
 # Calculate the fidelity with respect to the Bell state for the photon-photon single-mode ZALM source

    cov = covariance_matrix(μ)
    K = k_function_matrix(cov)

    # The loss matrix will be unique for calculating the fidelity    
    # A1 (fidelity loss)
    A1 = K + loss_bsm_matrix_fid(ηᵗ, ηᵈ, ηᵇ)

    factoredA1 = lu(A1) # factorizes for reuse
    Ainv1 = inv(factoredA1)
    D1 = sqrt(det(factoredA1)) # reuses factorization

    # Wick terms (cached)
    Fsum =
        W(moment_terms.bell_aa, Ainv1) +
        W(moment_terms.bell_ab, Ainv1) +
        W(moment_terms.bell_ba, Ainv1) +
        W(moment_terms.bell_bb, Ainv1)

    # --- A2 (trace / generation normalization loss) ---
    A2 = K + loss_bsm_matrix_pgen(ηᵗ, ηᵈ, ηᵇ)
    factoredA2 = lu(A2) # factorizes for reuse
    Ainv2 = inv(factoredA2)
    N2 = sqrt(det(factoredA2)) # reuses factorization

    Trc = W(moment_terms.trc, Ainv2)

    N1 = (ηᵈ * ηᵗ) ^ 2
    coef = N1 * N2 / (2 * D1)

    value = coef * Fsum / Trc 

    if abs(imag(value)) > 1e-10
        @warn "fidelity has nontrivial imaginary part" imag=imag(value) value=value
    end
    return real(value)
end
fidelity(zalm::ZALM) = fidelity(zalm.mean_photon, zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency)

end # module
