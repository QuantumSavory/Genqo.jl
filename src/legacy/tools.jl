module tools

using LinearAlgebra
using Nemo
using BlockDiagonals

export permutation_matrix, reorder, k_function_matrix


"""
    permutation_matrix(permutations::Vector{Int})

Construct a permutation matrix from a permutation vector.

Returns the n×n matrix `P` where `P[i, permutations[i]] = 1` and all other entries are zero.
Used to reorder rows/columns of a covariance matrix between mode orderings.

# Parameters
- permutations: Integer vector of length n encoding the permutation (1-indexed)

# Returns
n×n `Float64` permutation matrix.
"""
function permutation_matrix(permutations::Vector{Int})::Matrix{Int}
    n = length(permutations)
    P = zeros(Int, n, n)
    for i in 1:n
        P[i, permutations[i]] = 1
    end
    return P
end

"""
    reorder(covariance_matrix)

Reorder a covariance matrix from qpqp to qqpp mode ordering.

Applies the permutation `[1, 3, 5, ..., 2, 4, 6, ...]` via a similarity transform so that all
q-quadratures come before all p-quadratures. Required before calling `k_function_matrix`.

# Parameters
- covariance_matrix: Real covariance matrix (or `BlockDiagonal`) in qpqp ordering

# Returns
Reordered covariance matrix in qqpp ordering.
"""
function reorder(covariance_matrix::Matrix{Float64})::Matrix{Float64}
    sz = size(covariance_matrix)[1]
    perm_matrix = permutation_matrix([1:2:sz; 2:2:sz])
    return perm_matrix * covariance_matrix * perm_matrix'
end

"""
    k_function_matrix(covariance_matrix::Matrix{Float64})

Construct the complex-valued K-matrix used to form the Gaussian contraction matrix `A`.

Given a physical covariance matrix (in qqpp ordering), this function forms Γ = cov + (1/2)I, computes
Γ⁻¹, and then builds the complex block matrix (and its conjugate block) used in the ZALM/SPDC moment
formalism. Downstream code forms `A = K + G`, where `G` encodes loss / measurement modeling, and then
uses `A⁻¹` as the contraction kernel for Wick evaluation via `W`.

Implementation note: This version is an unrolled/optimized construction that avoids intermediate block
arrays and reuses an LU factorization for Γ⁻¹.

# Parameters
- covariance_matrix : Real covariance matrix in qqpp ordering

# Returns
A `ComplexF64` matrix `K` (block diagonal `[BB, conj(BB)]`) suitable for `A = K + loss_matrix`.
"""
function k_function_matrix(covariance::Matrix{Float64})::Matrix{ComplexF64}
    mds = size(covariance, 1) ÷ 2
    Γ = covariance + 0.5*I
    Γinv = inv(Γ)

    # Views of Γinv blocks
    A  = @view Γinv[1:mds,      1:mds     ]
    C  = @view Γinv[1:mds,      mds+1:2mds]
    Cᵀ = @view Γinv[mds+1:2mds, 1:mds     ]
    B  = @view Γinv[mds+1:2mds, mds+1:2mds]

    K = zeros(ComplexF64, 4mds, 4mds)
    @views @. K[1:mds,      1:mds     ] = 0.5*A  + (0.25im)*(C + Cᵀ)
    @views @. K[1:mds,      mds+1:2mds] = 0.5*C  - (0.25im)*(A - B)
    @views @. K[mds+1:2mds, 1:mds     ] = 0.5*Cᵀ - (0.25im)*(A - B)
    @views @. K[mds+1:2mds, mds+1:2mds] = 0.5*B  - (0.25im)*(C + Cᵀ)
    @views @. K[2mds+1:4mds, 2mds+1:4mds] = conj(K[1:2mds, 1:2mds])
    K
end

end # module
