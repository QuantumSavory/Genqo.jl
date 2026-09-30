module tools

using DocStringExtensions

export permutation_matrix, reorder


"""
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

Reorder a covariance matrix from qpqp to qqpp mode ordering.

Applies the permutation `[1, 3, 5, ..., 2, 4, 6, ...]` via a similarity transform so that all
q-quadratures come before all p-quadratures. The legacy `tmsv` and `spdc` modules report their
covariance matrices in qpqp ordering, so this converts them to the qqpp ordering that the
[`Genqo.ProjectedPureGaussianState`](@ref) machinery — and Gabs' `QuadBlockBasis` — expects.

# Parameters
- covariance_matrix: Real covariance matrix in qpqp ordering

# Returns
Reordered covariance matrix in qqpp ordering.
"""
function reorder(covariance_matrix::Matrix{Float64})::Matrix{Float64}
    sz = size(covariance_matrix)[1]
    perm_matrix = permutation_matrix([1:2:sz; 2:2:sz])
    return perm_matrix * covariance_matrix * perm_matrix'
end

# Inverse of `reorder`: qqpp → qpqp. Used by the legacy modules that report their covariance
# matrices in qpqp ordering, since the underlying Gabs states are built in `QuadBlockBasis`.
function _unreorder(covariance_matrix::Matrix{Float64})::Matrix{Float64}
    sz = size(covariance_matrix)[1]
    perm_matrix = permutation_matrix([1:2:sz; 2:2:sz])
    return perm_matrix' * covariance_matrix * perm_matrix
end

end # module
