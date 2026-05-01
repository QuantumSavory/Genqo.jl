module tools

using LinearAlgebra
using Nemo
using BlockDiagonals

export wick_out, W, extract_W_terms, permutation_matrix, reorder, k_function_matrix


"""
Precompute Wick partitions (perfect pairings) of 1:n
Each partition is a Vector of (i, j) pairs (as Tuples)
"""
function _wick_partitions(N::Int)
    @assert iseven(N) "N must be even"

    n_parts = prod(1:2:(N-1)) # number of perfect matchings of n elements

    result = Array{Int, 3}(undef, n_parts, 2, N÷2) # will hold all partitions
    
    # Recursive helper
    idx = 1
    function backtrack(remaining::Vector{Int}, current::Vector{Tuple{Int,Int}})
        if isempty(remaining)
            # Found a complete pairing
            result[idx, :, :] .= reshape(collect(Iterators.flatten(current)), (2, N÷2))
            idx += 1
            return
        end
        
        # Always take the smallest remaining index to avoid duplicates
        i = remaining[1]
        
        # Try pairing i with each other remaining element
        for k in 2:length(remaining)
            j = remaining[k]
            
            # Build the next "remaining" list without i and j
            next_remaining = [remaining[2:k-1]; remaining[k+1:end]]
            
            push!(current, (i, j))
            backtrack(next_remaining, current)
            pop!(current)
        end
    end
    
    backtrack(collect(1:N), Tuple{Int,Int}[])
    @assert idx == n_parts + 1 "Expected to fill all $n_parts partitions, but filled $(idx-1)"
    return result
end
const wick_partitions = Dict(N => _wick_partitions(N) for N in (0, 2, 4, 6, 8)) # Precompute for N=0,2,4,6,8

"""
    wick_out(coef::ComplexF64, moment_vector::Vector{Int}, Ainv::Matrix{ComplexF64})

Evaluate a single monomial term via Wick's theorem (sum over perfect pairings).

For each Wick partition of the indices in `moment`, multiplies the corresponding
entries of `Ainv` and accumulates the result, then scales by `coef`.

# Parameters
- coef  : Complex coefficient of the monomial
- moment: Variable indices appearing in the monomial (length must be even)
- Ainv  : Inverse A-matrix providing the two-point contractions

# Returns
`coef` times the sum of all Wick-contraction products for this monomial.
"""
function wick_out(coef::ComplexF64, moment::AbstractVector{Int}, Ainv::Matrix{ComplexF64})
    # Iterate over Wick partitions
    s = zero(ComplexF64)
    parts = wick_partitions[length(moment)]
    n_parts = size(parts, 1); n_pairs = size(parts, 3)
    @inbounds for m in 1:n_parts
        f = one(ComplexF64)
        for n in 1:n_pairs
            i = parts[m, 1, n]; j = parts[m, 2, n]
            f *= Ainv[moment[i], moment[j]]
        end
        s += f
    end
    return s * coef
end

"""
    wick_out(coef, m, parts, n_pairs, Ainv)

Variant that reads monomial indices from a stack-resident `NTuple{8,Int}` `m` (zero-padded for
shorter monomials). Avoids heap allocation entirely and lets the compiler keep `m` in registers.
`n_pairs` selects how many index pairs of each partition to consume (4 for degree-8, 2 for degree-4).
"""
function wick_out(coef::ComplexF64, m::NTuple{8,Int}, parts::Array{Int,3}, n_pairs::Int, Ainv::Matrix{ComplexF64})
    s = zero(ComplexF64)
    n_parts = size(parts, 1)
    @inbounds for k in 1:n_parts
        f = one(ComplexF64)
        for n in 1:n_pairs
            i = parts[k, 1, n]; j = parts[k, 2, n]
            f *= Ainv[m[i], m[j]]
        end
        s += f
    end
    return s * coef
end

"""
    extract_W_terms(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem})

Precompile a Nemo polynomial into a reusable list of contraction terms for `W`.

The ZALM/SPDC code builds certain moment polynomials symbolically (in the `q/p` variables) and then
evaluates them many times numerically. This function converts the polynomial into a vector of
`(coef, idxs)` where:
- `coef` is the complex coefficient of a monomial, and
- `idxs` are the variable indices that appear with exponent 1 in that monomial.

This avoids repeated Nemo monomial parsing and exponent scanning during hot loops.

Assumption (matched to current usage): moment polynomials are multilinear in the variables used for Wick
evaluation (exponents are 0/1 for the variables of interest).

# Parameters
- C : Nemo multivariate polynomial over `ComplexField`

# Returns
`Vector{Tuple{ComplexF64, Vector{Int}}}` suitable for `W(terms, Ainv)`.
"""
function extract_W_terms(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem})::Tuple{Vector{ComplexF64}, Matrix{Int}}
    coeffs = coefficients(C)
    mons = monomials(C)
    n_vars = nvars(parent(C))
    c = Vector{ComplexF64}(undef, length(coeffs))
    ξ = zeros(Int, 8, length(coeffs)) # each column will hold the variable indices for one monomial

    x = 1
    for (mon, coeff) in zip(mons, coeffs)
        c[x] = ComplexF64(coeff)
        y = 1
        for i in 1:n_vars
            if exponent(mon, 1, i) == 1
                ξ[y, x] = i
                y += 1
            end
        end
        x += 1
    end
    return c, ξ
end

"""
    W(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem}, Ainv::Matrix{ComplexF64})

Evaluate a (symbolic) moment polynomial by Wick contraction against `Ainv`.

This expands `C` into monomials and, for each monomial, collects the indices of variables present
(exponent 1), then calls `wick_out` to perform the sum over pairings. This is the general (slower) path
used when terms have not been precompiled.

# Parameters
- C    : Moment polynomial to evaluate (symbolic)
- Ainv : Inverse A-matrix providing the contraction kernel

# Returns
Complex value of the Gaussian moment implied by `C` and `Ainv`.
"""
function W(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem}, Ainv::Matrix{ComplexF64})
    elm = zero(ComplexF64)
    n_vars = nvars(parent(C))
    for (mon, coeff) in zip(monomials(C), coefficients(C))
        elm += wick_out(ComplexF64(coeff), [i for i in 1:n_vars if exponent(mon, 1, i) == 1], Ainv)
    end
    return elm
end

"""
    W(terms::Vector{Tuple{ComplexF64, Vector{Int}}}, Ainv::Matrix{ComplexF64})

Fast Wick evaluator for precompiled moment terms.

This is the hot-path used by SPDC/ZALM fidelity and probability calculations. It skips Nemo entirely:
each `(coef, idxs)` term represents one monomial, and `wick_out` performs the contraction.

# Parameters
- c   : Vector of complex coefficients
- ξ   : Array of variable indices (each column represents a monomial)
- Ainv: Inverse A-matrix providing the contraction kernel

# Returns
Complex value of the contracted moment.
"""
function W(cξ::Tuple{Vector{ComplexF64}, Matrix{Int}}, Ainv::Matrix{ComplexF64})
    elm = zero(ComplexF64)
    c, ξ = cξ
    parts4 = wick_partitions[4]
    parts8 = wick_partitions[8]
    @inbounds for x in eachindex(c)
        m = (ξ[1,x], ξ[2,x], ξ[3,x], ξ[4,x], ξ[5,x], ξ[6,x], ξ[7,x], ξ[8,x])
        if m[5] == 0
            elm += wick_out(c[x], m, parts4, 2, Ainv)
        else
            elm += wick_out(c[x], m, parts8, 4, Ainv)
        end
    end
    return elm
end

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
function permutation_matrix(permutations::Vector{Int})
    n = length(permutations)
    P = zeros(n, n)
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
function reorder(covariance_matrix::Union{Matrix{Float64}, BlockDiagonal{Float64, Matrix{Float64}}})
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
function k_function_matrix(covariance::Matrix{Float64})
    Γ = covariance + (1/2)*I

    # Invert Γ via LU (same numerical result as inv(Γ), but lets us reuse LU storage)
    F = lu!(Γ)            # factors in-place
    Γinv = inv(F)         # 16×16 Float64

    sz = size(Γinv, 1) ÷ 2
    n = 2sz

    # Views of Γinv blocks (Float64)
    A  = @view Γinv[1:sz,    1:sz]
    C  = @view Γinv[1:sz,    sz+1:n]
    Cᵀ = @view Γinv[sz+1:n,  1:sz]
    B  = @view Γinv[sz+1:n,  sz+1:n]

    # Build BB (16×16 ComplexF64) without intermediates
    BB = Matrix{ComplexF64}(undef, n, n)

    @inbounds for j in 1:sz, i in 1:sz
        a  = A[i,j]
        b  = B[i,j]
        c  = C[i,j]
        ct = Cᵀ[i,j]

        # handy subexpressions
        csum = c + ct
        abd  = a - b

        # BB block entries (each multiplied by 1/2 overall)
        BB[i,     j     ] = 0.5*a  + (im/4)*csum  # (1/2)*(A  + (i/2)(C+Ct))
        BB[i,     j+sz  ] = 0.5*c  - (im/4)*abd   # (1/2)*(C  - (i/2)(A-B))
        BB[i+sz,  j     ] = 0.5*ct - (im/4)*abd   # (1/2)*(Ct - (i/2)(A-B))
        BB[i+sz,  j+sz  ] = 0.5*b  - (im/4)*csum  # (1/2)*(B  - (i/2)(C+Ct))
    end

    # Return block diagonal [BB, conj(BB)] as a plain 32×32 matrix
    K = zeros(ComplexF64, 2n, 2n)
    @inbounds for j in 1:n, i in 1:n
        v = BB[i,j]
        K[i,   j  ] = v
        K[i+n, j+n] = conj(v)
    end

    return K
end

end # module
