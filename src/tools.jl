module tools

using LinearAlgebra
using Nemo
using BlockDiagonals

export wick_out, W, extract_W_terms, permutation_matrix, reorder, k_function_matrix


"""
Precompute Wick partitions (perfect pairings) of 1:n
Each partition is a Vector of (i, j) pairs (as Tuples)
"""
function _wick_partitions(N::Int)::Array{Int, 3}
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
function wick_out(coef::ComplexF64, moment::AbstractVector{Int}, Ainv::Matrix{ComplexF64})::ComplexF64
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
    WBucket{N}

A homogeneous group of monomials that all have the same degree `N`. Stored as struct-of-arrays for
cache locality, with indices as `NTuple{N,Int}` so the inner Wick loop reads them from registers.
The `parts::Array{Int,3}` table is the same `wick_partitions[N]` cache, copied here so the kernel
needs no Dict lookup at call time.
"""
struct WBucket{N}
    coeffs::Vector{ComplexF64}
    indices::Vector{NTuple{N,Int}}
    parts::Array{Int,3}
end

"""
    WTerms{Buckets<:Tuple}

A precompiled moment polynomial as a heterogeneous tuple of `WBucket`s, one per degree present in
the polynomial. The tuple type carries each bucket's `N` at compile time so iteration unrolls and
each `_W_bucket` call specializes on its bucket's degree.
"""
struct WTerms{Buckets<:Tuple}
    buckets::Buckets
end

"""
    extract_W_terms(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem})

Precompile a Nemo polynomial into a `WTerms` object suitable for the fast `W(::WTerms, Ainv)` path.

Walks `C`'s monomials, groups by degree (count of variables with exponent 1), and emits one
`WBucket{N}` per present degree. Buckets are sorted by descending size so the largest one runs first.

Performance is unimportant — this runs once at module load.

Assumption (matched to current usage): moment polynomials are multilinear in the variables used for
Wick evaluation (exponents are 0/1 for the variables of interest).

# Parameters
- C : Nemo multivariate polynomial over `ComplexField`

# Returns
A `WTerms{<:Tuple}` whose buckets cover every degree present in `C`.
"""
function extract_W_terms(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem})::WTerms
    n_vars = nvars(parent(C))

    by_deg = Dict{Int, Tuple{Vector{ComplexF64}, Vector{Vector{Int}}}}()
    for (mon, coeff) in zip(monomials(C), coefficients(C))
        idxs = Int[]
        for i in 1:n_vars
            if exponent(mon, 1, i) == 1
                push!(idxs, i)
            end
        end
        N = length(idxs)
        haskey(wick_partitions, N) ||
            error("extract_W_terms: monomial of degree $N has no precomputed wick partitions (have keys $(sort(collect(keys(wick_partitions)))))")
        cv, iv = get!(by_deg, N) do
            (ComplexF64[], Vector{Int}[])
        end
        push!(cv, ComplexF64(coeff))
        push!(iv, idxs)
    end

    # Sort by descending bucket size so the dominant bucket runs first.
    degs_sorted = sort!(collect(keys(by_deg)); by = N -> -length(by_deg[N][1]))
    buckets = ((_make_bucket(N, by_deg[N]...) for N in degs_sorted)...,)
    return WTerms(buckets)
end

# Build a concrete `WBucket{N}` with N as a value-type-dispatched constant. Calling `WBucket{N}(...)`
# directly with N as a runtime Int would also work but this keeps the construction call site clean.
@inline function _make_bucket(N::Int, coeffs::Vector{ComplexF64}, idxs_list::Vector{Vector{Int}})
    return _make_bucket_typed(Val(N), coeffs, idxs_list)
end
@inline function _make_bucket_typed(::Val{N}, coeffs::Vector{ComplexF64}, idxs_list::Vector{Vector{Int}}) where {N}
    indices = [NTuple{N,Int}(idxs) for idxs in idxs_list]
    return WBucket{N}(coeffs, indices, wick_partitions[N])
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
function W(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem}, Ainv::Matrix{ComplexF64})::ComplexF64
    elm = zero(ComplexF64)
    n_vars = nvars(parent(C))
    for (mon, coeff) in zip(monomials(C), coefficients(C))
        elm += wick_out(ComplexF64(coeff), [i for i in 1:n_vars if exponent(mon, 1, i) == 1], Ainv)
    end
    return elm
end

"""
    W(t::WTerms, Ainv::Matrix{ComplexF64})

Fast Wick evaluator for precompiled moment terms.

This is the hot path used by SPDC/ZALM fidelity and probability calculations. Each `WBucket{N}` in
`t.buckets` carries its own monomial coefficients, indices (as `NTuple{N,Int}` per monomial), and
the precomputed Wick partition table. The bucket loop is unrolled by the compiler because
`Buckets<:Tuple` carries every bucket's type.

# Returns
Complex value of the contracted moment.
"""
W(t::WTerms, Ainv::Matrix{ComplexF64}) = _sum_buckets(t.buckets, Ainv)

# Recursive helper so a heterogeneous Tuple iterates type-stably (a plain `for` would infer the
# element type as the abstract join of the bucket types and dynamic-dispatch each call).
@inline _sum_buckets(::Tuple{}, ::Matrix{ComplexF64}) = zero(ComplexF64)
@inline _sum_buckets(bs::Tuple, Ainv::Matrix{ComplexF64}) =
    _W_bucket(first(bs), Ainv) + _sum_buckets(Base.tail(bs), Ainv)

@inline function _W_bucket(b::WBucket{N}, Ainv::Matrix{ComplexF64})::ComplexF64 where {N}
    parts = b.parts
    n_parts = size(parts, 1)
    n_pairs = N ÷ 2
    s = zero(ComplexF64)
    @inbounds for x in eachindex(b.coeffs)
        m = b.indices[x]                    # NTuple{N,Int}, stack-resident
        f = zero(ComplexF64)
        for k in 1:n_parts
            t = one(ComplexF64)
            for n in 1:n_pairs
                i = parts[k, 1, n]; j = parts[k, 2, n]
                t *= Ainv[m[i], m[j]]
            end
            f += t
        end
        s += b.coeffs[x] * f
    end
    return s
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
