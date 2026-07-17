using LinearAlgebra
using Nemo
using BlockDiagonals

export wick_out, W, WTerms, extract_W_terms


"""
Precompute Wick partitions (perfect pairings) of 1:N.
Stored as a (N-1)!! × 2 × (N/2) array of Int8, where each row is a perfect matching of the indices 1:N.
"""
function _wick_partitions(N::Int8)::Array{Int8,3}
    iseven(N) || throw(ArgumentError("N must be even"))

    n_parts = prod(1:2:(N-1)) # number of perfect matchings of n elements

    result = Array{Int8,3}(undef, n_parts, 2, N÷2) # will hold all partitions
    
    # Recursive helper
    idx = 1
    function backtrack(remaining::Vector{Int8}, current::Vector{Tuple{Int8,Int8}})
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
    
    backtrack(collect(Int8, 1:N), Tuple{Int8,Int8}[])
    @assert idx == n_parts + 1 "Expected to fill all $n_parts partitions, but filled $(idx-1)"
    return result
end
const _wick_partitions_cache = Ref(Dict{Int8, Array{Int8,3}}())
wick_partitions(N::Int8)::Array{Int8,3} = get!(() -> _wick_partitions(N), _wick_partitions_cache[], N)
wick_partitions(N::Int)::Array{Int8,3} = wick_partitions(Int8(N)) # dispatch to Int8 version for caching


"""
    WBucket{N}

A homogeneous group of monomials that all have the same degree `N`. Stored as struct-of-arrays for
cache locality, with indices as `NTuple{N,Int}` so the inner Wick loop reads them from registers.
"""
struct WBucket{N}
    coeffs::Vector{ComplexF64}
    indices::Vector{NTuple{N,Int}}
end
function Base.show(io::IO, b::WBucket{N}) where {N}
    print(io, "$N-factor Wick bucket with $(length(b.indices)) index sets")
end
Base.:*(a::C, b::WBucket{N}) where {C<:Number,N} = WBucket{N}(a * b.coeffs, b.indices)
Base.:*(a::WBucket{N}, b::C) where {C<:Number,N} = WBucket{N}(b * a.coeffs, a.indices)

"""
    WTerms{Buckets<:Tuple}

A precompiled moment polynomial as a heterogeneous tuple of `WBucket`s, one per degree present in
the polynomial. The tuple type carries each bucket's `N` at compile time so iteration unrolls and
each `_W_bucket` call specializes on its bucket's degree.
"""
struct WTerms{B<:Tuple}
    buckets::B
end
Base.zero(::Type{WTerms{B}}) where {B<:Tuple} = WTerms(B())
Base.:+(a::WTerms, b::WTerms) = WTerms((a.buckets..., b.buckets...)) # concatenate buckets
Base.:*(a::C, b::WTerms) where {C<:Number} = WTerms(b.buckets .* a)
Base.:*(a::WTerms, b::C) where {C<:Number} = WTerms(a.buckets .* b)

"""
    extract_W_terms(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem})

Precompile a Nemo polynomial into a `WTerms` object suitable for the fast `W(::WTerms, Ainv)` path.

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
    return WBucket{N}(coeffs, indices)
end

"""
    W(t::WTerms, Ainv::Matrix{ComplexF64})

Fast Wick evaluator for precompiled moment terms.

# Returns
Complex value of the contracted moment.
"""
W(t::WTerms, Ainv::Matrix{ComplexF64}) = _sum_buckets(t.buckets, Ainv)

# Recursive helper so a heterogeneous Tuple iterates type-stably (a plain `for` would infer the
# element type as the abstract join of the bucket types and dynamic-dispatch each call).
@inline _sum_buckets(::Tuple{}, ::Matrix{ComplexF64}) = zero(ComplexF64)
@inline _sum_buckets(bs::Tuple, Ainv::Matrix{ComplexF64}) =
    _W_bucket(first(bs), Ainv) + _sum_buckets(Base.tail(bs), Ainv)

# Degree up to which `_hafnian` is emitted as a single fully unrolled, inlined expression. The
# expression has (K-1)!! product terms, so compile time explodes past this; higher degrees expand
# along the first index at runtime, recursing into the unrolled kernels below the threshold.
const _HAFNIAN_UNROLL_MAX = 10

# Linear index of pair (i < j) within the upper-triangle gather tuple of a degree-K monomial.
_haf_pairidx(i::Int, j::Int, K::Int) = (i-1)*K - (i*(i+1))÷2 + j

# Expression for the hafnian over monomial positions `pos`, reading pair values from the gathered
# upper-triangle tuple `S`. Expands along the first position: haf(pos) = Σₜ S[1,t]·haf(pos∖{1,t}).
# Identical sub-hafnian subtrees repeat across branches and get CSE'd during codegen.
function _haf_expr(pos::Vector{Int}, K::Int)
    isempty(pos) && return :(one(ComplexF64))
    length(pos) == 2 && return :(S[$(_haf_pairidx(pos[1], pos[2], K))])
    terms = Expr[]
    for t in 2:length(pos)
        rest = [pos[r] for r in 2:length(pos) if r != t]
        push!(terms, :(S[$(_haf_pairidx(pos[1], pos[t], K))] * $(_haf_expr(rest, K))))
    end
    Expr(:call, :+, terms...)
end

"""
    _hafnian(A::Matrix{ComplexF64}, m::NTuple{K,Int})

Wick contraction of one degree-`K` monomial: the sum over all perfect pairings of `m` of the
product of `A[m[i], m[j]]` over the pairs, i.e. the hafnian of the submatrix `A[m, m]`.

Recursive expansion along the first index shares sub-hafnian prefixes between pairings, so it
needs far fewer multiplies than enumerating `wick_partitions(K)` (147 vs 420 at K = 8). Odd-degree
monomials have no perfect pairing and contract to zero, matching `wick_out`.
"""
@inline _hafnian(A::Matrix{ComplexF64}, ::Tuple{}) = one(ComplexF64)
@inline _hafnian(A::Matrix{ComplexF64}, m::NTuple{2,Int}) = @inbounds A[m[1], m[2]]
@generated function _hafnian(A::Matrix{ComplexF64}, m::NTuple{K,Int})::ComplexF64 where {K}
    isodd(K) && return :(zero(ComplexF64))
    if K <= _HAFNIAN_UNROLL_MAX
        # Gather the upper-triangle submatrix once, then evaluate the fully unrolled expansion in
        # registers. Inlined into `_W_bucket`'s monomial loop.
        gathers = Expr[]
        for i in 1:K-1, j in i+1:K
            push!(gathers, :(@inbounds A[m[$i], m[$j]]))
        end
        return quote
            $(Expr(:meta, :inline))
            S = $(Expr(:tuple, gathers...))
            $(_haf_expr(collect(1:K), K))
        end
    else
        # One level of expansion along the first index; recursion reaches the fast unrolled
        # kernels once the tuple shrinks to _HAFNIAN_UNROLL_MAX.
        stmts = Expr[]
        for t in 2:K
            rest = Expr(:tuple, (:(m[$r]) for r in 2:K if r != t)...)
            push!(stmts, :(s += @inbounds(A[m[1], m[$t]]) * _hafnian(A, $rest)))
        end
        return quote
            s = zero(ComplexF64)
            $(stmts...)
            s
        end
    end
end

# TODO: Implement some up-front bounds safety check so that the inner loop can be fully @inbounds.
@inline function _W_bucket(b::WBucket{N}, Ainv::Matrix{ComplexF64})::ComplexF64 where {N}
    s = zero(ComplexF64)
    @inbounds for x in eachindex(b.coeffs)
        s += b.coeffs[x] * _hafnian(Ainv, b.indices[x])
    end
    s
end

"""
    W(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem}, Ainv::Matrix{ComplexF64})

Evaluate a (symbolic) moment polynomial by Wick contraction against `Ainv`.

# Parameters
- C    : Moment polynomial to evaluate (symbolic)
- Ainv : Inverse A-matrix providing the contraction kernel

# Returns
Gaussian moment implied by `C` and `Ainv`.
"""
function W(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem}, Ainv::Matrix{ComplexF64})::ComplexF64
    n_vars = nvars(parent(C))
    n_vars ≤ size(Ainv, 1) || throw(ArgumentError("C has $n_vars variables but Ainv is only $(size(Ainv, 1))×$(size(Ainv, 2))"))
    elm = zero(ComplexF64)
    for (mon, coeff) in zip(monomials(C), coefficients(C))
        elm += coeff * wick_out([i for i in 1:n_vars if exponent(mon, 1, i) == 1], Ainv)
    end
    return elm
end

"""
Evaluate a single monomial term via Wick's theorem (sum over perfect pairings).
"""
function wick_out(moment::Vector{Int}, Ainv::Matrix{ComplexF64})::ComplexF64
    N = length(moment)
    iseven(N) || return 0 # monomial with odd number of variables has no PMP set

    # Iterate over Wick partitions
    s = zero(ComplexF64)
    parts = wick_partitions(N)
    n_parts = size(parts, 1); n_pairs = size(parts, 3)
    @inbounds for m in 1:n_parts
        f = one(ComplexF64)
        for n in 1:n_pairs
            i = parts[m, 1, n]; j = parts[m, 2, n]
            f *= Ainv[moment[i], moment[j]]
        end
        s += f
    end
    s
end
