using TheEggman
using LinearAlgebra
using Nemo
using BlockDiagonals

export wick_out, W, WTerms, extract_W_terms


"""
$(TYPEDSIGNATURES)

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
    result
end
_wick_partitions(N::Int)::Array{Int8,3} = _wick_partitions(Int8(N))
const _wick_partitions_cache = Ref(Dict{Int8, Array{Int8,3}}())
wick_partitions(N::Int8)::Array{Int8,3} = get!(() -> _wick_partitions(N), _wick_partitions_cache[], N)
wick_partitions(N::Int)::Array{Int8,3} = wick_partitions(Int8(N)) # dispatch to Int8 version for caching


"""
$(TYPEDEF)

Supertype for precompiled moment polynomials.
"""
abstract type WBucket end

"""
$(TYPEDEF)

A homogeneous group of monomials that all have the same degree `N`. Stored as struct-of-arrays for
cache locality, with indices as `NTuple{N,Int}` so the inner Wick loop reads them from registers.
All monomials must be multilinear in x⃗ (all factors of degree ≤ 1) for the SM type.
"""
struct WBucketSM{N} <: WBucket
    coeffs::Vector{ComplexF64}
    indices::Array{Int,2} # TODO: could SArray be faster / more memory efficient?
end
WBucketSM{N}(coeffs::Vector{ComplexF64}, indices::Vector{Vector{Int}}) where {N} = WBucketSM{N}(coeffs, reduce(vcat, transpose.(indices)))
Base.show(io::IO, b::WBucketSM{N}) where {N} = print(io, "$N-factor single-moment Wick bucket with $(length(b.indices)) index sets")
Base.:*(a::C, b::WBucketSM{N}) where {C<:Number,N} = WBucketSM{N}(a * b.coeffs, b.indices)
Base.:*(a::WBucketSM{N}, b::C) where {C<:Number,N} = WBucketSM{N}(b * a.coeffs, a.indices)

"""
$(TYPEDEF)

A homogeneous group of monomials that all have the same degree `N`. Stored as struct-of-arrays for
cache locality, with indices as `NTuple{N,Int}` so the inner Wick loop reads them from registers.
Moments can be repeated for the RM type.
"""
struct WBucketRM{N} <: WBucket
    coeffs::Vector{ComplexF64}
    indices::Array{Int,2}
    rpt::Array{Int,2}
end
WBucketRM{N}(coeffs::Vector{ComplexF64}, indices::Vector{Vector{Int}}, rpt::Vector{Vector{Int}}) where {N} = WBucketRM{N}(coeffs, reduce(vcat, transpose.(indices)), reduce(vcat, transpose.(rpt)))
Base.show(io::IO, b::WBucketRM{N}) where {N} = print(io, "$N-factor repeated-moment Wick bucket with $(length(b.indices)) index sets")
Base.:*(a::C, b::WBucketRM{N}) where {C<:Number,N} = WBucketRM{N}(a * b.coeffs, b.indices, b.rpt)
Base.:*(a::WBucketRM{N}, b::C) where {C<:Number,N} = WBucketRM{N}(b * a.coeffs, a.indices, a.rpt)

"""
$(TYPEDEF)

A precompiled moment polynomial as a heterogeneous tuple of `WBucket`s, one per degree present in
the polynomial. The tuple type carries each bucket's `N` at compile time so iteration unrolls and
each `_W_bucket` call specializes on its bucket's degree.
"""
struct WTerms{B<:Tuple}
    buckets::B
    mds::Int
end
Base.zero(::Type{WTerms{B}}) where {B<:Tuple} = WTerms(B(), 0)
function Base.:+(a::WTerms, b::WTerms)
    a.mds == 0 && return b
    b.mds == 0 && return a
    a.mds == b.mds || throw(DimensionMismatch("Cannot add WTerms with different mode counts: $(a.mds) and $(b.mds)"))
    WTerms((a.buckets..., b.buckets...), a.mds) # concatenate buckets
end
Base.:*(a::C, b::WTerms) where {C<:Number} = WTerms(b.buckets .* a, b.mds)
Base.:*(a::WTerms, b::C) where {C<:Number} = WTerms(a.buckets .* b, a.mds)

"""
$(TYPEDSIGNATURES)

Precompile a Nemo polynomial into a `WTerms` object suitable for the fast `W(::WTerms, invA)` path.
"""
function extract_W_terms(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem})::WTerms
    n_vars = nvars(parent(C))
    iseven(n_vars) || throw(ArgumentError("Polynomial ring must have an even number of variables (α and β for every mode)"))

    by_deg_single = Dict{Int, Tuple{Vector{ComplexF64}, Vector{Vector{Int}}}}()
    by_deg_repeated = Dict{Int, Tuple{Vector{ComplexF64}, Vector{Vector{Int}}, Vector{Vector{Int}}}}()
    for (mon, coeff) in zip(monomials(C), coefficients(C))
        indices = Int[]; sizehint!(indices, n_vars)
        rpt = Int[]; sizehint!(rpt, n_vars)
        for i in 1:n_vars
            rpt_i = exponent(mon, 1, i)
            if rpt_i > 0
                push!(indices, i)
                push!(rpt, rpt_i)
            end
        end
        N = length(indices)
        if all(isone, rpt)
            cv, iv = get!(by_deg_single, N) do
                (ComplexF64[], Vector{Int}[])
            end
            push!(cv, ComplexF64(coeff))
            push!(iv, indices)
        else
            cv, iv, rv = get!(by_deg_repeated, N) do
                (ComplexF64[], Vector{Int}[], Vector{Int}[])
            end
            push!(cv, ComplexF64(coeff))
            push!(iv, indices)
            push!(rv, rpt)
        end
    end

    # Sort by descending bucket size so the dominant bucket runs first
    buckets = (
        (WBucketSM{N}(by_deg_single[N]...) for N in sort!(collect(keys(by_deg_single)); by = N -> -length(by_deg_single[N][1])))...,
        (WBucketRM{N}(by_deg_repeated[N]...) for N in sort!(collect(keys(by_deg_repeated)); by = N -> -length(by_deg_repeated[N][1])))...,
    )
    WTerms(buckets, n_vars÷2)
end

"""
$(TYPEDSIGNATURES)

Fast Wick evaluator for precompiled moment terms.
"""
function W(t::WTerms, invA::Matrix{ComplexF64})
    2*t.mds == size(invA, 1) == size(invA, 2) || throw(DimensionMismatch("invA must be a square matrix of size $(2*t.mds)×$(2*t.mds) to contract against a $(t.mds)-mode polynomial, got $(size(invA, 1))×$(size(invA, 2))"))
    _sum_buckets(t.buckets, invA)
end

# Recursive helper so a heterogeneous Tuple iterates type-stably (a plain `for` would infer the
# element type as the abstract join of the bucket types and dynamic-dispatch each call).
@inline _sum_buckets(::Tuple{}, ::Matrix{ComplexF64}) = zero(ComplexF64)
@inline _sum_buckets(bs::Tuple, invA::Matrix{ComplexF64}) = _W_bucket(first(bs), invA) + _sum_buckets(Base.tail(bs), invA)

@inline function _W_bucket(b::WBucketSM{N}, invA::Matrix{ComplexF64})::ComplexF64 where {N}
    s = zero(ComplexF64)
    @inbounds for (coeff,inds) in zip(b.coeffs, eachrow(b.indices))
        s += coeff * hafnian(view(invA, inds, inds); check_symmetric=false)
    end
    s
end
@inline function _W_bucket(b::WBucketRM{N}, invA::Matrix{ComplexF64})::ComplexF64 where {N}
    s = zero(ComplexF64)
    @inbounds for (coeff,inds,rpt) in zip(b.coeffs, eachrow(b.indices), eachrow(b.rpt))
        s += coeff * hafnian_repeated(view(invA, inds, inds), rpt; check_symmetric=false)
    end
    s
end

"""
$(TYPEDSIGNATURES)

Evaluate a (symbolic) moment polynomial by Wick contraction against `invA`.
"""
function W(C::Nemo.Generic.MPoly{Nemo.ComplexFieldElem}, invA::Matrix{ComplexF64})::ComplexF64
    n_vars = nvars(parent(C))
    iseven(n_vars) || throw(ArgumentError("Polynomial ring must have an even number of variables (α and β for every mode)"))
    n_vars == size(invA, 1) == size(invA, 2) || throw(DimensionMismatch("invA must be a square matrix of size $n_vars×$n_vars to contract against a $(n_vars÷2)-mode polynomial, got $(size(invA, 1))×$(size(invA, 2))"))
    elm = zero(ComplexF64)
    for (mon, coeff) in zip(monomials(C), coefficients(C))
        indices = Int[]
        for i in 1:n_vars
            push!(indices, fill(i, exponent(mon, 1, i))...) # for the nth power of a variable, push n copies of its index
        end
        elm += coeff * wick_out(indices, invA)
    end
    elm
end

"""
$(TYPEDSIGNATURES)

Evaluate a single monomial term via Wick's theorem (sum over perfect pairings).
"""
function wick_out(moment::Vector{Int}, invA::Matrix{ComplexF64})::ComplexF64
    N = length(moment)
    iseven(N) || return zero(ComplexF64) # monomial with odd number of variables has no PMP set

    # Iterate over Wick partitions
    s = zero(ComplexF64)
    parts = wick_partitions(N)
    n_parts = size(parts, 1); n_pairs = size(parts, 3)
    @inbounds for m in 1:n_parts
        f = one(ComplexF64)
        for n in 1:n_pairs
            i = parts[m, 1, n]; j = parts[m, 2, n]
            f *= invA[moment[i], moment[j]]
        end
        s += f
    end
    s
end
