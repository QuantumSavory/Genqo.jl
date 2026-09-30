using LazyArrays
using LinearAlgebra
import LinearAlgebra: tr, diag, dot
import QuantumOpticsBase: Operator, DenseOpType, dense, entropy_vn, entropy_renyi

export LazyDensityMatrix, ncomputed


"""
$(TYPEDEF)

An `M`×`M` matrix whose entries are produced on first access by `f(i, j)` and memoized, so each
entry is computed at most once no matter how often it is read.

See also [`ncomputed`](@ref), [`duankimble`](@ref), [`emissiveload`](@ref).

# Fields
$(TYPEDFIELDS)
"""
struct LazyDensityMatrix{F} <: LazyArrays.LazyMatrix{ComplexF64}
    f::F
    cache::Matrix{ComplexF64}
    computed::Matrix{Bool}
    lock::ReentrantLock
    function LazyDensityMatrix(f::F, M::Int) where {F}
        M ≥ 0 || throw(ArgumentError("Matrix dimension must be non-negative, got $M"))
        new{F}(f, Matrix{ComplexF64}(undef, M, M), zeros(Bool, M, M), ReentrantLock())
    end
end

Base.size(A::LazyDensityMatrix) = size(A.cache)
Base.IndexStyle(::Type{<:LazyDensityMatrix}) = IndexCartesian()

# Evaluate and memoize entry (i, j) if it is not already known. Caller must hold `A.lock`.
@inline function _fill_entry!(A::LazyDensityMatrix, i::Int, j::Int)
    @inbounds if !A.computed[i, j]
        A.cache[i, j] = A.f(i, j)
        A.computed[i, j] = true
    end
    @inbounds A.cache[i, j]
end

function Base.getindex(A::LazyDensityMatrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    @lock A.lock _fill_entry!(A, i, j)
end

function Base.setindex!(A::LazyDensityMatrix, v, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    @lock A.lock begin
        @inbounds A.cache[i, j] = v
        @inbounds A.computed[i, j] = true
    end
    v
end

# Whole-matrix reductions: one lock acquisition for the entries they need, rather than one per entry.
function Base.Matrix(A::LazyDensityMatrix)
    @lock A.lock begin
        for j in axes(A, 2), i in axes(A, 1)
            _fill_entry!(A, i, j)
        end
        copy(A.cache)
    end
end
Base.collect(A::LazyDensityMatrix) = Matrix(A)

function diag(A::LazyDensityMatrix)
    @lock A.lock [_fill_entry!(A, i, i) for i in axes(A, 1)]
end

function tr(A::LazyDensityMatrix)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("Cannot take the trace of a non-square matrix"))
    @lock A.lock begin
        t = zero(ComplexF64)
        for i in axes(A, 1)
            t += _fill_entry!(A, i, i)
        end
        t
    end
end

"""
$(TYPEDSIGNATURES)

Number of entries of `A` that have been evaluated so far, out of `length(A)`. Useful for confirming
that a calculation really did skip the entries it never needed.
"""
ncomputed(A::LazyDensityMatrix) = @lock A.lock count(A.computed)


# A `QuantumOpticsBase.Operator` whose data has not been materialized
const LazyOpType{BL,BR} = Operator{BL,BR,<:LazyArrays.LazyArray}

# Materialize LazyOpType for functions that require a dense operator
entropy_vn(ρ::LazyOpType, args...; kwargs...) = entropy_vn(dense(ρ), args...; kwargs...)
entropy_renyi(ρ::LazyOpType, args...; kwargs...) = entropy_renyi(dense(ρ), args...; kwargs...)
Base.exp(ρ::LazyOpType) = exp(dense(ρ))


# Define special method for `LinearAlgebra.dot(x, A::LazyDensityMatrix, y)` to reduce number of Wick contractions required
const _RowVector = Union{Adjoint{<:Any,<:AbstractVector}, Transpose{<:Any,<:AbstractVector}}
function _sandwich(x, A::LazyDensityMatrix, y::AbstractVector)
    length(x) == size(A, 1) && length(y) == size(A, 2) ||
        throw(DimensionMismatch("cannot sandwich a $(size(A, 1))×$(size(A, 2)) matrix between vectors of length $(length(x)) and $(length(y))"))
    rows = [i for (i, xi) in enumerate(x) if !iszero(xi)]
    s = zero(typeof(zero(eltype(x)) * zero(eltype(A)) * zero(eltype(y))))
    @lock A.lock begin
        for (j, yj) in enumerate(y)
            iszero(yj) && continue
            t = zero(s)
            for i in rows
                t += conj(x[i]) * _fill_entry!(A, i, j)
            end
            s += t * yj
        end
    end
    s
end
dot(x::AbstractVector, A::LazyDensityMatrix, y::AbstractVector) = _sandwich(x, A, y)
dot(x::_RowVector, A::LazyDensityMatrix, y::AbstractVector) = _sandwich(x, A, y)
