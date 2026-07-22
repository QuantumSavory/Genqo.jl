using Nemo
using LinearAlgebra
import LinearAlgebra: tr, dot, norm
using Gabs
import Gabs: nmodes
using QuantumOpticsBase
import QuantumOpticsBase: fidelity, projector

export
    # Types
    AbstractClickState, ClickStateKet, ClickStateBra, AbstractClickOperator, ClickProjector, AbstractProjectionEngine, HybridProjectionEngine, AbstractProjectedState, AbstractProjectedPureGaussianState, ProjectedPureGaussianState,
    # Functions
    clicks, projector, norm, project, tr, dot, fidelity, to_fock, duankimble


# TODO: inherit from something in QuantumOptics.jl and make this a full-fledged state
abstract type AbstractClickState end

struct ClickStateKet <: AbstractClickState
    coefs::Vector{ComplexF64}
    clicks::Array{Int,2}
    function ClickStateKet(coefs::Vector{ComplexF64}, clicks::Array{Int,2})
        length(coefs) == size(clicks, 1) || throw(ArgumentError("Length of coefficients must match number of click patterns"))
        all(clicks .≥ 0) || throw(ArgumentError("Click patterns must be non-negative integers"))
        new(coefs, clicks)
    end
end
clicks(cl::Vector{Int}) = ClickStateKet([one(ComplexF64)], reshape(cl, 1, :))
function Base.show(io::IO, cs::ClickStateKet)
    Base.summary(io, cs)
    print(io, join(["\n($cf)|$(join(cl, ","))⟩" for (cf, cl) in zip(cs.coefs, eachrow(cs.clicks))], " + "))
end
Base.:(==)(a::ClickStateKet, b::ClickStateKet) = a.clicks == b.clicks && a.coefs == b.coefs
Base.isapprox(a::ClickStateKet, b::ClickStateKet; kwargs...) = a.clicks == b.clicks && isapprox(a.coefs, b.coefs; kwargs...)
Base.:+(a::ClickStateKet, b::ClickStateKet) = ClickStateKet(vcat(a.coefs, b.coefs), vcat(a.clicks, b.clicks))
Base.:-(a::ClickStateKet, b::ClickStateKet) = ClickStateKet(vcat(a.coefs, -b.coefs), vcat(a.clicks, b.clicks))
Base.:*(a::N, b::ClickStateKet) where {N<:Number} = ClickStateKet(a .* b.coefs, b.clicks)
Base.:*(a::ClickStateKet, b::N) where {N<:Number} = ClickStateKet(a.coefs .* b, a.clicks)
Base.:/(a::ClickStateKet, b::N) where {N<:Number} = ClickStateKet(a.coefs ./ b, a.clicks)
Base.adjoint(a::ClickStateKet) = ClickStateBra(conj(a.coefs), a.clicks)
nmodes(cs::ClickStateKet) = size(cs.clicks, 2)

struct ClickStateBra <: AbstractClickState
    coefs::Vector{ComplexF64}
    clicks::Array{Int,2}
    function ClickStateBra(coefs::Vector{ComplexF64}, clicks::Array{Int,2})
        length(coefs) == size(clicks, 1) || throw(ArgumentError("Length of coefficients must match number of click patterns"))
        all(clicks .≥ 0) || throw(ArgumentError("Click patterns must be non-negative integers"))
        new(coefs, clicks)
    end
end
function Base.show(io::IO, cs::ClickStateBra)
    Base.summary(io, cs)
    print(io, join(["\n($cf)⟨$(join(cl, ","))|" for (cf, cl) in zip(cs.coefs, eachrow(cs.clicks))], " + "))
end
Base.:(==)(a::ClickStateBra, b::ClickStateBra) = a.clicks == b.clicks && a.coefs == b.coefs
Base.isapprox(a::ClickStateBra, b::ClickStateBra; kwargs...) = a.clicks == b.clicks && isapprox(a.coefs, b.coefs; kwargs...)
Base.:+(a::ClickStateBra, b::ClickStateBra) = ClickStateBra(vcat(a.coefs, b.coefs), vcat(a.clicks, b.clicks))
Base.:-(a::ClickStateBra, b::ClickStateBra) = ClickStateBra(vcat(a.coefs, -b.coefs), vcat(a.clicks, b.clicks))
Base.:*(a::N, b::ClickStateBra) where {N<:Number} = ClickStateBra(a .* b.coefs, b.clicks)
Base.:*(a::ClickStateBra, b::N) where {N<:Number} = ClickStateBra(a.coefs .* b, a.clicks)
Base.:/(a::ClickStateBra, b::N) where {N<:Number} = ClickStateBra(a.coefs ./ b, a.clicks)
Base.adjoint(a::ClickStateBra) = ClickStateKet(conj(a.coefs), a.clicks)
nmodes(cs::ClickStateBra) = size(cs.clicks, 2)

dot(bra::ClickStateBra, ket::ClickStateKet) = sum(bcf * kcf * prod(bcl .== kcl) for (bcf, bcl) in zip(bra.coefs, eachrow(bra.clicks)), (kcf, kcl) in zip(ket.coefs, eachrow(ket.clicks)))
norm(cs::ClickStateKet) = √dot(cs', cs)
norm(cs::ClickStateBra) = √dot(cs, cs')


abstract type AbstractClickOperator end

_replace_colon(::Any) = throw(ArgumentError("Photon number clicks must be Int or Colon"))
_replace_colon(i::Int) = i
_replace_colon(::Colon) = -1

struct ClickProjector <: AbstractClickOperator
    clicks::Array{Int,2}
    function ClickProjector(clicks::Array{Int,2})
        all(clicks .≥ -1) || throw(ArgumentError("Click projector must be non-negative integers or -1 for traceout"))
        size(clicks, 1) == 1 || allequal(outcome .== -1 for outcome in eachrow(clicks)) || throw(ArgumentError("Placement of traceout modes must be consistent across all click pattern terms"))
        new(clicks)
    end
end
projector(clicks::Vector{Int}) = ClickProjector(reshape(clicks, 1, :)) # TODO: fix type piracy
projector(clicks::Vector) = projector(_replace_colon.(clicks))
function Base.show(io::IO, cp::ClickProjector)
    Base.summary(io, cp)
    for cl in eachrow(cp.clicks)
        s = join([c == -1 ? ':' : c for c in cl], ",")
        print(io, "\n|$s⟩⟨$s|")
    end
end
Base.:(==)(a::ClickProjector, b::ClickProjector) = a.clicks == b.clicks
Base.:+(a::ClickProjector, b::ClickProjector) = ClickProjector(vcat(a.clicks, b.clicks))
nmodes(cp::ClickProjector) = size(cp.clicks, 2)
freemodes(cp::ClickProjector) = findall(==(-1), cp.clicks[1,:])


abstract type AbstractProjectionEngine end

mutable struct HybridProjectionEngine <: AbstractProjectionEngine
    mds::Int

    # Phase-space variables for symbolic calculations
    α::Vector{Generic.MPoly{ComplexFieldElem}}
    β::Vector{Generic.MPoly{ComplexFieldElem}}

    # Cache for C polynomials (α_click, β_click) => contraction terms
    C_poly_cache::Dict{Tuple{Vector{Int8}, Vector{Int8}}, WTerms}
    const C_poly_cache_lock::ReentrantLock # for multithreading safety

    C_poly_cache_ext::Dict{Tuple{Vector{Int8}, Vector{Float64}, Vector{Int8}, Int8, Int8}, WTerms}
    const C_poly_cache_ext_lock::ReentrantLock

    function HybridProjectionEngine(mds::Int)
        # Define phase-space variables for the circuit
        _αi = ["α$i" for i in 1:mds]
        _βi = ["β$i" for i in 1:mds]
        CC = ComplexField()
        R, generators = polynomial_ring(CC, hcat(_αi, _βi))

        # Define the α and β* vectors (note that β is understood to refer to the β* block)
        α = generators[:,1]
        β = generators[:,2]

        new(
            mds, α, β,
            Dict{Tuple{Vector{Int8}, Vector{Int8}}, WTerms}(), ReentrantLock(),
            Dict{Tuple{Vector{Int8}, Vector{Float64}, Vector{Int8}, Int8, Int8}, WTerms}(), ReentrantLock(),
        )
    end
end

abstract type AbstractProjectedState end
abstract type AbstractProjectedPureGaussianState end

struct ProjectedPureGaussianState <: AbstractProjectedPureGaussianState
    st::GaussianState
    proj::ClickProjector
    engine::HybridProjectionEngine
    η::Vector{Float64}
    function ProjectedPureGaussianState(st::GaussianState, proj::ClickProjector, engine::HybridProjectionEngine, η::Vector{Float64})
        engine.mds == nmodes(st) == nmodes(proj) == length(η) || throw(ArgumentError("Engine, state, projector, and loss vector must have the same number of modes"))
        all(η .≥ 0) && all(η .≤ 1) || throw(ArgumentError("Loss vector must be between 0 and 1"))
        all(iszero.(st.mean)) || throw(ArgumentError("Input Gaussian state must have zero displacement"))
        purity(st) ≈ 1. || throw(ArgumentError("Input Gaussian state must be pure"))
        all(proj.clicks .== 0 .|| proj.clicks .== 1 .|| proj.clicks .== -1) || throw(ArgumentError("Detector outcomes must be 0, 1, or -1")) # TODO: support multi-photon number outcomes
        new(st, proj, engine, η)
    end
end
freemodes(projected_state::ProjectedPureGaussianState) = freemodes(projected_state.proj)
project(st::GaussianState, proj::ClickProjector; engine::HybridProjectionEngine, η::Vector{Float64}=ones(engine.mds)) = ProjectedPureGaussianState(st, proj, engine, η)

"""
    tr(projected_state::ProjectedPureGaussianState)

Computes the probability of success for a given projected pure Gaussian state.
"""
function tr(projected_state::ProjectedPureGaussianState)::Float64
    st = projected_state.st
    engine = projected_state.engine
    α = engine.α
    β = engine.β
    proj = projected_state.proj
    η = projected_state.η

    # Convert to K-function representation and compute probabilities.
    # Gabs uses the ħ=2 convention, so rescale accordingly.
    gstate = changebasis(QuadBlockBasis, st)
    σ = gstate.covar ./ gstate.ħ

    Tr = zero(ComplexF64)
    for n in eachrow(proj.clicks)
        nf = max.(n, 0) # Filter out -1 (traceout) modes
        A, denom = A_matrix(σ, η, n; traceout=true)
        invA = inv(A)

        ηweight = prod(η .^ nf) # √η per detected photon, matching dot()'s per-click η factor
        C = @lock engine.C_poly_cache_lock get!(engine.C_poly_cache, (nf, nf)) do
            # TODO: support higher photon number outcomes as well, which will involve including the appropriate Fock term (αβ*)ⁿ/n! in the C polynomial
            # tools.W() will need to be generalized
            prod((α.*β).^nf ./ factorial.(nf)) |> extract_W_terms
        end

        Tr += W(C, invA) * ηweight / denom
    end

    @assert abs(imag(Tr)) ≤ 1e-10 * abs(Tr) "Trace of a density matrix should be real, but got $Tr"
    real(Tr)
end

function dot(bra::ClickStateBra, projected_state::ProjectedPureGaussianState, ket::ClickStateKet)::ComplexF64 # TODO: is this the right return type?
    st = projected_state.st
    η = projected_state.η
    engine = projected_state.engine
    α = engine.α
    β = engine.β
    proj = projected_state.proj
    count(proj.clicks[1,:] .== -1) == nmodes(bra) == nmodes(ket) || throw(ArgumentError("Bra and ket must have the same number of modes as the projected state"))

    gstate = changebasis(QuadBlockBasis, st)
    σ = gstate.covar ./ gstate.ħ

    Dot = zero(ComplexF64)
    for n in eachrow(proj.clicks)
        A, denom = A_matrix(σ, η, n; traceout=false)
        invA = inv(A)

        C = zero(WTerms{Tuple{}}) # Start with empty WTerms
        for (bcf, bcl) in zip(bra.coefs, eachrow(bra.clicks)), (kcf, kcl) in zip(ket.coefs, eachrow(ket.clicks))
            bcl_full, kcl_full = copy(n), copy(n)
            bcl_full[n .== -1] .= bcl # Build full click patterns for the bra and ket, filling in the undetected modes with the click patterns from the bra and ket
            kcl_full[n .== -1] .= kcl
            Cij = @lock engine.C_poly_cache_lock get!(engine.C_poly_cache, (bcl_full, kcl_full)) do
                prod(α .^ bcl_full) * prod(β .^ kcl_full) |> extract_W_terms
            end
            ηweight = prod(η.^((bcl_full.+kcl_full)./2) ./ (sqrt.(factorial.(bcl_full)) .* sqrt.(factorial.(kcl_full)))) # √η per detected photon on each of the bra and ket sides, matching tr()'s per-click η factor
            C += bcf * kcf * ηweight * Cij
        end

        Dot += W(C, invA) / denom
    end
    Dot
end
fidelity(target::ClickStateKet, projected_state::ProjectedPureGaussianState) = dot(target', projected_state, target) / tr(projected_state)
fidelity(target::ClickStateBra, projected_state::ProjectedPureGaussianState) = dot(target, projected_state, target') / tr(projected_state)

"""
Computes the unnormalized Fock-basis photon-photon density matrix of a projected pure Gaussian state
"""
function to_fock(projected_state::ProjectedPureGaussianState; cutoff::Int=2)::Operator
    proj = projected_state.proj
    m = count(proj.clicks[1,:] .== -1) # Number of modes to include in the resulting density matrix (traceout placement is consistent across terms)
    basis = reduce(⊗, FockBasis(cutoff) for _ in 1:m) # (cutoff+1)^m-dimensional Hilbert space for the density matrix

    sz = length(basis)
    dm = Operator(basis, Matrix{ComplexF64}(undef, sz, sz))

    # TODO: check that the ordering given by the iterator matches QuantumOpticsBase's ordering in the density matrix
    inds = Iterators.product(repeat([0:cutoff], m)...)
    for (i, bcl) in enumerate(inds), (j, kcl) in enumerate(inds)
        bra = clicks(collect(bcl))'; ket = clicks(collect(kcl))
        dm.data[i, j] = dot(bra, projected_state, ket)
    end

    dm
end

"""
Models Duan-Kimble loading into a spin quantum memory
"""
function duankimble(projected_state::ProjectedPureGaussianState, d::Vector{Int}, modes::Vector{Tuple{Int,Int}}=_pair(freemodes(projected_state)))::Operator
    st = projected_state.st
    engine = projected_state.engine
    α = engine.α
    β = engine.β
    proj = projected_state.proj
    η = projected_state.η
    mds = engine.mds

    length(d) == 2length(modes) || throw(ArgumentError("Duan-Kimble loading requires one outcome per mode"))
    count(proj.clicks[1,:] .== -1) == 2length(modes) || throw(ArgumentError("Duan-Kimble loading requires that the projected state has traceout on the modes to be loaded"))

    n_mem = length(modes)
    M = 2^n_mem

    gstate = changebasis(QuadBlockBasis, st)
    σ = gstate.covar ./ gstate.ħ

    ρ = Operator(reduce(⊗, SpinBasis(1//2) for _ in 1:n_mem), zeros(ComplexF64, M, M))
    for n in eachrow(proj.clicks)
        A, denom = A_matrix(σ, η, n; traceout=false)
        invA = inv(A)

        for i in 1:mds
            A[i,      i     ] += η[i]
            A[i+mds,  i+mds ] += η[i]
            A[i+2mds, i+2mds] += η[i]
            A[i+3mds, i+3mds] += η[i]
        end

        nf = max.(n, 0) # Filter out -1 (traceout) modes
        ηweight = prod(η .^ nf) # √η per detected photon, matching dot()'s per-click η factor

        for r in 0:M-1, s in 0:M-1
            C = @lock engine.C_poly_cache_ext_lock get!(engine.C_poly_cache_ext, (nf, η, d, r, s)) do
                C = prod((α.*β).^nf ./ factorial.(nf))
                for (a,(i,j)) in enumerate(modes)
                    C *= (if (r >> (a-1)) & 1 == 0
                        (α[i]*√η[i] - α[j]*√η[j])^d[2a-1] * (α[i]*√η[i] + α[j]*√η[j])^d[2a]
                    else
                        (α[i]*√η[i] + α[j]*√η[j])^d[2a-1] * (α[i]*√η[i] - α[j]*√η[j])^d[2a]
                    end)
                end
                for (b,(i,j)) in enumerate(modes)
                    C *= (if (s >> (b-1)) & 1 == 0
                        (β[i]*√η[i] - β[j]*√η[j])^d[2b-1] * (β[i]*√η[i] + β[j]*√η[j])^d[2b]
                    else
                        (β[i]*√η[i] + β[j]*√η[j])^d[2b-1] * (β[i]*√η[i] - β[j]*√η[j])^d[2b]
                    end)
                end
                C |> extract_W_terms
            end
            ρ.data[r+1,s+1] += W(C, invA) * ηweight / denom
        end
    end
    ρ.data ./= 2^sum(d) * 4
    ρ
end
function _pair(modes::Vector{Int})
    iseven(length(modes)) || throw(ArgumentError("Duan-Kimble loading requires an even number of modes"))
    [(modes[2i-1], modes[2i]) for i in 1:(length(modes) ÷ 2)]
end


function A_matrix(σ::Matrix{Float64}, η::Vector{Float64}, n::SubArray{Int}; traceout::Bool=false)::Tuple{Matrix{ComplexF64}, Float64}
    size(σ, 1) == size(σ, 2) || throw(ArgumentError("Covariance matrix must be square"))
    mds = size(σ, 1) ÷ 2
    length(n) == length(η) == mds || throw(ArgumentError("Length of n and η must match number of modes in covariance matrix"))

    Γ = σ + 0.5*I
    Γinv = inv(Γ)

    # Views of Γinv blocks
    a  = @view Γinv[1:mds,      1:mds     ]
    c  = @view Γinv[1:mds,      mds+1:2mds]

    cᵀ = @view Γinv[mds+1:2mds, 1:mds     ]
    b  = @view Γinv[mds+1:2mds, mds+1:2mds]

    @assert 0.5 * (a + b - im*(c - cᵀ)) ≈ I # Ã purity check

    # Compute A matrix from blocks (block ordering: [α β* α* β])
    # TODO: is this possible with a 2mds×2mds matrix in [α β*] ordering? C polynomials never have α* or β terms in them. Tricky part is we need A⁻¹, not just A. We also need a way to calculate det(A)
    A = zeros(ComplexF64, 4mds, 4mds)
    @views @. A[2mds+1:3mds, 2mds+1:3mds] = 0.5 * (a - b + im*(c + cᵀ)) # C̃
    @views @. A[3mds+1:4mds, 3mds+1:4mds] = 0.5 * (a - b - im*(c + cᵀ)) # C̃*
    
    for (i, ni) in enumerate(n)
        # Expansion of αβ*(1 - η), or αβ* for A_pgen transmitted modes: blocks (1,2) and (2,1)
        l = traceout && ni == -1 ? -one(ComplexF64) : ComplexF64(η[i] - 1.)
        A[i,     i+mds] = l
        A[i+mds, i    ] = l

        # Identity: blocks (1,3) (2,4) (3,1) and (4,2)
        A[i,      i+2mds] = one(ComplexF64)
        A[i+mds,  i+3mds] = one(ComplexF64)
        A[i+2mds, i     ] = one(ComplexF64)
        A[i+3mds, i+mds ] = one(ComplexF64)
    end

    detΓ = det(Γ)
    denom = sqrt(det(A)) * detΓ^0.25 * conj(detΓ)^0.25
    (A, real(denom)) # sometimes denom ends up with small (1e-37im) imaginary part
end
