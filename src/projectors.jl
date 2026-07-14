using Nemo
using LinearAlgebra
import LinearAlgebra: tr, dot, norm
using Gabs
import Gabs: nmodes
using QuantumOpticsBase
import QuantumOpticsBase: fidelity

export
    # Types
    AbstractClickState, ClickStateKet, ClickStateBra, AbstractClickOperator, ClickProjector, AbstractProjectionEngine, HybridProjectionEngine, AbstractProjectedState, AbstractProjectedPureGaussianState, ProjectedPureGaussianState,
    # Functions
    clicks, projector, norm, project, tr, dot, fidelity, to_fock


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
projector(clicks::Vector{Int}) = ClickProjector(reshape(clicks, 1, :))
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


abstract type AbstractProjectionEngine end

mutable struct HybridProjectionEngine <: AbstractProjectionEngine
    mds::Int

    # Phase-space variables for symbolic calculations
    α::Vector{Generic.MPoly{ComplexFieldElem}}
    β::Vector{Generic.MPoly{ComplexFieldElem}}

    # Cache for C polynomials (α_click, β_click) => contraction terms
    C_poly_cache::Dict{Tuple{Vector{Int8}, Vector{Int8}}, WTerms}
    const C_poly_cache_lock::ReentrantLock # for multithreading safety

    function HybridProjectionEngine(mds::Int)
        # Define canonical phase-space variables for the circuit
        _qai = ["qa$i" for i in 1:mds]
        _pai = ["pa$i" for i in 1:mds]
        _qbi = ["qb$i" for i in 1:mds]
        _pbi = ["pb$i" for i in 1:mds]
        all_qps = hcat(_qai, _pai, _qbi, _pbi)
        CC = ComplexField()
        i = onei(CC) # Imaginary unit in CC ring
        R, generators = polynomial_ring(CC, all_qps)
        (qai, pai, qbi, pbi) = (generators[:,i] for i in 1:4)

        # Define the α and β* vectors (note that we pre-conjugate β)
        α = @. (qai + i * pai) / √2
        β = @. (qbi - i * pbi) / √2

        new(mds, α, β, Dict{Tuple{Vector{Int8}, Vector{Int8}}, WTerms}(), ReentrantLock())
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
        A, Γ = A_matrix(σ, η, n; traceout=true)

        invA, detA = inv(A), det(A)
        detΓ = det(Γ)
        ηweight = prod(η .^ nf) # √η per detected photon, matching dot()'s per-click η factor

        C = @lock engine.C_poly_cache_lock get!(engine.C_poly_cache, (nf, nf)) do
            # TODO: support higher photon number outcomes as well, which will involve including the appropriate Fock term (αβ*)ⁿ/n! in the C polynomial
            # tools.W() will need to be generalized
            prod((α .* β) .^ nf) |> extract_W_terms
        end

        Tr += W(C, invA) * ηweight / (sqrt(detA)*detΓ^0.25*conj(detΓ)^0.25)
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
        A, Γ = A_matrix(σ, η, n; traceout=false)

        invA, detA = inv(A), det(A)
        detΓ = det(Γ)

        C = zero(WTerms{Tuple{}}) # Start with empty WTerms
        for (bcf, bcl) in zip(bra.coefs, eachrow(bra.clicks)), (kcf, kcl) in zip(ket.coefs, eachrow(ket.clicks))
            bcl_full, kcl_full = copy(n), copy(n)
            bcl_full[n .== -1] .= bcl # Build full click patterns for the bra and ket, filling in the undetected modes with the click patterns from the bra and ket
            kcl_full[n .== -1] .= kcl
            Cij = @lock engine.C_poly_cache_lock get!(engine.C_poly_cache, (bcl_full, kcl_full)) do
                prod(α .^ bcl_full) * prod(β .^ kcl_full) |> extract_W_terms
            end
            ηweight = prod(η .^ ((bcl_full .+ kcl_full) ./ 2)) # √η per detected photon on each of the bra and ket sides, matching tr()'s per-click η factor
            C += bcf * kcf * ηweight * Cij
        end

        Dot += W(C, invA) / (sqrt(detA)*detΓ^0.25*conj(detΓ)^0.25)
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


function A_matrix(σ::Matrix{Float64}, η::Vector{Float64}, n::AbstractArray; traceout::Bool=false)::Tuple{Matrix{ComplexF64}, Matrix{Float64}}
    mds = size(σ, 1) ÷ 2
    Γ = σ + 0.5*I
    Γinv = inv(Γ)

    # Views of Γinv blocks
    a  = @view Γinv[1:mds,      1:mds     ]
    c  = @view Γinv[1:mds,      mds+1:2mds]

    cᵀ = @view Γinv[mds+1:2mds, 1:mds     ]
    b  = @view Γinv[mds+1:2mds, mds+1:2mds]

    # Compute K matrix (block diagonal [BB, conj(BB)]) from Γinv blocks
    A = zeros(ComplexF64, 4mds, 4mds)
    @views @. A[1:mds,      1:mds     ] = 0.5*a  + (0.25im)*(c + cᵀ)
    @views @. A[1:mds,      mds+1:2mds] = 0.5*c  - (0.25im)*(a - b)
    @views @. A[mds+1:2mds, 1:mds     ] = 0.5*cᵀ - (0.25im)*(a - b)
    @views @. A[mds+1:2mds, mds+1:2mds] = 0.5*b  - (0.25im)*(c + cᵀ)
    @views @. A[2mds+1:4mds, 2mds+1:4mds] = conj(A[1:2mds, 1:2mds])
    # TODO: can we eliminate this copy with views of K in the W() function? Or perhaps even a new type for this specific weird block structure of A = [a b; bᵀ a*]?
    # TODO: can we do one bounds check up front and @inbounds the whole thing below?
    
    # Add G matrix to K and return A = K + G
    A += 0.5*I
    for (i, ni) in enumerate(n)
        if traceout && ni == -1
            # No detector means we trace out the mode
            # Expansion of αβ*
            A[i,      i+2mds] += -0.5
            A[i,      i+3mds] +=  0.5im
            A[i+mds,  i+2mds] += -0.5im
            A[i+mds,  i+3mds] += -0.5
            A[i+2mds, i     ] += -0.5
            A[i+2mds, i+mds ] += -0.5im
            A[i+3mds, i     ] +=  0.5im
            A[i+3mds, i+mds ] += -0.5
        else
            # Fock term (αβ*)ⁿ/n! is handled in the moment polynomial C, as it is not inside an exp()
            # Expansion of αβ*(1 - η)
            li = 1 - η[i]
            A[i,      i+2mds] += -0.5   * li
            A[i,      i+3mds] +=  0.5im * li
            A[i+mds,  i+2mds] += -0.5im * li
            A[i+mds,  i+3mds] += -0.5   * li
            A[i+2mds, i     ] += -0.5   * li
            A[i+2mds, i+mds ] += -0.5im * li
            A[i+3mds, i     ] +=  0.5im * li
            A[i+3mds, i+mds ] += -0.5   * li
        end
    end
    (A, Γ)
end
