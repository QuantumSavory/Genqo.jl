using Nemo
using LinearAlgebra
import LinearAlgebra: tr, dot
using Gabs
using QuantumOpticsBase

export
    # Types
    AbstractClickState, ClickStateKet, ClickStateBra, AbstractProjector, HybridProjector, AbstractProjectedState, AbstractProjectedPureGaussianState, ProjectedPureGaussianState,
    # Functions
    clicks, project, tr, dot, fidelity, to_fock


abstract type AbstractClickState end

struct ClickStateKet <: AbstractClickState
    coefs::Vector{ComplexF64}
    clicks::Array{Int,2}
    function ClickStateKet(coefs::Vector{ComplexF64}, clicks::Array{Int,2})
        length(coefs) == size(clicks, 1) || throw(ArgumentError("Length of coefficients must match number of click patterns"))
        new(coefs, clicks)
    end
end
clicks(clicks::Vector{Int}) = ClickStateKet([one(ComplexF64)], reshape(clicks, 1, :))
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


abstract type AbstractProjector end

mutable struct HybridProjector <: AbstractProjector
    mds::Int

    # Phase-space variables for symbolic calculations
    α::Vector{Generic.MPoly{ComplexFieldElem}}
    β::Vector{Generic.MPoly{ComplexFieldElem}}

    # Cache for C polynomials (α_click, β_click) => contraction terms
    C_poly_cache::Dict{Tuple{Vector{Int8}, Vector{Int8}}, WTerms}
    const C_poly_cache_lock::ReentrantLock # for multithreading safety

    function HybridProjector(mds::Int)
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

function project(st::GaussianState, proj::HybridProjector, detector_outcomes::Vector{Int}; η::Vector{Float64}=ones(proj.mds))
    proj.mds == nmodes(st) || throw(ArgumentError("Projector and state must have the same number of modes"))
    proj.mds == length(detector_outcomes) || throw(ArgumentError("Projector and detector outcomes must have the same number of modes"))
    proj.mds == length(η) || throw(ArgumentError("Projector and loss vector must have the same number of modes"))
    all(η .>= 0) && all(η .<= 1) || throw(ArgumentError("Loss vector must be between 0 and 1"))
    all(detector_outcomes .== 0 .|| detector_outcomes .== 1 .|| detector_outcomes .== -1) || throw(ArgumentError("Detector outcomes must be 0, 1, or -1"))
    if purity(st) ≈ 1.
        ProjectedPureGaussianState(st, proj, detector_outcomes, η)
    else
        throw(ArgumentError("Only pure Gaussian states are currently supported."))
    end
end
_replace_colon(::Any) = throw(ArgumentError("Detector outcomes must be Int or Colon"))
_replace_colon(i::Int) = i
_replace_colon(::Colon) = -1
project(st::GaussianState, proj::HybridProjector, detector_outcomes::Vector{Any}; η::Vector{Float64}=ones(proj.mds)) = ProjectedPureGaussianState(st, proj, _replace_colon.(detector_outcomes), η)


abstract type AbstractProjectedState end
abstract type AbstractProjectedPureGaussianState end

struct ProjectedPureGaussianState <: AbstractProjectedPureGaussianState
    st::GaussianState
    proj::HybridProjector
    detector_outcomes::Vector{Int}
    η::Vector{Float64}
end

"""
    tr(projected_state::ProjectedPureGaussianState)

Computes the probability of success for a given projected pure Gaussian state.
"""
function tr(projected_state::ProjectedPureGaussianState)::Float64
    st = projected_state.st
    proj = projected_state.proj
    α = proj.α
    β = proj.β
    detector_outcomes = projected_state.detector_outcomes
    detector_outcomes_notraceout = max.(detector_outcomes, 0) # Filter out -1 (traceout) modes
    η = projected_state.η

    # Convert to K-function representation and compute probabilities.
    # Gabs uses the ħ=2 convention, so rescale accordingly.
    gstate = changebasis(QuadBlockBasis, st)
    σ = gstate.covar ./ gstate.ħ
    A, Γ = A_matrix(σ, η, detector_outcomes; traceout=true)

    invA, detA = inv(A), det(A)
    detΓ = det(Γ)
    ηweight = prod(η .^ detector_outcomes_notraceout) # √η per detected photon, matching dot()'s per-click η factor

    C = @lock proj.C_poly_cache_lock get!(proj.C_poly_cache, (detector_outcomes_notraceout, detector_outcomes_notraceout)) do
        # TODO: support higher photon number outcomes as well, which will involve including the appropriate Fock term (αβ*)ⁿ/n! in the C polynomial
        # tools.W() will need to be generalized
        prod((α .* β) .^ detector_outcomes_notraceout) |> extract_W_terms
    end

    Tr = W(C, invA) * ηweight / (sqrt(detA)*detΓ^0.25*conj(detΓ)^0.25)
    @assert isreal(Tr) "Trace of a density matrix should be real, but got $Tr"
    real(Tr)
end

function dot(bra::ClickStateBra, projected_state::ProjectedPureGaussianState, ket::ClickStateKet)::ComplexF64 # TODO: is this the right return type?
    st = projected_state.st
    η = projected_state.η
    proj = projected_state.proj
    α = proj.α
    β = proj.β
    detector_outcomes = projected_state.detector_outcomes
    count(detector_outcomes .== -1) == nmodes(bra) == nmodes(ket) || throw(ArgumentError("Bra and ket must have the same number of modes as the projected state"))

    gstate = changebasis(QuadBlockBasis, st)
    σ = gstate.covar ./ gstate.ħ
    A, Γ = A_matrix(σ, η, detector_outcomes; traceout=false)

    invA, detA = inv(A), det(A)
    detΓ = det(Γ)

    C = zero(WTerms{Tuple{}}) # Start with empty WTerms
    for (bcf, bcl) in zip(bra.coefs, eachrow(bra.clicks)), (kcf, kcl) in zip(ket.coefs, eachrow(ket.clicks))
        bcl_full, kcl_full = copy(detector_outcomes), copy(detector_outcomes)
        bcl_full[detector_outcomes .== -1] .= bcl # Build full click patterns for the bra and ket, filling in the undetected modes with the click patterns from the bra and ket
        kcl_full[detector_outcomes .== -1] .= kcl
        Cij = @lock proj.C_poly_cache_lock get!(proj.C_poly_cache, (bcl_full, kcl_full)) do
            prod(α .^ bcl_full) * prod(β .^ kcl_full) |> extract_W_terms
        end
        ηweight = prod(η .^ ((bcl_full .+ kcl_full) ./ 2)) # √η per detected photon on each of the bra and ket sides, matching tr()'s per-click η factor
        C += bcf * kcf * ηweight * Cij
    end

    W(C, invA) / (sqrt(detA)*detΓ^0.25*conj(detΓ)^0.25)
end
fidelity(target::ClickStateKet, projected_state::ProjectedPureGaussianState) = dot(target', projected_state, target) / tr(projected_state)
fidelity(target::ClickStateBra, projected_state::ProjectedPureGaussianState) = dot(target, projected_state, target') / tr(projected_state)

"""
Computes the unnormalized Fock-basis photon-photon density matrix of a projected pure Gaussian state
"""
function to_fock(projected_state::ProjectedPureGaussianState; cutoff::Int=2)::Operator
    detector_outcomes = projected_state.detector_outcomes
    m = count(detector_outcomes .== -1) # Number of modes to include in the resulting density matrix
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


function A_matrix(σ::Matrix{Float64}, η::Vector{Float64}, detector_outcomes::Vector{Int}; traceout::Bool=false)::Tuple{Matrix{ComplexF64}, Matrix{Float64}}
    mds = size(σ, 1) ÷ 2
    Γ = σ + 0.5*I
    Γinv = inv(Γ)

    # Views of Γinv blocks
    A  = @view Γinv[1:mds,      1:mds     ]
    C  = @view Γinv[1:mds,      mds+1:2mds]
    Cᵀ = @view Γinv[mds+1:2mds, 1:mds     ]
    B  = @view Γinv[mds+1:2mds, mds+1:2mds]

    # Compute K matrix (block diagonal [BB, conj(BB)]) from Γinv blocks
    G = zeros(ComplexF64, 4mds, 4mds)
    @views @. G[1:mds,      1:mds     ] = 0.5*A  + (0.25im)*(C + Cᵀ)
    @views @. G[1:mds,      mds+1:2mds] = 0.5*C  - (0.25im)*(A - B)
    @views @. G[mds+1:2mds, 1:mds     ] = 0.5*Cᵀ - (0.25im)*(A - B)
    @views @. G[mds+1:2mds, mds+1:2mds] = 0.5*B  - (0.25im)*(C + Cᵀ)
    @views @. G[2mds+1:4mds, 2mds+1:4mds] = conj(G[1:2mds, 1:2mds]) # TODO: can we eliminate this copy with views of K in the W() function? Or perhaps even a new type for this specific weird block structure of A = [a b; bᵀ a*]?
    
    # Add G matrix to K and return A = K + G
    G += 0.5*I
    for (i, n) in enumerate(detector_outcomes)
        if traceout && n == -1
            # No detector means we trace out the mode
            # Expansion of αβ*
            G[i,      i+2mds] += -0.5
            G[i,      i+3mds] +=  0.5im
            G[i+mds,  i+2mds] += -0.5im
            G[i+mds,  i+3mds] += -0.5
            G[i+2mds, i     ] += -0.5
            G[i+2mds, i+mds ] += -0.5im
            G[i+3mds, i     ] +=  0.5im
            G[i+3mds, i+mds ] += -0.5
        else
            # Fock term (αβ*)ⁿ/n! is handled in the moment polynomial C, as it is not inside an exp()
            # Expansion of αβ*(1 - η)
            li = 1 - η[i]
            G[i,      i+2mds] += -0.5   * li
            G[i,      i+3mds] +=  0.5im * li
            G[i+mds,  i+2mds] += -0.5im * li
            G[i+mds,  i+3mds] += -0.5   * li
            G[i+2mds, i     ] += -0.5   * li
            G[i+2mds, i+mds ] += -0.5im * li
            G[i+3mds, i     ] +=  0.5im * li
            G[i+3mds, i+mds ] += -0.5   * li
        end
    end
    (G, Γ)
end
