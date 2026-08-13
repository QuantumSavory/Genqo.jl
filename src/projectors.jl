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
    clicks, projector, norm, get_phase_space_generators_half, get_phase_space_generators_full, get_default_engine, project, tr, dot, fidelity, to_fock, duankimble, emissiveload, nmodes, nfreemodes, freemodes,
    # Re-exported from QuantumOpticsBase
    Operator


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
nfreemodes(cp::ClickProjector) = count(==(-1), cp.clicks[1,:])


abstract type AbstractProjectionEngine end

mutable struct HybridProjectionEngine <: AbstractProjectionEngine
    mds::Int

    # Phase-space variables for symbolic calculations
    # Most calculations only require α and βc, allowing us to use the smaller upper-left block of the A⁻¹ matrix for the Wick contractions. For this, phase_space_generators_half is used.
    # For calculations that require α βc αc and β, and hence the full A⁻¹ matrix, phase_space_generators_full is used.
    # The two sets of generators are kept separate so that A⁻¹ can be properly bounds-checked against the number of modes in the ring.
    phase_space_generators_half::Array{Generic.MPoly{ComplexFieldElem},2}
    phase_space_generators_full::Array{Generic.MPoly{ComplexFieldElem},2}

    # Cache for C polynomials (α_click, β_click) => contraction terms
    C_poly_cache::Dict{Tuple{Vector{Int}, Vector{Int}}, WTerms}
    const C_poly_cache_lock::ReentrantLock # for multithreading safety

    C_poly_cache_ext::Dict{Tuple{Vector{Int}, Vector{Float64}, Vector{Int}, Int, Int}, WTerms}
    const C_poly_cache_ext_lock::ReentrantLock

    function HybridProjectionEngine(mds::Int)
        # Define phase-space variables for the circuit
        _αi = ["α$i" for i in 1:mds]
        _βci = ["βc$i" for i in 1:mds]
        _αci = ["αc$i" for i in 1:mds]
        _βi = ["β$i" for i in 1:mds]
        CC = ComplexField()
        R_half, generators_half = polynomial_ring(CC, hcat(_αi, _βci))
        R_full, generators_full = polynomial_ring(CC, hcat(_αi, _βci, _αci, _βi))

        new(
            mds, generators_half, generators_full,
            Dict{Tuple{Vector{Int}, Vector{Int}}, WTerms}(), ReentrantLock(),
            Dict{Tuple{Vector{Int}, Vector{Float64}, Vector{Int}, Int, Int}, WTerms}(), ReentrantLock(),
        )
    end
end
get_phase_space_generators_half(engine::HybridProjectionEngine) = (engine.phase_space_generators_half[:,i] for i in 1:2)
get_phase_space_generators_full(engine::HybridProjectionEngine) = (engine.phase_space_generators_full[:,i] for i in 1:4)
const _default_engines = Dict{Int, HybridProjectionEngine}()
const _default_engines_lock = ReentrantLock() # for multithreading safety
function get_default_engine(mds::Int)
    @lock _default_engines_lock get!(_default_engines, mds) do
        HybridProjectionEngine(mds)
    end
end

abstract type AbstractProjectedState end
abstract type AbstractProjectedPureGaussianState end

struct ProjectedPureGaussianState <: AbstractProjectedPureGaussianState
    st::GaussianState
    proj::ClickProjector
    η::Vector{Float64}
    function ProjectedPureGaussianState(st::GaussianState, proj::ClickProjector, η::Vector{Float64})
        nmodes(st) == nmodes(proj) == length(η) || throw(ArgumentError("State, projector, and loss vector must have the same number of modes"))
        all(η .≥ 0) && all(η .≤ 1) || throw(ArgumentError("Loss vector must be between 0 and 1"))
        all(iszero.(st.mean)) || throw(ArgumentError("Input Gaussian state must have zero displacement"))
        purity(st) ≈ 1. || throw(ArgumentError("Input Gaussian state must be pure"))
        all(proj.clicks .== 0 .|| proj.clicks .== 1 .|| proj.clicks .== -1) || throw(ArgumentError("Detector outcomes must be 0, 1, or -1")) # TODO: support multi-photon number outcomes
        new(st, proj, η)
    end
end
nmodes(projected_state::ProjectedPureGaussianState) = nmodes(projected_state.st)
freemodes(projected_state::ProjectedPureGaussianState) = freemodes(projected_state.proj)
nfreemodes(projected_state::ProjectedPureGaussianState) = nfreemodes(projected_state.proj)
project(st::GaussianState, proj::ClickProjector; η::Vector{Float64}=ones(nmodes(st))) = ProjectedPureGaussianState(st, proj, η)

"""
    tr(projected_state::ProjectedPureGaussianState)

Computes the probability of success for a given projected pure Gaussian state.
"""
function tr(projected_state::ProjectedPureGaussianState; engine::HybridProjectionEngine=get_default_engine(nmodes(projected_state)))::Float64
    nmodes(projected_state) == engine.mds || throw(ArgumentError("Engine must have the same number of modes as the projected state"))

    st = projected_state.st
    α, βc = get_phase_space_generators_half(engine)
    proj = projected_state.proj
    η = projected_state.η

    # Convert to K-function representation and compute probabilities.
    # Gabs uses the ħ=2 convention, so rescale accordingly.
    gstate = changebasis(QuadBlockBasis, st)
    σ = gstate.covar ./ gstate.ħ

    Tr = zero(ComplexF64)
    for n in eachrow(proj.clicks)
        nf = max.(n, 0) # Filter out -1 (traceout) modes
        invA, denom = _invA_UL(σ, η, n)

        ηweight = prod(η .^ nf) # √η per detected photon, matching dot()'s per-click η factor
        C = @lock engine.C_poly_cache_lock get!(engine.C_poly_cache, (nf, nf)) do
            prod((α.*βc).^nf ./ factorial.(nf)) |> extract_W_terms
        end

        Tr += W(C, invA) * ηweight / denom
    end

    @assert abs(imag(Tr)) ≤ 1e-10 * abs(Tr) "Trace of a density matrix should be real, but got $Tr"
    real(Tr)
end

function dot(bra::ClickStateBra, projected_state::ProjectedPureGaussianState, ket::ClickStateKet; engine::HybridProjectionEngine=get_default_engine(nmodes(projected_state)))::ComplexF64 # TODO: is this the right return type?
    nmodes(projected_state) == engine.mds || throw(ArgumentError("Engine must have the same number of modes as the projected state"))

    st = projected_state.st
    η = projected_state.η
    α, βc = get_phase_space_generators_half(engine)
    proj = projected_state.proj
    nfreemodes(projected_state) == nmodes(bra) == nmodes(ket) || throw(ArgumentError("Bra and ket must have the same number of modes as the projected state"))

    gstate = changebasis(QuadBlockBasis, st)
    σ = gstate.covar ./ gstate.ħ

    Dot = zero(ComplexF64)
    for n in eachrow(proj.clicks)
        nf = max.(n, 0) # Filter out -1 (traceout) modes
        invA, denom = _invA_UL(σ, η, nf) # A_Ψ is same as A_P, just with no traceout modes

        C = zero(WTerms{Tuple{}}) # Start with empty WTerms
        for (bcf, bcl) in zip(bra.coefs, eachrow(bra.clicks)), (kcf, kcl) in zip(ket.coefs, eachrow(ket.clicks))
            bcl_full, kcl_full = copy(n), copy(n)
            bcl_full[n .== -1] .= bcl # Build full click patterns for the bra and ket, filling in the undetected modes with the click patterns from the bra and ket
            kcl_full[n .== -1] .= kcl
            Cij = @lock engine.C_poly_cache_lock get!(engine.C_poly_cache, (bcl_full, kcl_full)) do
                prod(α .^ bcl_full) * prod(βc .^ kcl_full) |> extract_W_terms
            end
            ηweight = prod(η.^((bcl_full.+kcl_full)./2) ./ (sqrt.(factorial.(bcl_full)) .* sqrt.(factorial.(kcl_full)))) # √η per detected photon on each of the bra and ket sides, matching tr()'s per-click η factor
            C += bcf * kcf * ηweight * Cij
        end

        Dot += W(C, invA) / denom
    end
    Dot
end
fidelity(target::ClickStateKet, projected_state::ProjectedPureGaussianState; engine::HybridProjectionEngine=get_default_engine(nmodes(projected_state))) = dot(target', projected_state, target; engine) / tr(projected_state; engine)
fidelity(target::ClickStateBra, projected_state::ProjectedPureGaussianState; engine::HybridProjectionEngine=get_default_engine(nmodes(projected_state))) = dot(target, projected_state, target'; engine) / tr(projected_state; engine)

"""
Computes the unnormalized Fock-basis photon-photon density matrix of a projected pure Gaussian state
"""
function to_fock(projected_state::ProjectedPureGaussianState; engine::HybridProjectionEngine=get_default_engine(nmodes(projected_state)), cutoff::Int=2)::Operator
    m = nfreemodes(projected_state) # Number of modes to include in the resulting density matrix (traceout placement is consistent across terms)
    basis = reduce(⊗, fill(FockBasis(cutoff), m)) # (cutoff+1)^m-dimensional Hilbert space for the density matrix

    sz = length(basis)
    dm = Operator(basis, Matrix{ComplexF64}(undef, sz, sz))

    # TODO: check that the ordering given by the iterator matches QuantumOpticsBase's ordering in the density matrix
    inds = Iterators.product(repeat([0:cutoff], m)...)
    for (i, bcl) in enumerate(inds), (j, kcl) in enumerate(inds)
        bra = clicks(collect(bcl))'; ket = clicks(collect(kcl))
        dm.data[i, j] = dot(bra, projected_state, ket; engine=engine)
    end

    dm
end

function _pair(modes::Vector{Int})
    iseven(length(modes)) || throw(ArgumentError("Loading of dual-rail states requires an even number of modes"))
    [(modes[2i-1], modes[2i]) for i in Base.OneTo(length(modes) ÷ 2)]
end

"""
Models Duan-Kimble loading into a spin quantum memory
"""
function duankimble(projected_state::ProjectedPureGaussianState, d::Vector{Int}, modes::Vector{Tuple{Int,Int}}=_pair(freemodes(projected_state)); engine::HybridProjectionEngine=get_default_engine(nmodes(projected_state)))::Operator
    nmodes(projected_state) == engine.mds || throw(ArgumentError("Engine must have the same number of modes as the projected state"))

    st = projected_state.st
    α, βc = get_phase_space_generators_half(engine)
    proj = projected_state.proj
    η = projected_state.η

    length(d) == 2length(modes) || throw(ArgumentError("Duan-Kimble loading requires one outcome per mode"))
    nfreemodes(projected_state) == 2length(modes) || throw(ArgumentError("Duan-Kimble loading requires that the projected state has traceout on the modes to be loaded"))

    n_mem = length(modes)
    M = 2^n_mem

    gstate = changebasis(QuadBlockBasis, st)
    σ = gstate.covar ./ gstate.ħ

    ρ = Operator(reduce(⊗, fill(SpinBasis(1//2), n_mem)), zeros(ComplexF64, M, M))
    for n in eachrow(proj.clicks)
        nf = max.(n, 0) # Filter out -1 (traceout) modes
        invA, denom = _invA_UL(σ, η, nf)
        
        ηweight = prod(η .^ nf) # √η per detected photon, matching dot()'s per-click η factor

        for r in 0:M-1, s in 0:M-1
            C = @lock engine.C_poly_cache_ext_lock get!(engine.C_poly_cache_ext, (nf, η, d, r, s)) do
                C = prod((α.*βc).^nf ./ factorial.(nf))
                mask = 0x1
                a = 1
                for (i,j) in modes
                    if r & mask == 0
                        C *= (α[i]*√η[i] - α[j]*√η[j])^d[2a-1] * (α[i]*√η[i] + α[j]*√η[j])^d[2a]
                    else
                        C *= (α[i]*√η[i] + α[j]*√η[j])^d[2a-1] * (α[i]*√η[i] - α[j]*√η[j])^d[2a]
                    end
                    mask <<= 1
                    a += 1
                end
                mask = 0x1
                a = 1
                for (i,j) in modes
                    if s & mask == 0
                        C *= (βc[i]*√η[i] - βc[j]*√η[j])^d[2a-1] * (βc[i]*√η[i] + βc[j]*√η[j])^d[2a]
                    else
                        C *= (βc[i]*√η[i] + βc[j]*√η[j])^d[2a-1] * (βc[i]*√η[i] - βc[j]*√η[j])^d[2a]
                    end
                    mask <<= 1
                    a += 1
                end
                C |> extract_W_terms
            end
            ρ.data[r+1,s+1] += W(C, invA) * ηweight / denom
        end
    end
    ρ.data ./= 2^sum(d) * 4
    ρ
end

"""
Models emissive loading into a quantum memory
"""
function emissiveload(projected_state::ProjectedPureGaussianState, load::Vector{Tuple{Int,Int}}=_pair(freemodes(projected_state)), emit::Vector{Tuple{Int,Int}}=collect((nmodes(projected_state)+2h-1,nmodes(projected_state)+2h) for h in Base.OneTo(length(load))); engine=get_default_engine(nmodes(projected_state.st)+2length(load)))::Operator
    st = projected_state.st
    α, βc, αc, β = get_phase_space_generators_full(engine)
    proj = projected_state.proj
    η = projected_state.η
    mds = engine.mds

    nmodes_st = nmodes(st)
    n_mem = length(load)
    M = 2^n_mem

    mds == nmodes_st + 2n_mem || throw(ArgumentError("Emissive loading requires an engine with enough modes for the state plus an extra for each loaded mode"))
    length(emit) == n_mem || throw(ArgumentError("Emissive loading requires one emission per loaded mode"))
    nfreemodes(projected_state) == 2n_mem || throw(ArgumentError("Emissive loading requires that the projected state has traceout on the modes to be loaded"))

    gstate = changebasis(QuadBlockBasis, st) ⊗ vacuumstate(QuadBlockBasis(2n_mem))
    σ = gstate.covar ./ gstate.ħ

    ρ = Operator(reduce(⊗, fill(SpinBasis(1//2), n_mem)), zeros(ComplexF64, M, M))
    for n in eachrow(proj.clicks)
        nfp = vcat(max.(n, 0), zeros(Int64, 2n_mem)) # Filter out -1 (traceout) modes
        ηp = vcat(η, ones(Float64, 2n_mem)) # Pad η to include the emission modes, assuming no loss
        invA, denom = _invA(σ, ηp, nfp)

        for r in 0:M-1, s in 0:M-1
            C = prod((α.*βc.*ηp).^nfp ./ factorial.(nfp))
            for (a, ((i,j), (k,l))) in enumerate(zip(load, emit))
                C *= (α[i]*√ηp[i] + α[k]*√ηp[k]) * (α[j]*√ηp[j] + α[l]*√ηp[l]) * (βc[i]*√ηp[i] + βc[k]*√ηp[k]) * (βc[j]*√ηp[j] + βc[l]*√ηp[l])
            end

            mask = 0x1
            for (i,j) in emit
                if r & mask == 0
                    C *= αc[j]
                else
                    C *= αc[i]
                end
                mask <<= 1
            end
            mask = 0x1
            for (i,j) in emit
                if s & mask == 0
                    C *= β[j]
                else
                    C *= β[i]
                end
                mask <<= 1
            end
            ρ.data[r+1,s+1] += W(extract_W_terms(C), invA) / denom
        end
    end
    ρ.data ./= 4^n_mem * 2
    ρ
end


function _invA(σ::Matrix{Float64}, η::Vector{Float64}, n::AbstractArray{Int})::Tuple{Matrix{ComplexF64}, Float64}
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

    C̃ = 0.5 * (a - b + im*(c + cᵀ))
    y = ones(ComplexF64, mds)
    y[n .!== -1] -= η[n .!== -1] # y_i = 1 - η_i for detected modes, y_i = 1 for transmitted modes
    Y = Diagonal(y)

    # Compute A matrix from blocks (block ordering: [α β* α* β])
    A = zeros(ComplexF64, 4mds, 4mds)

    copyto!(view(A, 1:mds,       mds+1:2mds ), -Y)
    copyto!(view(A, mds+1:2mds,  1:mds      ), -Y)
    copyto!(view(A, 1:mds,       2mds+1:3mds), I)
    copyto!(view(A, mds+1:2mds,  3mds+1:4mds), I)
    copyto!(view(A, 2mds+1:3mds, 1:mds      ), I)
    copyto!(view(A, 3mds+1:4mds, mds+1:2mds ), I)
    copyto!(view(A, 2mds+1:3mds, 2mds+1:3mds), C̃) # C̃
    conj!(C̃)
    copyto!(view(A, 3mds+1:4mds, 3mds+1:4mds), C̃) # C̃*

    invA = inv(A)
    detA = det(A)

    denom = sqrt(real(detA)) * sqrt(abs(det(Γ)))
    (invA, denom)
end

function _invA_UL(σ::Matrix{Float64}, η::Vector{Float64}, n::AbstractArray{Int})::Tuple{Matrix{ComplexF64}, Float64}
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

    local invA_UL, detA
    C̃ = 0.5 * (a - b + im*(c + cᵀ))
    y = ones(ComplexF64, mds)
    y[n .!== -1] -= η[n .!== -1] # y_i = 1 - η_i for detected modes, y_i = 1 for transmitted modes
    Y = Diagonal(y)
    try
        # Compute upper-left block of A⁻¹ from block matrix inversion formula
        invC̃ = inv(C̃)
        invP = Y*conj(C̃)*Y - invC̃
        P = inv(invP)
        nPYC̃ = -P*Y*conj(C̃)
        invA_UL = Matrix{ComplexF64}(undef, 2mds, 2mds)
        copyto!(view(invA_UL, 1:mds,      1:mds     ), P)
        copyto!(view(invA_UL, 1:mds,      mds+1:2mds), nPYC̃) # -PYC̃*
        conj!(nPYC̃)
        copyto!(view(invA_UL, mds+1:2mds, 1:mds     ), nPYC̃) # -P*YC̃
        conj!(P)
        copyto!(view(invA_UL, mds+1:2mds, mds+1:2mds), P) # P*
        detA = det(C̃) * det(invP)
    catch e
        if e isa LinearAlgebra.SingularException
            # Fallback to using the full inverse of A if the block inversion fails due to singularity of C̃
            # Compute A matrix from blocks (block ordering: [α β* α* β])
            A = zeros(ComplexF64, 4mds, 4mds)

            copyto!(view(A, 1:mds,       mds+1:2mds ), -Y)
            copyto!(view(A, mds+1:2mds,  1:mds      ), -Y)
            copyto!(view(A, 1:mds,       2mds+1:3mds), I)
            copyto!(view(A, mds+1:2mds,  3mds+1:4mds), I)
            copyto!(view(A, 2mds+1:3mds, 1:mds      ), I)
            copyto!(view(A, 3mds+1:4mds, mds+1:2mds ), I)
            copyto!(view(A, 2mds+1:3mds, 2mds+1:3mds), C̃) # C̃
            conj!(C̃)
            copyto!(view(A, 3mds+1:4mds, 3mds+1:4mds), C̃) # C̃*

            invA_UL = inv(A)[1:2mds, 1:2mds] # Extract upper-left block of A⁻¹
            detA = det(A)
        else
            rethrow(e)
        end
    end

    denom = sqrt(real(detA)) * sqrt(abs(det(Γ)))
    (invA_UL, denom)
end
