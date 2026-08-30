using Nemo
using LinearAlgebra
import LinearAlgebra: tr, dot, norm
using LazyArrays
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
"""
$(TYPEDEF)

Supertype for photon-number ("click") states: superpositions of click patterns over a
fixed number of modes. See [`ClickStateKet`](@ref) and [`ClickStateBra`](@ref).
"""
abstract type AbstractClickState end

"""
$(TYPEDEF)

A superposition of photon-number patterns over a fixed number of modes. Each row of
`clicks` is one pattern of non-negative photon numbers and `coefs[i]` is its amplitude.

Construct single patterns with [`clicks`](@ref) and combine them with `+`, `-`, and scalar
`*` and `/`. `adjoint` gives the corresponding [`ClickStateBra`](@ref). Click states are the
targets of [`fidelity`](@ref) and the bra and ket arguments of [`dot`](@ref).

# Fields
$(TYPEDFIELDS)
"""
struct ClickStateKet <: AbstractClickState
    coefs::Vector{ComplexF64}
    clicks::Array{Int,2}
    function ClickStateKet(coefs::Vector{ComplexF64}, clicks::Array{Int,2})
        length(coefs) == size(clicks, 1) || throw(ArgumentError("Length of coefficients must match number of click patterns"))
        all(clicks .≥ 0) || throw(ArgumentError("Click patterns must be non-negative integers"))
        new(coefs, clicks)
    end
end
"""
$(TYPEDSIGNATURES)

The single-pattern click state with `cl[i]` photons in mode `i` and unit amplitude.

Superpositions are built by combining these, e.g. `(clicks([1,0]) + clicks([0,1]))/sqrt(2)`.
"""
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

"""
$(TYPEDEF)

The dual of a [`ClickStateKet`](@ref), normally obtained as `clicks([...])'`. Supports the
same arithmetic, and `adjoint` converts back to a ket.

# Fields
$(TYPEDFIELDS)
"""
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

"""
$(TYPEDSIGNATURES)

Overlap of two click states. Patterns are orthonormal, so this is the sum of amplitude
products over the patterns the two states share.
"""
dot(bra::ClickStateBra, ket::ClickStateKet) = sum(bcf * kcf * prod(bcl .== kcl) for (bcf, bcl) in zip(bra.coefs, eachrow(bra.clicks)), (kcf, kcl) in zip(ket.coefs, eachrow(ket.clicks)))
"""
$(TYPEDSIGNATURES)

Euclidean norm of a click state. Click states are not normalized on construction, so
normalize a target before passing it to [`fidelity`](@ref).
"""
norm(cs::ClickStateKet) = √dot(cs', cs)
norm(cs::ClickStateBra) = √dot(cs, cs')


"""
$(TYPEDEF)

Supertype for operators diagonal in the photon-number basis. See [`ClickProjector`](@ref).
"""
abstract type AbstractClickOperator end

_replace_colon(::Any) = throw(ArgumentError("Photon number clicks must be Int or Colon"))
_replace_colon(i::Int) = i
_replace_colon(::Colon) = -1

"""
$(TYPEDEF)

A Fock-space projector describing a detection outcome. A numeric entry represents a Fock
state in that mode. An entry of `-1` marks a mode that is traced out rather than detected.

`ClickProjector`s can be summed with `+` to describe projection onto a superposition of
click patterns, such as a parity measurement. Only `+` is defined because scalar
multiplication would not preserve the idempotent property of a projector.

# Fields
$(TYPEDFIELDS)
"""
struct ClickProjector <: AbstractClickOperator
    clicks::Array{Int,2}
    function ClickProjector(clicks::Array{Int,2})
        all(clicks .≥ -1) || throw(ArgumentError("Click projector must be non-negative integers or -1 for traceout"))
        size(clicks, 1) == 1 || allequal(outcome .== -1 for outcome in eachrow(clicks)) || throw(ArgumentError("Placement of traceout modes must be consistent across all click pattern terms"))
        new(clicks)
    end
end
"""
$(TYPEDSIGNATURES)

The detection outcome measuring `clicks[i]` photons in mode `i`. A `Colon` (`:`) marks a
mode that is traced out instead of detected, so `projector([1, :])` detects one photon in
mode 1 and leaves mode 2 free.

Detected outcomes are currently restricted to 0 or 1 photons per mode. Sum projectors with
`+` to describe projection onto a superposition of click patterns, such as a parity
measurement.
"""
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
"""
$(TYPEDSIGNATURES)

Indices of the modes left free (traced out) by a [`ClickProjector`](@ref) or a
[`ProjectedPureGaussianState`](@ref). These modes survive into the density matrix produced
by [`to_fock`](@ref).
"""
freemodes(cp::ClickProjector) = findall(==(-1), cp.clicks[1,:])
"""
$(TYPEDSIGNATURES)

Number of free (traced-out) modes, i.e. `length(freemodes(x))`.
"""
nfreemodes(cp::ClickProjector) = count(==(-1), cp.clicks[1,:])


"""
$(TYPEDEF)

Supertype for the reusable workspaces that evaluate projections. See
[`HybridProjectionEngine`](@ref).
"""
abstract type AbstractProjectionEngine end

"""
$(TYPEDEF)

Reusable workspace for projections on an `mds`-mode circuit.

The engine holds the symbolic phase-space variables for the circuit and memoizes compiled
moment polynomials ([`WTerms`](@ref)) keyed by click pattern, so repeated calls that share
patterns skip recompilation. One engine may be shared across threads.

Every entry point ([`tr`](@ref), [`dot`](@ref), [`fidelity`](@ref), [`to_fock`](@ref),
[`duankimble`](@ref), [`emissiveload`](@ref)) takes an `engine` keyword and otherwise falls
back to [`get_default_engine`](@ref).
"""
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
"""
$(TYPEDSIGNATURES)

The symbolic alpha and beta-conjugate phase-space variables of `engine`.

Most moments involve only these two, which lets the Wick contraction use the upper-left
block of the inverse kernel rather than the full matrix. See
[`get_phase_space_generators_full`](@ref).
"""
get_phase_space_generators_half(engine::HybridProjectionEngine) = (engine.phase_space_generators_half[:,i] for i in 1:2)
"""
$(TYPEDSIGNATURES)

All four symbolic phase-space variable sets of `engine`, for moments that need the full
inverse kernel.
"""
get_phase_space_generators_full(engine::HybridProjectionEngine) = (engine.phase_space_generators_full[:,i] for i in 1:4)
const _default_engines = Dict{Int, HybridProjectionEngine}()
const _default_engines_lock = ReentrantLock() # for multithreading safety
"""
$(TYPEDSIGNATURES)

The process-wide [`HybridProjectionEngine`](@ref) for `mds` modes, constructed on first use
and cached thereafter. Access is thread-safe.
"""
function get_default_engine(mds::Int)
    @lock _default_engines_lock get!(_default_engines, mds) do
        HybridProjectionEngine(mds)
    end
end

"""
$(TYPEDEF)

Supertype for states carrying a pending detection outcome.
"""
abstract type AbstractProjectedState end
"""
$(TYPEDEF)

Supertype for pure Gaussian states carrying a pending detection outcome. See
[`ProjectedPureGaussianState`](@ref).
"""
abstract type AbstractProjectedPureGaussianState end

"""
$(TYPEDEF)

A pure Gaussian state together with a detection outcome and per-mode detector
efficiencies. Build one with [`project`](@ref) rather than calling this constructor.

Nothing is evaluated on construction: the object records what was measured, and the
quantity of interest is then extracted with [`tr`](@ref) for the success probability,
[`dot`](@ref) for density-matrix elements, [`fidelity`](@ref), or [`to_fock`](@ref).

# Fields
$(TYPEDFIELDS)
"""
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
"""
$(TYPEDSIGNATURES)

Apply the detection outcome `proj` to the pure Gaussian state `st`, with per-mode detector
efficiency given by the keyword `η`, returning a [`ProjectedPureGaussianState`](@ref).

`st` must be pure with zero displacement, and `st`, `proj` and the efficiency vector must
agree on the number of modes. Nothing is computed here; pass the result to [`tr`](@ref),
[`dot`](@ref), [`fidelity`](@ref) or [`to_fock`](@ref).
"""
project(st::GaussianState, proj::ClickProjector; η::Vector{Float64}=ones(nmodes(st))) = ProjectedPureGaussianState(st, proj, η)

"""
$(TYPEDSIGNATURES)

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

"""
$(TYPEDSIGNATURES)

The unnormalized density-matrix element of the projected state between `bra` and `ket`,
which range over the free modes of the projection.
"""
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
        invA, denom = _invA_UL(σ, η, nf) # A_F is same as A_P, just with no traceout modes

        C = zero(WTerms{Tuple{}}) # Start with empty WTerms
        for (bcf, bcl) in zip(bra.coefs, eachrow(bra.clicks)), (kcf, kcl) in zip(ket.coefs, eachrow(ket.clicks))
            u, v = copy(n), copy(n)
            u[n .== -1] .= bcl # Build full click patterns for the bra and ket, filling in the undetected modes with the click patterns from the bra and ket
            v[n .== -1] .= kcl
            Cij = @lock engine.C_poly_cache_lock get!(engine.C_poly_cache, (u, v)) do
                prod(α .^ u) * prod(βc .^ v) |> extract_W_terms
            end
            ηweight = prod(η.^((u.+v)./2) ./ (sqrt.(factorial.(u) .* factorial.(v)))) # √η per detected photon on each of the bra and ket sides
            C += bcf * kcf * ηweight * Cij
        end

        Dot += W(C, invA) / denom
    end
    Dot
end
"""
$(TYPEDSIGNATURES)

Computes the fidelity of the heralded state against a click-state `target`, normalized by [`tr`](@ref).

`target` may be a [`ClickStateKet`](@ref) or [`ClickStateBra`](@ref) over the free modes of
the projection, and should be normalized.
"""
fidelity(target::ClickStateKet, projected_state::ProjectedPureGaussianState; engine::HybridProjectionEngine=get_default_engine(nmodes(projected_state))) = dot(target', projected_state, target; engine) / tr(projected_state; engine)
fidelity(target::ClickStateBra, projected_state::ProjectedPureGaussianState; engine::HybridProjectionEngine=get_default_engine(nmodes(projected_state))) = dot(target, projected_state, target'; engine) / tr(projected_state; engine)

"""
$(TYPEDSIGNATURES)

Computes the unnormalized Fock-basis photon-photon density matrix of a projected pure Gaussian state.
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
$(TYPEDSIGNATURES)

Models Duan-Kimble loading into a spin quantum memory.

The returned operator is backed by a [`LazyDensityMatrix`](@ref), which builds and contracts only
the entries that are read.
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

    # Shared by every density-matrix entry, so computed once here rather than per entry.
    shared = [
        let
            nf = max.(n, 0) # Filter out -1 (traceout) modes
            invA, denom = _invA_UL(σ, η, nf)
            ηweight = prod(η .^ nf) # √η per detected photon, matching dot()'s per-click η factor
            (nf, invA, ηweight, denom)
        end
        for n in eachrow(proj.clicks)
    ]

    normalization = 2^sum(d) * 4 # TODO: is this 4 actually 2^n_mem?
    function element(i::Int, j::Int)
        r, s = i - 1, j - 1
        ρᵣₛ = zero(ComplexF64)
        for (nf, invA, ηweight, denom) in shared
            C = @lock engine.C_poly_cache_ext_lock get!(engine.C_poly_cache_ext, (nf, η, d, r, s)) do
                _duankimble_poly(α, βc, η, nf, d, modes, r, s)
            end
            ρᵣₛ += W(C, invA) * ηweight / denom
        end
        ρᵣₛ / normalization
    end

    Operator(reduce(⊗, fill(SpinBasis(1//2), n_mem)), LazyDensityMatrix(element, M))
end

# Moment polynomial for the (r, s) entry of a Duan-Kimble spin density matrix.
function _duankimble_poly(α, βc, η, nf, d::Vector{Int}, modes::Vector{Tuple{Int,Int}}, r::Int, s::Int)
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

"""
$(TYPEDSIGNATURES)

Models emissive loading into a quantum memory

The returned operator is backed by a [`LazyDensityMatrix`](@ref), which builds and contracts only
the entries that are read.
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

    ηp = vcat(η, ones(Float64, 2n_mem)) # Pad η to include the emission modes, assuming no loss

    # Shared by every density-matrix entry, so computed once here rather than per entry.
    patterns = [let nfp = vcat(max.(n, 0), zeros(Int64, 2n_mem)) # Filter out -1 (traceout) modes
            invA, denom = _invA(σ, ηp, nfp)
            (nfp, invA, denom)
        end for n in eachrow(proj.clicks)]

    normalization = 4^n_mem * 2
    function element(i::Int, j::Int)
        r, s = i - 1, j - 1
        ρᵣₛ = zero(ComplexF64)
        for (nfp, invA, denom) in patterns
            C = _emissiveload_poly(α, βc, αc, β, ηp, nfp, load, emit, r, s)
            ρᵣₛ += W(C, invA) / denom
        end
        ρᵣₛ / normalization
    end

    Operator(reduce(⊗, fill(SpinBasis(1//2), n_mem)), LazyDensityMatrix(element, M))
end

# Moment polynomial for the (r, s) entry of an emissive-loading spin density matrix.
function _emissiveload_poly(α, βc, αc, β, ηp, nfp, load::Vector{Tuple{Int,Int}}, emit::Vector{Tuple{Int,Int}}, r::Int, s::Int)
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
    extract_W_terms(C)
end


function _invA(σ::Matrix{Float64}, η::Vector{Float64}, n::AbstractArray{Int})::Tuple{Matrix{ComplexF64}, Float64}
    size(σ, 1) == size(σ, 2) || throw(ArgumentError("Covariance matrix must be square"))
    mds = size(σ, 1) ÷ 2
    length(n) == length(η) == mds || throw(ArgumentError("Length of n and η must match number of modes in covariance matrix"))

    Γ = σ + 0.5*I
    Γinv = inv(Γ) # TODO: invert more efficiently using Cholesky factorization

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

    # Compute A⁻¹ from block matrix inversion formula
    # A⁻¹ = [
    #   -GC̃          GC̃YC̃*      G      -GC̃Y
    #    G*C̃*YC̃     -G*C̃*      -G*C̃*Y   G*
    #    I+YG*C̃*YC̃  -YG*C̃*     -YG*C̃*Y  YG*
    #   -YGC̃         I+YGC̃YC̃*   YG     -YGC̃Y
    # ]
    invG = I - C̃*Y*conj(C̃)*Y
    G = inv(invG)
    invA = Matrix{ComplexF64}(undef, 4mds, 4mds)

    # Upper-left block
    invA[1:mds, 1:mds] = -G*C̃
    @views copyto!(invA[mds+1:2mds, mds+1:2mds], invA[1:mds, 1:mds])
    @views conj!(invA[mds+1:2mds, mds+1:2mds])
    invA[1:mds, mds+1:2mds] =  G*C̃*Y*conj(C̃)
    @views copyto!(invA[mds+1:2mds, 1:mds], invA[1:mds, mds+1:2mds])
    @views conj!(invA[1:mds, mds+1:2mds])

    # Upper-right block
    invA[1:mds, 2mds+1:3mds] = G
    @views copyto!(invA[mds+1:2mds, 3mds+1:4mds], invA[1:mds, 2mds+1:3mds])
    @views conj!(invA[mds+1:2mds, 3mds+1:4mds])
    invA[1:mds, 3mds+1:4mds] = -G*C̃*Y
    @views copyto!(invA[mds+1:2mds, 2mds+1:3mds], invA[1:mds, 3mds+1:4mds])
    @views conj!(invA[mds+1:2mds, 2mds+1:3mds])

    # Lower-left block
    invA[3mds+1:4mds, mds+1:2mds] = I + Y*G*C̃*Y*conj(C̃)
    @views copyto!(invA[2mds+1:3mds, 1:mds], invA[3mds+1:4mds, mds+1:2mds])
    @views conj!(invA[2mds+1:3mds, 1:mds])
    invA[3mds+1:4mds, 1:mds] = -Y*G*C̃
    @views copyto!(invA[2mds+1:3mds, mds+1:2mds], invA[3mds+1:4mds, 1:mds])
    @views conj!(invA[2mds+1:3mds, mds+1:2mds])

    # Lower-right block
    invA[3mds+1:4mds, 3mds+1:4mds] = -Y*G*C̃*Y
    @views copyto!(invA[2mds+1:3mds, 2mds+1:3mds], invA[3mds+1:4mds, 3mds+1:4mds])
    @views conj!(invA[2mds+1:3mds, 2mds+1:3mds])
    invA[2mds+1:3mds, 3mds+1:4mds] = Y*G
    @views copyto!(invA[3mds+1:4mds, 2mds+1:3mds], invA[2mds+1:3mds, 3mds+1:4mds])
    @views conj!(invA[3mds+1:4mds, 2mds+1:3mds])

    detA = det(invG)

    denom = sqrt(real(detA) * abs(det(Γ)))
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

    # Compute upper-left block of A⁻¹ from block matrix inversion formula
    # A⁻¹_UL = [
    #   -GC̃     GC̃YC̃*
    #   G*C̃*YC̃  -G*C̃*
    # ]
    invG = I - C̃*Y*conj(C̃)*Y
    G = inv(invG)
    invA_UL = Matrix{ComplexF64}(undef, 2mds, 2mds)
    invA_UL[1:mds, 1:mds] = -G*C̃
    @views copyto!(invA_UL[mds+1:2mds, mds+1:2mds], invA_UL[1:mds, 1:mds])
    @views conj!(invA_UL[mds+1:2mds, mds+1:2mds])
    invA_UL[1:mds, mds+1:2mds] = G*C̃*Y*conj(C̃)
    @views copyto!(invA_UL[mds+1:2mds, 1:mds], invA_UL[1:mds, mds+1:2mds])
    @views conj!(invA_UL[1:mds, mds+1:2mds])

    detA = det(invG)

    denom = sqrt(real(detA) * abs(det(Γ)))
    (invA_UL, denom)
end
