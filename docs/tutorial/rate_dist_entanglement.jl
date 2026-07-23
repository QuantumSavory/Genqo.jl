using Genqo
using Gabs
using QuantumOpticsBase
using LinearAlgebra
import LinearAlgebra: tr, dot, norm
using SpecialFunctions: binomial
using Colors 
using GLMakie  
#3d stuff

#EH function formalism: 
function pmf_zP(N_I::Int, q::Float64)
    # binomial pmf for z_H (or z_V)
    return [binomial(N_I, ℓ) * q^ℓ * (1 - q)^(N_I - ℓ) for ℓ in 0:N_I]
end

function p_k_array(N_I::Int, q::Float64)
    Pz = pmf_zP(N_I, q)
    p = zeros(Float64, N_I + 1)
    for k in 0:N_I
        term = Pz[k + 1]
        tail = sum(Pz[(k + 2):end])
        p[k + 1] = term * (term + 2 * tail)
    end
    return p
end

function E_H(p, N_M::Integer)
    N_I = length(p) - 1
    k = collect(0:N_I)
    N_M_eff = clamp(Int(N_M), 0, N_I)
    if N_M_eff <= 0
        return 0.0
    end
    return sum(k[1:N_M_eff] .* p[1:N_M_eff]) + N_M_eff * sum(p[(N_M_eff + 1):end])
end

#Zalm
function zalm_Pbell_numerical(μ::Float64, ηR::Float64, ηT::Float64; engine::HybridProjectionEngine)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, modeswap(QuadBlockBasis(4)), [2,4,5,7])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5,4,6])

    η = [ηR,ηR,ηT,ηT,ηT,ηT,ηR,ηR]
    Π = projector([:,:,1,1,0,0,:,:])
    st = project(st, Π; engine, η=η)

    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    ψ⁻ = (clicks([1,0,0,1]) - clicks([0,1,1,0])) / √2
    ϕ⁺ = (clicks([1,0,1,0]) + clicks([0,1,0,1])) / √2
    ϕ⁻ = (clicks([1,0,1,0]) - clicks([0,1,0,1])) / √2

    real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻))
end
function zalm_Pload_numerical(μ::Float64, ηR::Float64, ηT::Float64; engine::HybridProjectionEngine)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, modeswap(QuadBlockBasis(4)), [2,4, 5,7])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])

    η = [ηR,ηR,ηT,ηT,ηT,ηT,ηR,ηR]
    Π_S = projector([:,:,1,1,0,0,:,:])
    Π_A0 = projector([0,0,1,1,0,0,:,:])
    Π_B0 = projector([:,:,1,1,0,0,0,0])
    Π_S0 = projector([0,0,1,1,0,0,0,0])
    st_S = project(st, Π_S; engine, η=η)
    st_A0 = project(st, Π_A0; engine, η=η)
    st_B0 = project(st, Π_B0; engine, η=η)
    st_S0 = project(st, Π_S0; engine, η=η)

    tr(st_S) - tr(st_A0) - tr(st_B0) + tr(st_S0)
end

function zalm_bell_fraction_numerical(μ::Float64, ηR::Float64, ηT::Float64; engine::HybridProjectionEngine)
    Pload = zalm_Pload_numerical.(μ, ηR, ηT; engine=engine)
    PBell = zalm_Pbell_numerical.(μ, ηR, ηT; engine=engine)
    PBell ./ Pload
end

function zalm_rate_dist_entanglement_numerical(μ::Float64, ηR::Float64, ηT::Float64, Rp::Float64, N_I::Integer, N_M; engine::HybridProjectionEngine)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, modeswap(QuadBlockBasis(4)), [2,4, 5,7])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])

    η = [ηR,ηR,ηT,ηT,ηT,ηT,ηR,ηR]
    Π = projector([:,:,1,1,0,0,:,:])
    st = project(st, Π; engine=engine, η=η)
     
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    ψ⁻ = (clicks([1,0,0,1]) - clicks([0,1,1,0])) / √2
    ϕ⁺ = (clicks([1,0,1,0]) + clicks([0,1,0,1])) / √2
    ϕ⁻ = (clicks([1,0,1,0]) - clicks([0,1,0,1])) / √2

    PHerald = 4*tr(st)
    Pbell = (real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻)) / PHerald ) * 4
    Fidelity = real(dot(ψ⁺', st, ψ⁺)) / real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻))
    q = sqrt(PHerald)
    p_array = p_k_array(Int(N_I), q)
    EH = E_H(p_array, N_M)
    return Rp * EH * Pbell * Fidelity
end

function zalm_bell_fidelity_numerical(μ::Float64,ηR::Float64, ηT::Float64; engine::HybridProjectionEngine)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, modeswap(QuadBlockBasis(4)), [2,4, 5,7])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])

    η = [ηR,ηR,ηT,ηT,ηT,ηT,ηR,ηR]
    Π = projector([:,:,1,1,0,0,:,:])
    st = project(st, Π; engine, η=η)

    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    ψ⁻ = (clicks([1,0,0,1]) - clicks([0,1,1,0])) / √2
    ϕ⁺ = (clicks([1,0,1,0]) + clicks([0,1,0,1])) / √2
    ϕ⁻ = (clicks([1,0,1,0]) - clicks([0,1,0,1])) / √2

    real(dot(ψ⁺', st, ψ⁺)) / real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻))
end

function zalm_Pbell_symbolic(μ::Float64, ηR::Float64, ηT::Float64)
    Ns  = (ηT*μ + 1) / (μ + 1)
    N′s = (ηR/Ns + (1 - ηR))^(-1)
    Pr_Bell = 2*N′s^4 * (
        2*(1 - N′s)^2
        - (2*ηR*(3*N′s^3 - 5*N′s^2 + 2*N′s)) / Ns
        + (ηR^2*(4*N′s^4 - 6*N′s^3 + 2*N′s^2)) / Ns^2
    ) + (ηR^2 * N′s^8) / (2*Ns^2)
    
    return Pr_Bell
end
function zalm_bell_fidelity_symbolic(μ::Real, ηR::Real, ηT::Real)::Real

    Ns  = (ηT*μ + 1) / (μ + 1)
    
    N′s = (ηR/Ns + (1 - ηR))^(-1)
    
    Pr_psi_minus = (N′s^4 / 2) * (
        2*(1 - N′s)^2
        - (2*ηR*(3*N′s^3 - 5*N′s^2 + 2*N′s)) / Ns
        + (ηR^2*(5*N′s^4 - 6*N′s^3 + 2*N′s^2)) / Ns^2  
    )
    Pr_Bell = 2*N′s^4 * (
        2*(1 - N′s)^2
        - (2*ηR*(3*N′s^3 - 5*N′s^2 + 2*N′s)) / Ns
        + (ηR^2*(4*N′s^4 - 6*N′s^3 + 2*N′s^2)) / Ns^2
    ) + (ηR^2 * N′s^8) / (2*Ns^2)

    return Pr_psi_minus / Pr_Bell
end
function zalm_PHeralded_symbolic(μ::Float64, ηR::Float64, ηT::Float64)::Float64
    PH = (4 * (ηT * μ)^2)/ ((ηT * μ) + 1)^6    
    return PH
end
function zalm_q_symbolic(μ::Float64, ηT::Float64)::Float64
    q = (2 * ηT * μ) / (ηT * μ + 1)^3
    return q
end
function zalm_rate_dist_entanglement_symbolic(μ::Float64, ηR::Float64, ηT::Float64, Rp, N_I::Integer, N_M::Integer)
    Pbell = zalm_Pbell_symbolic(μ, ηR, ηT)
    F = zalm_bell_fidelity_symbolic(μ, ηR, ηT)
    q = zalm_q_symbolic(μ, ηT)
    p_array = p_k_array(N_I, q)
    EH = E_H(p_array, N_M)

    return Rp * EH * F * Pbell
end  
#Chahine
function chahine_Pbell_numerical(μ::Float64, ηR::Float64, ηT::Float64; engine::HybridProjectionEngine)
    st = eprstate(QuadBlockBasis(4), asinh(√μ), 0.) ⊗ vacuumstate(QuadBlockBasis(2))
    apply!(st, modeswap(QuadBlockBasis(2)), [2,4])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])

    η = [ηT,ηT,ηR,ηR,ηR,ηR]
    Π_S = projector([1,1,:,:,:,:])

    st = project(st, Π_S; engine=engine, η=η)
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    ψ⁻ = (clicks([1,0,0,1]) - clicks([0,1,1,0])) / √2
    ϕ⁺ = (clicks([1,0,1,0]) + clicks([0,1,0,1])) / √2
    ϕ⁻ = (clicks([1,0,1,0]) - clicks([0,1,0,1])) / √2

    real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻))
end
function chahine_Pload_numerical(μ::Float64, ηR::Float64, ηT::Float64; engine::HybridProjectionEngine)
    st = eprstate(QuadBlockBasis(4), asinh(√μ), 0.) ⊗ vacuumstate(QuadBlockBasis(2))
    apply!(st, modeswap(QuadBlockBasis(2)), [2,4])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])

    η = [ηT,ηT,ηR,ηR,ηR,ηR]
    Π_S = projector([1,1,:,:,:,:])
    Π_A0 = projector([1,1,0,0,:,:])
    Π_B0 = projector([1,1,:,:,0,0])
    Π_S0 = projector([1,1,0,0,0,0])

    st_S = project(st, Π_S; engine=engine, η=η)
    st_A0 = project(st, Π_A0; engine=engine, η=η)
    st_B0 = project(st, Π_B0; engine=engine, η=η)
    st_S0 = project(st, Π_S0; engine=engine, η=η)

    tr(st_S) - tr(st_A0) - tr(st_B0) + tr(st_S0)
end
function chahine_bell_fraction_numerical(μ::Float64, ηR::Float64, ηT::Float64; engine::HybridProjectionEngine)
    Pbell = chahine_Pbell_numerical.(μ, ηR, ηT; engine=engine)
    Pload = chahine_Pload_numerical.(μ, ηR, ηT; engine=engine)
    Pbell / Pload
end
function chahine_rate_dist_entanglement_numerical(μ::Float64, ηR::Float64, ηT::Float64, Rp::Float64, N_I::Integer, N_M; engine::HybridProjectionEngine)
     st = eprstate(QuadBlockBasis(4), asinh(√μ), 0.) ⊗ vacuumstate(QuadBlockBasis(2))
    apply!(st, modeswap(QuadBlockBasis(2)), [2,4])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])

    η = [ηT,ηT,ηR,ηR,ηR,ηR]
    Π = projector([1,1,:,:,:,:])
    st = project(st, Π; engine=engine, η=η)

    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    ψ⁻ = (clicks([1,0,0,1]) - clicks([0,1,1,0])) / √2
    ϕ⁺ = (clicks([1,0,1,0]) + clicks([0,1,0,1])) / √2
    ϕ⁻ = (clicks([1,0,1,0]) - clicks([0,1,0,1])) / √2

    PHerald = tr(st)
    Pbell = real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻)) / tr(st)
    Fidelity = real(dot(ψ⁺', st, ψ⁺)) / real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻))
    q = sqrt(PHerald)
    p_array = p_k_array(Int(N_I), q)
    EH = E_H(p_array, N_M)

    Rp * EH * Fidelity * Pbell
end
function chahine_Pbell_symbolic(μ::Float64, ηR::Float64, ηT::Float64)
    Ns  = (ηT*μ + 1) / (μ + 1)
    N′s = (ηR/Ns + (1 - ηR))^(-1)
    Ñs = 2*N′s/(N′s + 1)
    
    Pr_Bell = N′s^2 * (
        3*(1 - N′s)^2 /
        - (3*ηR*N′s*(1 - 2*N′s)*(1 - N′s)) / Ns
        + (ηR^2*N′s^2*((1 - 2*N′s)^2 + 2*(1 - 3*N′s)*(1 - N′s))) / (2*Ns^2)
    )
    return Pr_Bell
end
function chahine_bell_fidelity_numerical(μ::Float64, ηR::Float64, ηT::Float64; engine::HybridProjectionEngine)
    st = eprstate(QuadBlockBasis(4), asinh(√μ), 0.) ⊗ vacuumstate(QuadBlockBasis(2))
    apply!(st, modeswap(QuadBlockBasis(2)), [2,4])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])

    η = [ηT,ηT,ηR,ηR,ηR,ηR]
    Π = projector([1,1,:,:,:,:])
    st = project(st, Π; engine=engine, η=η)

    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    ψ⁻ = (clicks([1,0,0,1]) - clicks([0,1,1,0])) / √2
    ϕ⁺ = (clicks([1,0,1,0]) + clicks([0,1,0,1])) / √2
    ϕ⁻ = (clicks([1,0,1,0]) - clicks([0,1,0,1])) / √2

    real(dot(ψ⁺', st, ψ⁺)) / real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻))
end
function chahine_PHeralded_symbolic(μ::Float64, ηR::Float64, ηT::Float64)::Float64
    PH = ((ηT * μ)^2)/ ((ηT * μ) + 1)^4    
    return PH
end
function chahine_q_symbolic(μ::Float64, ηT::Float64)::Float64
    q = (ηT * μ) / (ηT * μ + 1)^2
    return q
end
function chahine_rate_dist_entanglement_symbolic(μ::Float64, ηR::Float64, ηT::Float64, Rp, N_I::Integer, N_M::Integer)
    Pbell = chahine_Pbell_symbolic(μ, ηR, ηT)
    F = chahine_bell_fidelity_symbolic(μ, ηR, ηT)
    q = chahine_q_symbolic(μ, ηT)
    p_array = p_k_array(N_I, q)
    EH = E_H(p_array, N_M)

    return Rp * EH * F * Pbell
end 

#SPDC
function spdc_rate_dist_entanglement_numerical(μ::Float64, ηR::Float64, Rp::Float64, N_M; engine::HybridProjectionEngine)
    PBell = spdc_Pbell_numerical(μ, ηR; engine=engine)
    F = spdc_bell_fidelity_numerical(μ, ηR; engine=engine)
    return Rp * N_M * PBell * F
end
function spdc_bell_fidelity_symbolic(μ::Float64, ηR::Float64)::Float64
    G = μ + 1 
    Pr_psi_minus = (ηR^2 * μ * (3*G - 1 + ηR*(ηR - 2)*μ)) / ((ηR*(ηR - 2)*μ - 1)^4)
    Pr_Bell = (ηR^2 * μ * (3*G - 1 + ηR*(ηR - 2)*μ) + 3*(ηR*(ηR - 1)*μ)^2) / ((ηR*(ηR - 2)*μ - 1)^4)
    return Pr_psi_minus / Pr_Bell
end
function spdc_Pbell_symbolic(μ::Float64, ηR::Float64)::Float64
    G = μ + 1 
    Pr_Bell = (ηR^2 * μ * (3*G - 1 + ηR*(ηR - 2)*μ) + 3*(ηR*(ηR - 1)*μ)^2) / ((ηR*(ηR - 2)*μ - 1)^4)
    return Pr_Bell
end
# function spdc_rate_dist_entanglement_symbolic(μ::Float64, ηR::Float64, Rp::Float64, N_M::Int)
#     st = eprstate(QuadBlockBasis(4), asinh(√μ), 0.)
#     apply!(st, modeswap(QuadBlockBasis(2)), [2,4])
#     η = [ηR,ηR,ηR,ηR]
#     Π = projector([:,:,:,:])
#     st = project(st, Π; engine, η=η)

#     ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
#     ψ⁻ = (clicks([1,0,0,1]) - clicks([0,1,1,0])) / √2
#     ϕ⁺ = (clicks([1,0,1,0]) + clicks([0,1,0,1])) / √2
#     ϕ⁻ = (clicks([1,0,1,0]) - clicks([0,1,0,1])) / √2

#     PBell = real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻))
#     F = real(dot(ψ⁺', st, ψ⁺))/ real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻))

#     return Rp * N_M * PBell * F
# end

# pbp

function pbp4(μ::Float64, ηᵗ::Float64, ηᵇ::Float64, ηᵍ::Float64; engine::HybridProjectionEngine)
    sagnac = eprstate(QuadBlockBasis(4), asinh(√μ), 0.)
    apply!(sagnac, modeswap(QuadBlockBasis(2)), [1,3])
    st = sagnac ⊗ sagnac ⊗ sagnac ⊗ sagnac

    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])
    apply!(st, greenmachine(QuadBlockBasis(4), 4), [7,8, 9,10])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [11,13, 12,14])

    η = [ηᵗ,ηᵗ,ηᵇ,ηᵇ,ηᵇ,ηᵇ,ηᵍ,ηᵍ,ηᵍ,ηᵍ,ηᵇ,ηᵇ,ηᵇ,ηᵇ,ηᵗ,ηᵗ]
    Π = projector([:,:,1,0,0,1,1,0,0,1,1,0,0,1,:,:])
    project(st, Π; engine, η=η)
end

function pbp4_rate_dist_entanglement(μ::Float64, ηR::Float64, ηT::Float64, Rp, N_I::Integer, N_M::Integer; engine::HybridProjectionEngine)
    st = pbp4.(μ, ηR, ηT, 1.0; engine = engine)
    
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    ψ⁻ = (clicks([1,0,0,1]) - clicks([0,1,1,0])) / √2
    ϕ⁺ = (clicks([1,0,1,0]) + clicks([0,1,0,1])) / √2
    ϕ⁻ = (clicks([1,0,1,0]) - clicks([0,1,0,1])) / √2

    PHerald = tr(st)
    Pbell = real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻)) / tr(st)
    Fidelity = real(dot(ψ⁺', st, ψ⁺)) / real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻))
    q = sqrt(PHerald)
    p_array = p_k_array(Int(N_I), q)
    EH = E_H(p_array, N_M)

    Rp * EH * Fidelity * Pbell
end

function plot_rate_dist_entanglement_with_Nm_3D()
    zalm_engine    = HybridProjectionEngine(8)
    chahine_engine = HybridProjectionEngine(6)
    spdc_engine = HybridProjectionEngine(4)
    pbp4_engine = HybridProjectionEngine(16)

    ηR = 0.01
    ηT_vals = logrange(0.7, 1, 100)
    Rp = 10e9
    N_I = 10
    N_M = 1
    μ_vals = logrange(1e-4, 10, 100)

    zalm_rates = [zalm_rate_dist_entanglement_numerical(μ, ηR, ηT, Rp, N_I, N_M; engine=zalm_engine)
                  for μ in μ_vals, ηT in ηT_vals]
    chahine_rates = [chahine_rate_dist_entanglement_numerical(μ, ηR, ηT, Rp, N_I, N_M; engine=chahine_engine)
                     for μ in μ_vals, ηT in ηT_vals]
    # spdc_rates = [spdc_rate_dist_entanglement_numerical(μ, ηR, Rp, N_M; engine=spdc_engine) for μ in μ_vals, ηT in ηT_vals]
    #spcd_rates_symbolic = [spdc_rate_dist_entanglement_symbolic(μ, ηR, Rp, N_M) for μ in μ_vals, ηT in ηT_vals]

    #pbp4_rates = [pbp4_rate_dist_entanglement(μ, ηR, ηT, Rp, N_I, N_M; engine=pbp4_engine) for μ in μ_vals, ηT in ηT_vals]
    zmax = maximum(filter(!isnan, vcat(vec(zalm_rates), vec(chahine_rates))))
     
    fig = Figure(size = (1000, 700))

    ax = Axis3(fig[1, 1];
        xlabel = "Mean photon number μ",
        ylabel = "Source loss ηT",
        zlabel = "Rate (bits/s)",
        title  = "Rate of distributed entanglement vs μ and ηT (NI = $(N_I), NM = $(N_M))",
        azimuth   = 35 * π/180,
        elevation = 25 * π/180,
        limits = (nothing, nothing, (0, zmax)),
        xticks = LinearTicks(10),
        yticks = LinearTicks(6),
        zticks = LinearTicks(5),
    )

    zalm_color    = fill(RGBAf(0.18, 0.545, 0.341, 0.8), size(zalm_rates))    
    chahine_color = fill(RGBAf(1.0, 0.271, 0.0, 0.8), size(chahine_rates))
    spdc_color = fill(RGBAf(0.85, 0.74, 0.0, 0.4))

    surfaces = [
        (name = "ZALM",    data = zalm_rates,    color = RGBAf(0.18, 0.545, 0.341, 0.8)),
        (name = "Chahine", data = chahine_rates, color = RGBAf(1.0,  0.271, 0.0,   0.8)),
       #(name = "PBP4", data  = pbp4_rates, color = RGBAf(0.85, 0.74,  0.0,   0.4) )
        # (name = "SPDC",    data = spdc_rates,    color = RGBAf(0.85, 0.74,  0.0,   0.4)),
        #(name = "SPDC symbolic", data = spcd_rates_symbolic,  color = RGBAf(1.0,  0.271, 0.0,   0.8))
    ]

    for s in surfaces
        surface!(ax, μ_vals, ηT_vals, s.data;
            color = fill(s.color, size(s.data)), shading = NoShading)
    end  

    Legend(fig[2, 1],
        [PolyElement(color = s.color) for s in surfaces],
        [s.name for s in surfaces];
        orientation = :horizontal,
    )
    rowgap!(fig.layout, 30)

    fig
end

function plot_fidelity_with_Nm_3D()
    zalm_engine = HybridProjectionEngine(8)
    chahine_engine = HybridProjectionEngine(6)
    spdc_engine = HybridProjectionEngine(4)
    pbp4_engine = HybridProjectionEngine(16)

    ηR = 0.01
    ηT_vals = logrange(0.7, 1, 100)
    Rp = 10e9
    N_I = 10
    N_M = 1
    μ_vals = logrange(1e-4, 10, 100)

    zalm_rates = [zalm_bell_fidelity_numerical(μ, ηR, ηT; engine=zalm_engine)
                  for μ in μ_vals, ηT in ηT_vals]
    chahine_rates = [chahine_bell_fidelity_numerical(μ, ηR, ηT; engine=chahine_engine)
                     for μ in μ_vals, ηT in ηT_vals]
    # spdc_rates = [spdc_rate_dist_entanglement_numerical(μ, ηR, Rp, N_M; engine=spdc_engine) for μ in μ_vals, ηT in ηT_vals]
    #spcd_rates_symbolic = [spdc_rate_dist_entanglement_symbolic(μ, ηR, Rp, N_M) for μ in μ_vals, ηT in ηT_vals]

    #pbp4_rates = [pbp4_rate_dist_entanglement(μ, ηR, ηT, Rp, N_I, N_M; engine=pbp4_engine) for μ in μ_vals, ηT in ηT_vals]
    zmax = maximum(filter(!isnan, vcat(vec(zalm_rates), vec(chahine_rates))))

    fig = Figure(size = (1000, 700))

    ax = Axis3(fig[1, 1];
        xlabel = "Mean photon number μ",
        ylabel = "Transmission loss ηT",
        zlabel = "Fidelity",
        title  = "Bell fidelity vs μ and ηT (NI = $(N_I), NM = $(N_M))",
        azimuth   = 35 * π/180,
        elevation = 25 * π/180,
        limits = (nothing, nothing, (0, zmax)),
        xticks = LinearTicks(10),
        yticks = LinearTicks(6),
        zticks = LinearTicks(5),
    )

    zalm_color    = fill(RGBAf(0.18, 0.545, 0.341, 0.8), size(zalm_rates))    
    chahine_color = fill(RGBAf(1.0, 0.271, 0.0, 0.8), size(chahine_rates))
    spdc_color = fill(RGBAf(0.85, 0.74, 0.0, 0.4))

    surfaces = [
        (name = "ZALM",    data = zalm_rates,    color = RGBAf(0.18, 0.545, 0.341, 0.8)),
        (name = "Chahine", data = chahine_rates, color = RGBAf(1.0,  0.271, 0.0,   0.8)),
       #(name = "PBP4", data  = pbp4_rates, color = RGBAf(0.85, 0.74,  0.0,   0.4) )
        # (name = "SPDC",    data = spdc_rates,    color = RGBAf(0.85, 0.74,  0.0,   0.4)),
        #(name = "SPDC symbolic", data = spcd_rates_symbolic,  color = RGBAf(1.0,  0.271, 0.0,   0.8))
    ]

    for s in surfaces
        surface!(ax, μ_vals, ηT_vals, s.data;
            color = fill(s.color, size(s.data)), shading = NoShading)
    end

    Legend(fig[2, 1],
        [PolyElement(color = s.color) for s in surfaces],
        [s.name for s in surfaces];
        orientation = :horizontal,
    )

    rowgap!(fig.layout, 30) 

    fig
end

function plot_bell_fraction_with_Nm_3D()
    zalm_engine    = HybridProjectionEngine(8)
    chahine_engine = HybridProjectionEngine(6)
    # spdc_engine = HybridProjectionEngine(4)
    # pbp4_engine = HybridProjectionEngine(16)

    ηR = 0.01
    ηT_vals = logrange(0.7, 1, 100)
    Rp = 10e9
    N_I = 10
    N_M = 1
    μ_vals = logrange(1e-4, 10, 100)

    zalm_rates = [zalm_bell_fraction_numerical(μ, ηR, ηT; engine=zalm_engine)
                  for μ in μ_vals, ηT in ηT_vals]
    chahine_rates = [chahine_bell_fraction_numerical(μ, ηR, ηT; engine=chahine_engine)
                     for μ in μ_vals, ηT in ηT_vals]
    # spdc_rates = [spdc_rate_dist_entanglement_numerical(μ, ηR, Rp, N_M; engine=spdc_engine) for μ in μ_vals, ηT in ηT_vals]
    #spcd_rates_symbolic = [spdc_rate_dist_entanglement_symbolic(μ, ηR, Rp, N_M) for μ in μ_vals, ηT in ηT_vals]
    #pbp4_rates = [pbp4_rate_dist_entanglement(μ, ηR, ηT, Rp, N_I, N_M; engine=pbp4_engine) for μ in μ_vals, ηT in ηT_vals]
    zmax = maximum(filter(!isnan, vcat(vec(zalm_rates), vec(chahine_rates))))
    zmin = minimum(filter(!isnan, vcat(vec(zalm_rates), vec(chahine_rates))))
    fig = Figure(size = (1000, 700))

    ax = Axis3(fig[1, 1];
        xlabel = "Mean photon number μ",
        ylabel = "Transmission loss ηT",
        zlabel = "Bell Fraction",
        title  = "Bell purity vs μ and ηT (NI = $(N_I), NM = $(N_M))",
        azimuth   = 35 * π/180,
        elevation = 25 * π/180,
        limits = (nothing, nothing, (zmin, zmax)),
        xticks = LinearTicks(10),
        yticks = LinearTicks(6),
        zticks = LinearTicks(5),
    )

    zalm_color    = fill(RGBAf(0.18, 0.545, 0.341, 0.8), size(zalm_rates))    
    chahine_color = fill(RGBAf(1.0, 0.271, 0.0, 0.8), size(chahine_rates))
    spdc_color = fill(RGBAf(0.85, 0.74, 0.0, 0.4))

    surfaces = [
        (name = "ZALM",    data = zalm_rates,    color = RGBAf(0.18, 0.545, 0.341, 0.8)),
        (name = "Chahine", data = chahine_rates, color = RGBAf(1.0,  0.271, 0.0,   0.8)),
        #(name = "PBP4", data  = pbp4_rates, color = RGBAf(0.85, 0.74,  0.0,   0.4) )
        # (name = "SPDC",    data = spdc_rates,    color = RGBAf(0.85, 0.74,  0.0,   0.4)),
        # (name = "SPDC symbolic", data = spcd_rates_symbolic,  color = RGBAf(1.0,  0.271, 0.0,   0.8)),
    ]

    for s in surfaces
        surface!(ax, μ_vals, ηT_vals, s.data;
            color = fill(s.color, size(s.data)), shading = NoShading)
    end

    Legend(fig[2, 1],
        [PolyElement(color = s.color) for s in surfaces],
        [s.name for s in surfaces];
        orientation = :horizontal,
    )
    rowgap!(fig.layout, 30)
    fig
end

function plot_metrics_comparison_with_3D()
    zalm_engine    = HybridProjectionEngine(8)
    chahine_engine = HybridProjectionEngine(6)

    ηR = 0.01
    ηT_vals = logrange(0.7, 1, 100)
    Rp = 10e9
    N_I = 10
    N_M = 1
    μ_vals = logrange(1e-4, 10, 100)

    # One entry per metric/panel: name, how to compute each engine's values, z label
    metrics = [
        (
            name   = "Rate",
            zlabel = "Rate (bits/s)",
            zalm_fn    = (μ, ηT) -> zalm_rate_dist_entanglement_numerical(μ, ηR, ηT, Rp, N_I, N_M; engine=zalm_engine),
            chahine_fn = (μ, ηT) -> chahine_rate_dist_entanglement_numerical(μ, ηR, ηT, Rp, N_I, N_M; engine=chahine_engine),
        ),
        (
            name   = "Bell fraction",
            zlabel = "Bell fraction",
            zalm_fn    = (μ, ηT) -> zalm_bell_fraction_numerical(μ, ηR, ηT; engine=zalm_engine),
            chahine_fn = (μ, ηT) -> chahine_bell_fraction_numerical(μ, ηR, ηT; engine=chahine_engine),
        ),
        (
            name   = "Fidelity",
            zlabel = "Fidelity",
            zalm_fn    = (μ, ηT) -> zalm_bell_fidelity_numerical(μ, ηR, ηT; engine=zalm_engine),
            chahine_fn = (μ, ηT) -> chahine_bell_fidelity_numerical(μ, ηR, ηT; engine=chahine_engine),
        ),
    ]

    fig = Figure(size = (1800, 700))

    for (i, m) in enumerate(metrics)
        zalm_data    = [m.zalm_fn(μ, ηT)    for μ in μ_vals, ηT in ηT_vals]
        chahine_data = [m.chahine_fn(μ, ηT) for μ in μ_vals, ηT in ηT_vals]

        zmax = maximum(filter(!isnan, vcat(vec(zalm_data), vec(chahine_data))))

        ax = Axis3(fig[1, i];
            xlabel = "Mean photon number μ",
            ylabel = "Source loss ηT",
            zlabel = m.zlabel,
            title  = "$(m.name) vs μ and ηT (NI = $(N_I), NM = $(N_M))",
            azimuth   = 35 * π/180,
            elevation = 25 * π/180,
            limits = (nothing, nothing, (0, zmax)),
            xticks = LinearTicks(10),
            yticks = LinearTicks(6),
            zticks = LinearTicks(5),
        )

        surfaces = [
            (name = "ZALM",    data = zalm_data,    color = RGBAf(0.18, 0.545, 0.341, 0.8)),
            (name = "Chahine", data = chahine_data, color = RGBAf(1.0,  0.271, 0.0,   0.8)),
        ]

        for s in surfaces
            surface!(ax, μ_vals, ηT_vals, s.data;
                color = fill(s.color, size(s.data)), shading = NoShading)
        end

        Legend(fig[2, i],
            [PolyElement(color = s.color) for s in surfaces],
            [s.name for s in surfaces];
            orientation = :horizontal,
        )
    end

    fig
end

try
    fig1 = plot_rate_dist_entanglement_with_Nm_3D()
    fig2 = plot_fidelity_with_Nm_3D()
    fig3 = plot_bell_fraction_with_Nm_3D()
    display(GLMakie.Screen(), fig1)
    display(GLMakie.Screen(), fig2)
    display(GLMakie.Screen(), fig3)
    # fig = plot_metrics_comparison_with_3D()
    # display(fig)
catch e
    showerror(stdout, e, catch_backtrace())
end