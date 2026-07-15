using Genqo
using Gabs

using Plots


# ZALM model

function zalm2(μ::Float64, ηᵗ::Float64, ηᵇ::Float64; engine::HybridProjectionEngine)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, modeswap(QuadBlockBasis(4)), [2,4, 5,7])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])

    η = [ηᵗ,ηᵗ,ηᵇ,ηᵇ,ηᵇ,ηᵇ,ηᵗ,ηᵗ]
    Π = projector([:,:,1,1,0,0,:,:])
    project(st, Π; engine, η=η)
end

## Probability of generation
function plot_zalm2_probability()
    engine = HybridProjectionEngine(8)
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    states = zalm2.(μ, 1., η'; engine=engine)

    Pgen = tr.(states)
    Pgen_ground = zalm.probability_success.(μ, 1., 1., η', 0)

    plot(μ, Pgen_ground, label="Genqo v1 (ground truth)", xscale=:log10, yscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Probability of Generation", legend=:bottomright, color=[1 2 3 4])
    plot!(μ, Pgen, label="Genqo v2", linestyle=:dash, color=[:blue :orange :green :red])
    plot!(μ, μ.^2 ./ (μ.+1).^6, label="Analytical", linestyle=:dot, color=:black)
end
@time plot_zalm2_probability()

## Bell-state fraction
function plot_zalm2_Bell_state_fraction()
    engine = HybridProjectionEngine(8)
    μ = logrange(1e-4, 1.5, 100)

    function zalm2_Pload(μ::Float64, ηR::Float64, ηT::Float64; engine::HybridProjectionEngine)
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

    function zalm2_PBell(μ::Float64, ηR::Float64, ηT::Float64; engine::HybridProjectionEngine)
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

        real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻))
    end

    Pload = zalm2_Pload.(μ, 1., 0.9; engine=engine)
    PBell = zalm2_PBell.(μ, 1., 0.9; engine=engine)
    B = PBell ./ Pload
    
    plot(μ, B, label="ZALM", xlabel="Mean Photon Number Per Mode", ylabel="Bell-state Fraction", legend=:bottomright, color=1)
end
@time plot_zalm2_Bell_state_fraction()

## Fidelity
function plot_zalm2_fidelity()
    engine = HybridProjectionEngine(8)
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    states = zalm2.(μ, 1., η'; engine=engine)
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    F = similar(states, Float64)
    Threads.@threads for I in CartesianIndices(states)
        F[I] = real(dot(ψ⁺', states[I], ψ⁺)) / tr(states[I])
    end
    F_ground = zalm.fidelity.(μ, 1., 1., η')

    plot(μ, F_ground, label="Genqo v1 (ground truth)", ylim=[0,1], xscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", legend=:topleft, color=[1 2 3 4])
    plot!(μ, F, label="Genqo v2", linestyle=:dash, color=[1 2 3 4])
end
@time plot_zalm2_fidelity()
