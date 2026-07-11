using Genqo
using Gabs

using Plots


# ZALM model

function zalm2(μ::Float64, ηᵗ::Float64, ηᵈ::Float64; engine::HybridProjectionEngine)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, modeswap(QuadBlockBasis(4)), [2,4, 5,7])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])

    η = [ηᵗ,ηᵗ,ηᵈ,ηᵈ,ηᵈ,ηᵈ,ηᵗ,ηᵗ]
    Π = projector([:,:,1,1,0,0,:,:])
    project(st, Π; engine, η=η)
end

## Probability of generation
function plot_zalm2_probability()
    engine = HybridProjectionEngine(8)
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    states = zalm2.(μ, η', η'; engine=engine)

    Pgen = tr.(states)
    Pgen_ground = zalm.probability_success.(μ, η', 1, η', 0)

    plot(μ, Pgen_ground, label="Genqo v1 (ground truth)", xscale=:log10, yscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Probability of Success", legend=:bottomright, color=[1 2 3 4])
    plot!(μ, Pgen, label="Genqo v2", linestyle=:dash, color=[:blue :orange :green :red])
    plot!(μ, μ.^2 ./ (μ.+1).^6, label="Analytical", linestyle=:dot, color=:black)
end
@time plot_zalm2_probability()

## Fidelity
function plot_zalm2_fidelity()
    engine = HybridProjectionEngine(8)
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    states = zalm2.(μ, η', η'; engine=engine)
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    F = similar(states, Float64)
    Threads.@threads for I in CartesianIndices(states)
        F[I] = real(dot(ψ⁺', states[I], ψ⁺)) / tr(states[I])
    end
    F_ground = zalm.fidelity.(μ, η', 1, η')

    plot(μ, F_ground, label="Genqo v1 (ground truth)", xscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", legend=:topleft, color=[1 2 3 4])
    plot!(μ, F, label="Genqo v2", linestyle=:dash, color=[1 2 3 4])
end
@time plot_zalm2_fidelity()
