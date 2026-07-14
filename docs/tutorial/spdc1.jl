using Genqo
using Gabs

using Plots


# SPDC model

function spdc1(μ::Float64, ηᵗ::Float64; engine::HybridProjectionEngine)
    st = eprstate(QuadBlockBasis(4), asinh(√μ), 0.)
    apply!(st, modeswap(QuadBlockBasis(2)), [2,4])

    η = [ηᵗ,ηᵗ,ηᵗ,ηᵗ]
    Π = projector([:,:,:,:]) # TODO: is there a better way to do this?
    project(st, Π; engine, η=η)
end

## Probability of generation
function plot_spdc1_probability()
    engine = HybridProjectionEngine(4)
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    states = spdc1.(μ, η'; engine=engine)

    Pgen = tr.(states)
    Pgen_ground = ones(100)

    plot(μ, Pgen_ground, label="Genqo v1 (ground truth)", xscale=:log10, yscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Probability of Generation", legend=:bottomright, color=[1 2 3 4])
    plot!(μ, Pgen, label="Genqo v2", linestyle=:dash, color=[:blue :orange :green :red])
end
@time plot_spdc1_probability()

## Fidelity
function plot_spdc1_fidelity()
    engine = HybridProjectionEngine(4)
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    states = spdc1.(μ, η'; engine=engine)
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    F = @. η'^2 * real(dot([ψ⁺'], states, [ψ⁺])) / tr(states) # TODO: find out where this extra factor of η^2 comes from
    F_ground = spdc.fidelity.(μ, η', 1.)

    plot(μ, F_ground, label="Genqo v1 (ground truth)", xscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", legend=:topleft, color=[1 2 3 4])
    plot!(μ, F, label="Genqo v2", linestyle=:dash, color=[:blue :orange :green :red])
end
@time plot_spdc1_fidelity()
