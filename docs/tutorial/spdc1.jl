using Genqo
using Gabs

using Plots


# SPDC model

function spdc1(μ::Float64, ηᵗ::Float64)
    st = eprstate(QuadBlockBasis(4), asinh(√μ), 0.)
    apply!(st, modeswap(QuadBlockBasis(2)), [2,4])

    η = [ηᵗ,ηᵗ,ηᵗ,ηᵗ]
    Π = projector([:,:,:,:]) # TODO: is there a better way to do this?
    project(st, Π; η=η)
end

## Probability of generation
function plot_spdc1_probability()
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    states = spdc1.(μ, η')

    Pgen = tr.(states)
    Pgen_ground = ones(100)

    plot(μ, Pgen_ground, label="Genqo v1 (ground truth)", xscale=:log10, yscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Probability of Generation", legend=:bottomright, color=[1 2 3 4])
    plot!(μ, Pgen, label="Genqo v2", linestyle=:dash, color=[:blue :orange :green :red])
end
@time plot_spdc1_probability()

## Fidelity
function plot_spdc1_fidelity()
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    states = spdc1.(μ, η')
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    F = @. real(dot([ψ⁺'], states, [ψ⁺])) / tr(states)
    F_ground = spdc.fidelity.(μ, η', 1.)

    plot(μ, F_ground, label="Genqo v1 (ground truth)", xscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", legend=:topleft, color=[1 2 3 4])
    plot!(μ, F, label="Genqo v2", linestyle=:dash, color=[:blue :orange :green :red])
end
@time plot_spdc1_fidelity()

## Spin-spin density matrix after Duan-Kimble loading
function check_spdc1_spin_density_matrix()
    μ = logrange(1e-4, 10, 4)
    η = 10 .^ -([0, 3, 6, 9]/10)
    states = spdc1.(μ, η')
    ρ = duankimble.(states, [[1,0,1,0]])
    ρ_ground = spdc.spin_density_matrix.(μ, η', 1., [[1,0,1,0]])

    println("Spin-spin density matrix after Duan-Kimble loading:")
    println("Genqo v2:")
    display(ρ[3,2])
    println("Genqo v1 (ground truth):")
    display(ρ_ground[3,2])
    println("Equal? ", all(ρi.data ≈ ρi_ground for (ρi, ρi_ground) in zip(ρ, ρ_ground)) ? "✓" : "✗")
end
@time check_spdc1_spin_density_matrix()
