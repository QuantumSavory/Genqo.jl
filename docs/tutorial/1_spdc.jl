using Genqo
using Gabs
using QuantumOpticsBase

using Plots
using ProgressBars


# SPDC model
function spdc1(μ::Float64, ηR::Float64)
    st = eprstate(QuadBlockBasis(4), asinh(√μ), 0.)
    apply!(st, [2,4], modeswap(QuadBlockBasis(2)))

    η = [ηR,ηR,ηR,ηR]
    Π = projector([:,:,:,:])
    project(st, Π; η=η)
end

# The unheralded source delivers a pair whenever one photon lands in each arm of the dual-rail state
coincidence_probability(st) = sum(ψ -> real(dot(ψ', st, ψ)), clicks.([[1,0,0,1], [0,1,1,0], [1,0,1,0], [0,1,0,1]]))

## Probability of generation
function plot_spdc1_pgen()
    μ = logrange(1e-4, 10, 100)
    ηR = 1. # we want the probability of generating a pair straight out of the source

    states = spdc1.(μ, ηR)
    Pgen = coincidence_probability.(states)

    p = plot(
        xscale=:log10, yscale=:log10,
        xticks=10. .^ (-4:1),
        xlabel="Mean photon number per mode \$G-1\$", ylabel="Pair generation probability \$P_{gen}\$",
        title="SPDC: Probability of generation", legend=:bottomright, dpi=300
    )
    plot!(p, μ, Pgen, label="")
    p
end
@time plot_spdc1_pgen()

## Photon-photon fidelity
function plot_spdc1_photon_fidelity()
    μ = logrange(1e-4, 10, 100)
    ηR = 1. # we want the fidelity of the photonic state straight out of the source
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2

    states = spdc1.(μ, ηR)
    photon_fidelity(st) = real(dot(ψ⁺', st, ψ⁺)) / coincidence_probability(st) # unheralded, so we condition on a coincidence instead of on a herald
    Fp = photon_fidelity.(states)

    p = plot(
        xscale=:log10,
        xticks=10. .^ (-4:1),
        xlabel="Mean photon number per mode \$G-1\$", ylabel="Photon-photon fidelity \$F_p\$",
        title="SPDC: Photonic state fidelity", legend=:bottomleft, dpi=300
    )

    plot!(p, μ, Fp, label="")
    p
end
@time plot_spdc1_photon_fidelity()

## Spin-spin fidelity
function plot_spdc1_spin_fidelity()
    μ = logrange(1e-4, 10, 100)
    ηR = 1. # once the total transmission is small the loaded state saturates, so keep the memory coupling ideal to expose the channel loss dependence

    states = spdc1.(μ, ηR)
    ρ = duankimble.(states, [[1,0,1,0]])

    ϕ⁻s = [1,0,0,-1] / √2 # Duan-Kimble loading maps the photonic ψ⁺ onto the spin-spin ϕ⁻
    fidelity_ϕ⁻s(ρ::Operator) = dot(ϕ⁻s', ρ.data, ϕ⁻s) / tr(ρ) |> real
    Fs = fidelity_ϕ⁻s.(ρ)

    p = plot(
        xscale=:log10,
        xticks=10. .^ (-4:1),
        ylim=[0,1],
        xlabel="Mean Photon number per mode \$G-1\$", ylabel="Spin-spin fidelity \$F_s\$",
        title="SPDC: Loaded state fidelity", legend=:topleft, dpi=300
    )
    plot!(p, μ, Fs, label="")
    p
end
@time plot_spdc1_spin_fidelity()

## Distillable entanglement rate
function plot_spdc1_distillable_entanglement_rate()
    μ = logrange(1e-4, 10, 100)
    ηR = 0.01
    states = spdc1.(μ, ηR)

    ρABD = duankimble.(states, [[1,0,1,0]])
    ρABE = emissiveload.(states)

    RD = similar(states, Float64)
    RE = similar(states, Float64)
    Threads.@threads for I in ProgressBar(eachindex(states))
        for (ρAB_unnorm,R) in zip((ρABD[I], ρABE[I]), (RD, RE))
            Pgen = tr(ρAB_unnorm) |> real
            ρAB = ρAB_unnorm / Pgen

            # Compute Hashing bound
            SρAB = entropy_vn(ρAB) |> real
            SρA = entropy_vn(ptrace(ρAB, 2)) |> real
            SρB = entropy_vn(ptrace(ρAB, 1)) |> real
            hashing = max(SρA - SρAB, SρB - SρAB)

            R[I] = max(hashing * Pgen, 1e-20) # send nonpositive numbers to 1e-20 for log scale
        end
    end

    p = plot(
        xscale=:log10, yscale=:log10,
        xticks=10. .^ (-4:1), yticks=10. .^ (-14:2:-4),
        xlim=[1e-4,1e1], ylim=[1e-14,1e-4],
        xlabel="Mean Photon number per mode \$G-1\$", ylabel="Distillable entanglement rate (ebit/pulse)",
        title="SPDC: Distillable entanglement rate", legend=:bottomright, dpi=300
    )
    plot!(μ, RD, label="Duan-Kimble loading")
    plot!(μ, RE, label="Emissive loading", linestyle=:dash)
    p
end
@time plot_spdc1_distillable_entanglement_rate()
