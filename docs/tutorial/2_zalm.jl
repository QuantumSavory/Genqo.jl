using Genqo
using Gabs
using QuantumOpticsBase

using Plots
using ProgressBars


# ZALM model
function zalm2(μ::Float64, ηR::Float64, ηT::Float64)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, [2,4, 5,7], modeswap(QuadBlockBasis(4)))
    apply!(st, [3,5, 4,6], beamsplitter(QuadBlockBasis(4), 0.5))

    η = [ηR,ηR,ηT,ηT,ηT,ηT,ηR,ηR]
    Π = projector([:,:,1,0,0,1,:,:])
    project(st, Π; η=η)
end

## Probability of herald
function plot_zalm2_pherald()
    μ = logrange(1e-4, 10, 100)
    loss_dB = [0, 3, 6, 9]
    ηR = 0.01
    ηT = 10 .^ -(loss_dB ./ 10)

    states = zalm2.(μ, ηR, ηT')
    Pherald = tr.(states) * 4

    p = plot(
        xscale=:log10, yscale=:log10,
        xticks=10. .^ (-4:1),
        xlabel="Mean photon number per mode \$G-1\$", ylabel="Accepted herald probability \$P_H\$",
        title="ZALM: Probability of herald", legend=:bottomright, dpi=300
    )

    for (loss_dB_i, pHerald_i) in zip(loss_dB, eachcol(Pherald))
        plot!(p, μ, pHerald_i, label="ηT = $loss_dB_i dB")
    end
    p
end
@time plot_zalm2_pherald()

## Photon-photon fidelity
function plot_zalm2_photon_fidelity()
    μ = logrange(1e-4, 10, 100)
    loss_dB = [0, 3, 6, 9]
    ηR = 1. # we want the fidelity of the photonic state straight out of the source
    ηT = 10 .^ -(loss_dB ./ 10)
    ψ⁻ = (clicks([1,0,0,1]) - clicks([0,1,1,0])) / √2

    states = zalm2.(μ, ηR, ηT')
    Fp = fidelity.([ψ⁻], states) .|> real

    p = plot(
        xscale=:log10,
        xticks=10. .^ (-4:1),
        # ylim=[0,1],
        xlabel="Mean photon number per mode \$G-1\$", ylabel="Photon-photon fidelity \$F_p\$",
        title="ZALM: Photonic state fidelity", legend=:bottomleft, dpi=300
    )

    for (loss_dB_i, Fp_i) in zip(loss_dB, eachcol(Fp))
        plot!(p, μ, Fp_i, label="ηT = $loss_dB_i dB")
    end
    p
end
@time plot_zalm2_photon_fidelity()

## Spin-spin fidelity
function plot_zalm2_spin_fidelity()
    μ = logrange(1e-4, 10, 100)
    loss_dB = [0, 3, 6, 9]
    ηR = 0.01
    ηT = 10 .^ -(loss_dB ./ 10)

    states = zalm2.(μ, ηR, ηT')
    ρ = duankimble.(states, [[1,0,1,0]]) .* 4

    ψ⁻s = [0,1,-1,0] / √2
    fidelity_ψ⁻s(ρ::Operator) = dot(ψ⁻s', ρ.data, ψ⁻s) / tr(ρ) |> real
    Fs = fidelity_ψ⁻s.(ρ)

    p = plot(
        xscale=:log10,
        xticks=10. .^ (-4:1),
        ylim=[0,1],
        xlabel="Mean Photon number per mode \$G-1\$", ylabel="Spin-spin fidelity \$F_s\$",
        title="ZALM: Loaded state fidelity", legend=:topleft, dpi=300
    )
    for (loss_dB_i, Fs_i) in zip(loss_dB, eachcol(Fs))
        plot!(p, μ, Fs_i, label="ηT = $loss_dB_i dB")
    end
    p
end
@time plot_zalm2_spin_fidelity()

## Distillable entanglement rate
function plot_zalm2_distillable_entanglement_rate()
    μ = logrange(1e-4, 10, 100)
    ηR = 0.01
    ηT = 1.0:-0.1:0.6
    states = zalm2.(μ, ηR, ηT')

    ρABD = duankimble.(states, [[1,0,1,0]]) .* 4
    ρABE = emissiveload.(states) .* 4

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
        title="ZALM: Distillable entanglement rate", legend=:topleft, dpi=300
    )
    for i in eachindex(ηT)
        plot!(μ, RD[:,i], label="\\eta_T = $(ηT[i])", color=i)
    end
    for i in eachindex(ηT)
        plot!(μ, RE[:,i], label="", linestyle=:dash, color=i)
    end
    plot!([NaN], [NaN], label="Duan-Kimble loading", color=:black, linestyle=:solid)
    plot!([NaN], [NaN], label="Emissive loading", color=:black, linestyle=:dash)
    p
end
@time plot_zalm2_distillable_entanglement_rate()
