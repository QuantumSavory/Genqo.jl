using Genqo
using Gabs
using QuantumOpticsBase

using Plots


# 3-source PBP model

function pbp3(μ::Float64, ηᵗ::Float64, ηᵇ::Float64, ηᵍ::Float64)
    sagnac = eprstate(QuadBlockBasis(4), asinh(√μ), 0.)
    apply!(sagnac, [1,3], modeswap(QuadBlockBasis(2)))
    st = sagnac ⊗ sagnac ⊗ sagnac

    apply!(st, [3,5, 4,6], beamsplitter(QuadBlockBasis(4), 0.5))
    apply!(st, [7,8, 9,10], greenmachine(QuadBlockBasis(4), 4))

    η = [ηᵗ,ηᵗ,ηᵇ,ηᵇ,ηᵇ,ηᵇ,ηᵍ,ηᵍ,ηᵍ,ηᵍ,ηᵗ,ηᵗ]
    Π = projector([:,:,1,0,0,1,1,0,0,1,:,:])
    project(st, Π; η=η)
end

## Probability of generation
function plot_pbp3_probability()
    μ = range(1e-3, 1, 100)
    ηT = [1., 0.8, 0.5, 0.25]
    states = pbp3.(μ, 1., ηT', ηT')
    Pgen = tr.(states)

    local p
    for (i,ηTi) in enumerate(ηT)
        if i == 1
            p = plot(μ, Pgen[:,i], label="\\eta = $ηTi", xlabel="Mean Photon Number Per Mode", ylabel="Probability of Generation", title="PBP3: Probability of generation", legend=:bottomright, dpi=300, color=i)
        else
            plot!(p, μ, Pgen[:,i], label="\\eta = $ηTi", color=i)
        end
    end
    # True 3-pair heralds + 4-pair false heralds (both photons of an outer-source double pair detected in the same BSM)
    plot!(p, μ, μ.^3 ./ (μ.+1).^9 ./ 8 .+ μ.^4 ./ (μ.+1).^10 ./ 16, label="\\eta = 1.0 analytical", linestyle=:dash, color=:blue)
    p
end
@time plot_pbp3_probability()

## Fidelity
function plot_pbp3_fidelity()
    μ = range(1e-4, 1.0, 100)
    η = [1., 0.8, 0.5, 0.25]
    states = pbp3.(μ, 1., η', η')
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    F = @. real(dot([ψ⁺'], states, [ψ⁺])) / tr(states)

    local p
    for (i,ηi) in enumerate(η)
        if i == 1
            p = plot(μ, F[:,i], label="\\eta = $ηi", ylim=[0,1], xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", title="PBP3: Fidelity", legend=:bottomright, dpi=300, color=i)
        else
            plot!(p, μ, F[:,i], label="\\eta = $ηi", color=i)
        end
    end
    p
end
@time plot_pbp3_fidelity()

## Distillable entanglement rate
function plot_pbp3_distillable_entanglement_rate()
    μ = logrange(1e-4, 10, 100)
    ηR = 0.01
    ηT = 1.0:-0.1:0.6
    states = pbp3.(μ, ηR, ηT', ηT')
    ρAB = duankimble.(states, [[1,0,1,0]]) .* 4^2
    Pgen = tr.(ρAB) .|> real
    ρAB ./= Pgen

    # Compute Hashing bound
    SρAB = @. entropy_vn(ρAB) |> real
    SρA = @. entropy_vn(ptrace(ρAB, 2)) |> real
    SρB = @. entropy_vn(ptrace(ρAB, 1)) |> real
    I = max.(SρA - SρAB, SρB - SρAB)

    # Compute distillable entanglement rate
    R = max.(I .* Pgen, 1e-20) # send nonpositive numbers to 1e-20 for log scale

    local p
    for (i,ηTi) in enumerate(ηT)
        if i == 1
            p = plot(μ, R[:,i], label="\\eta_T = $ηTi", xlabel="Mean Photon Number Per Mode", ylabel="Distillable Entanglement Rate", title="PBP3: Distillable entanglement rate", xscale=:log10, yscale=:log10, xticks=10. .^ (-4:1), yticks=10. .^ (-14:2:-4), xlim=[1e-4,1e1], ylim=[1e-14,1e-4], legend=:topleft, dpi=300, color=i)
        else
            plot!(p, μ, R[:,i], label="\\eta_T = $ηTi", color=i)
        end
    end
    p
end
@time plot_pbp3_distillable_entanglement_rate()
