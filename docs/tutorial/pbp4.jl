using Genqo
using Gabs

using Plots


# 4-source PBP model

function pbp4(μ::Float64, ηᵗ::Float64, ηᵇ::Float64, ηᵍ::Float64)
    sagnac = eprstate(QuadBlockBasis(4), asinh(√μ), 0.)
    apply!(sagnac, modeswap(QuadBlockBasis(2)), [1,3])
    st = sagnac ⊗ sagnac ⊗ sagnac ⊗ sagnac

    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])
    apply!(st, greenmachine(QuadBlockBasis(4), 4), [7,8, 9,10])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [11,13, 12,14])

    η = [ηᵗ,ηᵗ,ηᵇ,ηᵇ,ηᵇ,ηᵇ,ηᵍ,ηᵍ,ηᵍ,ηᵍ,ηᵇ,ηᵇ,ηᵇ,ηᵇ,ηᵗ,ηᵗ]
    Π = projector([:,:,1,0,0,1,1,0,0,1,1,0,0,1,:,:])
    project(st, Π; η=η)
end

## Probability of generation
function plot_pbp4_probability()
    μ = range(1e-3, 1, 100)
    η = [1., 0.8, 0.5, 0.25]
    states = pbp4.(μ, 1., η', η')
    Pgen = tr.(states)

    plot(μ, Pgen[:,1], label="\\eta = 1.0", xlabel="Mean Photon Number Per Mode", ylabel="Probability of Generation", legend=:bottomright, color=1)
    plot!(μ, Pgen[:,2], label="\\eta = 0.8", color=2)
    plot!(μ, Pgen[:,3], label="\\eta = 0.5", color=3)
    plot!(μ, Pgen[:,4], label="\\eta = 0.25", color=4)
end
@time plot_pbp4_probability()

## Fidelity
function plot_pbp4_fidelity()
    μ = range(1e-4, 1.0, 100)
    η = [1., 0.8, 0.5, 0.25]
    states = pbp4.(μ, 1., η', η')
    ψ⁻ = (clicks([1,0,0,1]) - clicks([0,1,1,0])) / √2
    F = @. dot([ψ⁻'], states, [ψ⁻]) / tr(states) |> real

    plot(μ, F[:,1], label="\\eta = 1.0", ylim=[0,1], xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", legend=:bottomright, color=1)
    plot!(μ, F[:,2], label="\\eta = 0.8")
    plot!(μ, F[:,3], label="\\eta = 0.5")
    plot!(μ, F[:,4], label="\\eta = 0.25")
end
@time plot_pbp4_fidelity()
