using Genqo
using Gabs

using Plots


# Alternative ZALM model

function alt_zalm(μ::Float64, ηᵗ::Float64, ηᵈ::Float64; proj::HybridProjector)
    basis4 = QuadBlockBasis(4)
    basis8 = QuadBlockBasis(8)
    st = eprstate(basis8, asinh(√μ), 0.)
    ms = modeswap(basis4)
    bs = beamsplitter(basis4, 0.5)

    apply!(st, ms, [2,4, 5,7])
    apply!(st, bs, [3,5, 4,6])

    η = [ηᵗ,ηᵗ,ηᵈ,ηᵈ,ηᵈ,ηᵈ,ηᵗ,ηᵗ]
    Π_even = (clicks([:,:,1,0,1,0,:,:]) + clicks([:,:,0,1,0,1,:,:])) / √2
    Π_odd = (clicks([:,:,1,0,0,1,:,:]) + clicks([:,:,0,1,1,0,:,:])) / √2
    (project(st, proj, Π_even; η=η), project(st, proj, Π_odd; η=η))
end

## Probability of generation
function plot_alt_zalm_probability()
    proj = HybridProjector(8)
    μ = logrange(1e-4, 10, 100)
    ηᵈ = 10 .^ -([0, 3, 6, 9]/10)
    states = alt_zalm.(μ, 1., ηᵈ'; proj=proj)

    Pgen = tr.(states)

    plot(μ, Pgen[:,1], label="\\eta_d = 0 dB", xlabel="Mean Photon Number Per Mode", ylabel="Probability of Success", legend=:bottomright, color=1)
    plot!(μ, Pgen[:,2], label="\\eta_d = 3 dB", color=2)
    plot!(μ, Pgen[:,3], label="\\eta_d = 6 dB", color=3)
    plot!(μ, Pgen[:,4], label="\\eta_d = 9 dB", color=4)
end
@time plot_alt_zalm_probability()

## Fidelity
function plot_alt_zalm_fidelity()
    proj = HybridProjector(8)
    μ = logrange(1e-4, 10, 100)
    ηᵈ = 10 .^ -([0, 3, 6, 9]/10)
    states = alt_zalm.(μ, 1., ηᵈ'; proj=proj)
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    F = similar(states, Float64)
    Threads.@threads for I in CartesianIndices(states)
        F[I] = real(dot(ψ⁺', states[I], ψ⁺)) / tr(states[I])
    end

    plot(μ, F[:,1], label="\\eta_d = 0 dB", ylim=[0,1], xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", legend=:bottomright, color=1)
    plot!(μ, F[:,2], label="\\eta_d = 3 dB", color=2)
    plot!(μ, F[:,3], label="\\eta_d = 6 dB", color=3)
    plot!(μ, F[:,4], label="\\eta_d = 9 dB", color=4)
end
@time plot_alt_zalm_fidelity()
