using Genqo
using Gabs

using Plots


# Alternative ZALM model

function alt_zalm(μ::Float64, ηᵗ::Float64, ηᵈ::Float64; engine::HybridProjectionEngine, parity::Symbol)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, modeswap(QuadBlockBasis(4)), [2,4, 5,7])

    η = [ηᵗ,ηᵗ,ηᵈ,ηᵈ,ηᵈ,ηᵈ,ηᵗ,ηᵗ]
    if parity == :even
        Π_even = projector([:,:,1,0,1,0,:,:]) + projector([:,:,0,1,0,1,:,:])
        project(st, Π_even; engine, η=η)
    elseif parity == :odd
        Π_odd = projector([:,:,1,0,0,1,:,:]) + projector([:,:,0,1,1,0,:,:])
        project(st, Π_odd; engine, η=η)
    else
        throw(ArgumentError("parity must be :even or :odd"))
    end
end

## Probability of generation
function plot_alt_zalm_probability()
    engine = HybridProjectionEngine(8)
    μ = logrange(1e-4, 10, 100)
    ηᵈ = 10 .^ -([0, 3, 6, 9]/10)
    states = alt_zalm.(μ, 1., ηᵈ'; engine=engine, parity=:odd)

    Pgen = tr.(states)

    plot(μ, Pgen[:,1], label="\\eta_d = 0 dB", xlabel="Mean Photon Number Per Mode", ylabel="Probability of Success", legend=:bottomright, color=1)
    plot!(μ, Pgen[:,2], label="\\eta_d = 3 dB", color=2)
    plot!(μ, Pgen[:,3], label="\\eta_d = 6 dB", color=3)
    plot!(μ, Pgen[:,4], label="\\eta_d = 9 dB", color=4)
end
@time plot_alt_zalm_probability()

## Fidelity
function plot_alt_zalm_fidelity()
    engine = HybridProjectionEngine(8)
    μ = logrange(1e-4, 10, 100)
    ηᵈ = 10 .^ -([0, 3, 6, 9]/10)
    states = alt_zalm.(μ, 1., ηᵈ'; engine=engine, parity=:even)
    ϕ⁺ = (clicks([1,0,1,0]) + clicks([0,1,0,1])) / √2
    F = similar(states, Float64)
    Threads.@threads for I in CartesianIndices(states)
        F[I] = real(dot(ϕ⁺', states[I], ϕ⁺)) / tr(states[I])
    end

    plot(μ, F[:,1], label="\\eta_d = 0 dB", ylim=[0,1], xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", legend=:bottomright, color=1)
    plot!(μ, F[:,2], label="\\eta_d = 3 dB", color=2)
    plot!(μ, F[:,3], label="\\eta_d = 6 dB", color=3)
    plot!(μ, F[:,4], label="\\eta_d = 9 dB", color=4)
end
@time plot_alt_zalm_fidelity()
