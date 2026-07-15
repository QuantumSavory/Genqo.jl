using Genqo
using Gabs

using Plots


# Alternative ZALM model

function alt_zalm(μ::Float64, ηᵗ::Float64, ηᵇ::Float64; engine::HybridProjectionEngine, parity::Symbol)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, modeswap(QuadBlockBasis(4)), [2,4, 5,7])

    η = [ηᵗ,ηᵗ,ηᵇ,ηᵇ,ηᵇ,ηᵇ,ηᵗ,ηᵗ]
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

# ZALM model for comparison

function zalm2(μ::Float64, ηᵗ::Float64, ηᵇ::Float64; engine::HybridProjectionEngine)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, modeswap(QuadBlockBasis(4)), [2,4, 5,7])
    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])

    η = [ηᵗ,ηᵗ,ηᵇ,ηᵇ,ηᵇ,ηᵇ,ηᵗ,ηᵗ]
    Π = projector([:,:,1,1,0,0,:,:])
    project(st, Π; engine, η=η)
end

## Probability of generation
function plot_alt_zalm_probability()
    engine = HybridProjectionEngine(8)
    μ = logrange(1e-4, 10, 100)
    ηᵇ = 10 .^ -([0, 3, 6, 9]/10)
    ηᵗ = 1.
    states_alt = alt_zalm.(μ, ηᵗ, ηᵇ'; engine=engine, parity=:even)
    states_orig = zalm2.(μ, ηᵗ, ηᵇ'; engine=engine)

    Pgen_alt = tr.(states_alt)
    Pgen_orig = tr.(states_orig)

    plot(μ, Pgen_alt[:,1], label="\\eta_b = 0 dB (alt)", xlabel="Mean Photon Number Per Mode", ylabel="Probability of Generation", title="Alternative ZALM vs. Original ZALM: Pgen", xscale=:log10, yscale=:log10, legend=:bottomright, color=1)
    plot!(μ, Pgen_alt[:,2], label="\\eta_b = 3 dB (alt)", color=2)
    plot!(μ, Pgen_alt[:,3], label="\\eta_b = 6 dB (alt)", color=3)
    plot!(μ, Pgen_alt[:,4], label="\\eta_b = 9 dB (alt)", color=4)
    plot!(μ, Pgen_orig[:,1], label="\\eta_b = 0 dB (orig)", linestyle=:dash, color=1)
    plot!(μ, Pgen_orig[:,2], label="\\eta_b = 3 dB (orig)", linestyle=:dash, color=2)
    plot!(μ, Pgen_orig[:,3], label="\\eta_b = 6 dB (orig)", linestyle=:dash, color=3)
    plot!(μ, Pgen_orig[:,4], label="\\eta_b = 9 dB (orig)", linestyle=:dash, color=4)
end
@time plot_alt_zalm_probability()

## Fidelity
function plot_alt_zalm_fidelity()
    engine = HybridProjectionEngine(8)
    μ = logrange(1e-4, 10, 100)
    ηᵇ = 10 .^ -([0, 3, 6, 9]/10)
    ηᵗ = 1.
    states_alt = alt_zalm.(μ, ηᵗ, ηᵇ'; engine=engine, parity=:odd)
    states_orig = zalm2.(μ, ηᵗ, ηᵇ'; engine=engine)

    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    F_alt = similar(states_alt, Float64)
    F_orig = similar(states_orig, Float64)
    Threads.@threads for I in CartesianIndices(states_alt)
        F_alt[I] = real(dot(ψ⁺', states_alt[I], ψ⁺)) / tr(states_alt[I])
        F_orig[I] = real(dot(ψ⁺', states_orig[I], ψ⁺)) / tr(states_orig[I])
    end

    plot(μ, F_alt[:,1], label="\\eta_b = 0 dB (alt)", xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", title="Alternative ZALM vs. Original ZALM: Fidelity", xscale=:log10, legend=:topleft, color=1)
    plot!(μ, F_alt[:,2], label="\\eta_b = 3 dB (alt)", color=2)
    plot!(μ, F_alt[:,3], label="\\eta_b = 6 dB (alt)", color=3)
    plot!(μ, F_alt[:,4], label="\\eta_b = 9 dB (alt)", color=4)
    plot!(μ, F_orig[:,1], label="\\eta_b = 0 dB (orig)", linestyle=:dash, color=1)
    plot!(μ, F_orig[:,2], label="\\eta_b = 3 dB (orig)", linestyle=:dash, color=2)
    plot!(μ, F_orig[:,3], label="\\eta_b = 6 dB (orig)", linestyle=:dash, color=3)
    plot!(μ, F_orig[:,4], label="\\eta_b = 9 dB (orig)", linestyle=:dash, color=4)
end
@time plot_alt_zalm_fidelity()
