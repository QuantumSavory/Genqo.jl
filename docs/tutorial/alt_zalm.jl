using Genqo
using Gabs
using QuantumOpticsBase

using Plots


# Alternative ZALM model

function alt_zalm(μ::Float64, ηᵗ::Float64, ηᵇ::Float64; parity::Symbol)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, [2,4, 5,7], modeswap(QuadBlockBasis(4)))

    η = [ηᵗ,ηᵗ,ηᵇ,ηᵇ,ηᵇ,ηᵇ,ηᵗ,ηᵗ]
    if parity == :even
        Π_even = projector([:,:,1,0,1,0,:,:]) + projector([:,:,0,1,0,1,:,:])
        project(st, Π_even; η=η)
    elseif parity == :odd
        Π_odd = projector([:,:,1,0,0,1,:,:]) + projector([:,:,0,1,1,0,:,:])
        project(st, Π_odd; η=η)
    else
        throw(ArgumentError("parity must be :even or :odd"))
    end
end

# ZALM model for comparison

function zalm2(μ::Float64, ηᵗ::Float64, ηᵇ::Float64)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, [2,4, 5,7], modeswap(QuadBlockBasis(4)))
    apply!(st, [3,5, 4,6], beamsplitter(QuadBlockBasis(4), 0.5))

    η = [ηᵗ,ηᵗ,ηᵇ,ηᵇ,ηᵇ,ηᵇ,ηᵗ,ηᵗ]
    Π = projector([:,:,1,1,0,0,:,:])
    project(st, Π; η=η)
end

## Probability of generation
function plot_alt_zalm_probability()
    μ = logrange(1e-4, 10, 100)
    loss_dB = [0, 3, 6, 9]
    ηT = 10 .^ -(loss_dB/10)
    ηR = 1.
    states_alt = alt_zalm.(μ, ηR, ηT'; parity=:even)
    states_orig = zalm2.(μ, ηR, ηT')

    Pgen_alt = tr.(states_alt)
    Pgen_orig = tr.(states_orig)

    local p
    for (i,l) in enumerate(loss_dB)
        if i == 1
            p = plot(μ, Pgen_alt[:,i], label="\\eta_T = 0 dB", xlabel="Mean Photon Number Per Mode", ylabel="Probability of Generation", title="Alt ZALM vs ZALM: Probability of generation", xscale=:log10, yscale=:log10, legend=:bottomright, color=1, dpi=300)
        else
            plot!(p, μ, Pgen_alt[:,i], label="\\eta_T = $(l) dB", color=i)
        end
    end
    for i in 1:length(loss_dB)
        plot!(p, μ, Pgen_orig[:,i], label="", linestyle=:dash, color=i)
    end
    plot!(p, [NaN], [NaN], label="ZALM", color=:black, linestyle=:dash)
    plot!(p, [NaN], [NaN], label="Alt ZALM", color=:black, linestyle=:solid)
    p
end
@time plot_alt_zalm_probability()

## Fidelity
function plot_alt_zalm_fidelity()
    μ = logrange(1e-4, 10, 100)
    loss_dB = [0, 3, 6, 9]
    ηT = 10 .^ -(loss_dB/10)
    ηR = 1.
    states_alt = alt_zalm.(μ, ηR, ηT'; parity=:odd)
    states_orig = zalm2.(μ, ηR, ηT')

    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    F_alt = @. dot([ψ⁺'], states_alt, [ψ⁺]) / tr(states_alt) |> real
    F_orig = @. dot([ψ⁺'], states_orig, [ψ⁺]) / tr(states_orig) |> real

    local p
    for (i,l) in enumerate(loss_dB)
        if i == 1
            p = plot(μ, F_alt[:,i], label="\\eta_T = 0 dB", xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", title="Alt ZALM vs ZALM: Fidelity", xscale=:log10, legend=:topleft, color=1, dpi=300)
        else
            plot!(p, μ, F_alt[:,i], label="\\eta_T = $(l) dB", color=i)
        end
    end
    for i in 1:length(loss_dB)
        plot!(p, μ, F_orig[:,i], label="", linestyle=:dash, color=i)
    end
    plot!(p, [NaN], [NaN], label="ZALM", color=:black, linestyle=:dash)
    plot!(p, [NaN], [NaN], label="Alt ZALM", color=:black, linestyle=:solid)
    annotate!(p, 7e-4, 0.05, Plots.text("(lines are indistinguishable)", 10, :left, :bottom))
    p
end
@time plot_alt_zalm_fidelity()

## Distillable entanglement rate
function plot_alt_zalm_distillable_entanglement_rate()
    μ = logrange(1e-4, 10, 100)
    ηR = 0.1
    ηT = 1.0:-0.1:0.6
    states = alt_zalm.(μ, ηR, ηT'; parity=:odd)
    ρAB = duankimble.(states, [[1,0,1,0]]) .* 4
    Pgen = tr.(ρAB) .|> real
    ρAB ./= Pgen

    # Compute Hashing bound
    SρAB = @. entropy_vn(ρAB) |> real
    SρA = @. entropy_vn(ptrace(ρAB, 2)) |> real
    SρB = @. entropy_vn(ptrace(ρAB, 1)) |> real
    I = max.(SρA - SρAB, SρB - SρAB)

    # Compute distillable entanglement rate
    R = I .* Pgen
    # R = max.(I .* Pgen, 1e-20) # send nonpositive numbers to 1e-20 for log scale

    local p
    for (i,ηTi) in enumerate(ηT)
        if i == 1
            p = plot(μ, R[:,i], label="\\eta_T = $ηTi", xlabel="Mean Photon Number Per Mode", ylabel="Distillable Entanglement Rate", title="Alt ZALM: Distillable entanglement rate", xscale=:log10, legend=:topleft, dpi=300, color=i)
        else
            plot!(p, μ, R[:,i], label="\\eta_T = $ηTi", color=i)
        end
    end
    p
end
@time plot_alt_zalm_distillable_entanglement_rate()
