using Genqo
using Gabs
using QuantumOpticsBase

using ProgressBars
using Plots


const engine = HybridProjectionEngine(12)

# 3-source PBP model

function pbp3(μ::Float64, ηᵗ::Float64, ηᵇ::Float64, ηᵍ::Float64)
    sagnac = eprstate(QuadBlockBasis(4), asinh(√μ), 0.)
    apply!(sagnac, modeswap(QuadBlockBasis(2)), [1,3])
    st = sagnac ⊗ sagnac ⊗ sagnac

    apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])
    apply!(st, greenmachine(QuadBlockBasis(4), 4), [7,8, 9,10])

    η = [ηᵗ,ηᵗ,ηᵇ,ηᵇ,ηᵇ,ηᵇ,ηᵍ,ηᵍ,ηᵍ,ηᵍ,ηᵗ,ηᵗ]
    Π = projector([:,:,1,0,0,1,1,0,0,1,:,:])
    project(st, Π; engine, η=η)
end

## Probability of generation
function plot_pbp3_probability()
    μ = range(1e-3, 1, 100)
    η = [1., 0.8, 0.5, 0.25]
    states = pbp3.(μ, 1., η', η')
    Pgen = tr.(states)

    plot(μ, Pgen[:,1], label="\\eta = 1.0", xlabel="Mean Photon Number Per Mode", ylabel="Probability of Generation", legend=:bottomright, color=1)
    plot!(μ, Pgen[:,2], label="\\eta = 0.8", color=2)
    plot!(μ, Pgen[:,3], label="\\eta = 0.5", color=3)
    plot!(μ, Pgen[:,4], label="\\eta = 0.25", color=4)
    # True 3-pair heralds + 4-pair false heralds (both photons of an outer-source double pair detected in the same BSM)
    plot!(μ, μ.^3 ./ (μ.+1).^9 ./ 8 .+ μ.^4 ./ (μ.+1).^10 ./ 16, label="\\eta = 1.0 analytical", linestyle=:dash, color=:blue)
end
@time plot_pbp3_probability()

## Fidelity
function plot_pbp3_fidelity()
    μ = range(1e-4, 1.0, 100)
    η = [1., 0.8, 0.5, 0.25]
    states = pbp3.(μ, 1., η', η')
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    F = similar(states, Float64)
    Threads.@threads for I in ProgressBar(CartesianIndices(states))
        F[I] = real(dot(ψ⁺', states[I], ψ⁺)) / tr(states[I])
    end

    plot(μ, F[:,1], label="\\eta = 1.0", ylim=[0,1], xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", legend=:bottomright, color=1)
    plot!(μ, F[:,2], label="\\eta = 0.8")
    plot!(μ, F[:,3], label="\\eta = 0.5")
    plot!(μ, F[:,4], label="\\eta = 0.25")
end
@time plot_pbp3_fidelity()

## Distillable entanglement rate
function plot_pbp3_distillable_entanglement_rate()
    μ = logrange(1e-4, 10, 100)
    ηR = 0.01
    ηT = 1.0:-0.1:0.6
    states = pbp3.(μ, ηR, ηT', ηT')
    ρAB = similar(states, Operator)
    Pgen = similar(states, Float64)
    Threads.@threads for I in ProgressBar(CartesianIndices(states))
        ρAB[I] = duankimble(states[I], [1,0,1,0]) * 4
        Pgen[I] = tr(ρAB[I])
        ρAB[I] /= Pgen[I]
    end

    # Compute Hashing bound
    SρAB = @. entropy_vn(ρAB) |> real
    SρA = @. entropy_vn(ptrace(ρAB, 2)) |> real
    SρB = @. entropy_vn(ptrace(ρAB, 1)) |> real
    I = max.(SρA - SρAB, SρB - SρAB)

    # Compute distillable entanglement rate
    R = max.(I .* Pgen, 1e-20) # send nonpositive numbers to 1e-20 for log scale

    plot(μ, R[:,1], label="\\eta_T = $(ηT[1])", xlabel="Mean Photon Number Per Mode", ylabel="Distillable Entanglement Rate", xscale=:log10, yscale=:log10, legend=:topleft)
    plot!(μ, R[:,2], label="\\eta_T = $(ηT[2])")
    plot!(μ, R[:,3], label="\\eta_T = $(ηT[3])")
    plot!(μ, R[:,4], label="\\eta_T = $(ηT[4])")
    plot!(μ, R[:,5], label="\\eta_T = $(ηT[5])")
end
@time plot_pbp3_distillable_entanglement_rate()
