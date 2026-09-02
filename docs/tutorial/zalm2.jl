using Genqo
using Gabs
using QuantumOpticsBase

using Plots


# ZALM model

function zalm2(μ::Float64, ηR::Float64, ηT::Float64)
    st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
    apply!(st, [2,4, 5,7], modeswap(QuadBlockBasis(4)))
    apply!(st, [3,5, 4,6], beamsplitter(QuadBlockBasis(4), 0.5))

    η = [ηR,ηR,ηT,ηT,ηT,ηT,ηR,ηR]
    Π = projector([:,:,1,1,0,0,:,:])
    project(st, Π; η=η)
end

## Probability of generation
function plot_zalm2_probability()
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    states = zalm2.(μ, 1., η')

    Pgen = tr.(states)
    Pgen_ground = zalm.probability_success.(μ, 1., 1., η', 0)

    plot(μ, Pgen_ground, label="Genqo v1 (ground truth)", xscale=:log10, yscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Probability of Generation", title="ZALM: Probability of generation", legend=:bottomright, dpi=300, color=[1 2 3 4])
    plot!(μ, Pgen, label="Genqo v2", linestyle=:dash, color=[:blue :orange :green :red])
    plot!(μ, μ.^2 ./ (μ.+1).^6, label="Analytical", linestyle=:dot, color=:black)
end
@time plot_zalm2_probability()

## Bell-state fraction

## Fidelity
function plot_zalm2_fidelity()
    engine = HybridProjectionEngine(8)
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    states = zalm2.(μ, 1., η'; engine=engine)
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    F = similar(states, Float64)
    Threads.@threads for I in CartesianIndices(states)
        F[I] = real(dot(ψ⁺', states[I], ψ⁺)) / tr(states[I])
    end
    F_ground = zalm.fidelity.(μ, 1., 1., η')

    plot(μ, F_ground, label="Genqo v1 (ground truth)", ylim=[0,1], xscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", legend=:topleft, color=[1 2 3 4])
    plot!(μ, F, label="Genqo v2", linestyle=:dash, color=[1 2 3 4])
end
@time plot_zalm2_fidelity()
function plot_zalm2_bell_fidelity()
    engine = HybridProjectionEngine(8)
    μ = logrange(1e-4, 1.5, 100)
    
    function zalm2_bell_fidelity(μ::Float64,ηT::Float64, ηR::Float64; engine::HybridProjectionEngine)
        st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
        apply!(st, modeswap(QuadBlockBasis(4)), [2,4, 5,7])
        apply!(st, beamsplitter(QuadBlockBasis(4), 0.5), [3,5, 4,6])

        η = [ηR,ηR,ηT,ηT,ηT,ηT,ηR,ηR]
        Π = projector([:,:,1,1,0,0,:,:])
        st = project(st, Π; engine, η=η)

        ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
        ψ⁻ = (clicks([1,0,0,1]) - clicks([0,1,1,0])) / √2
        ϕ⁺ = (clicks([1,0,1,0]) + clicks([0,1,0,1])) / √2
        ϕ⁻ = (clicks([1,0,1,0]) - clicks([0,1,0,1])) / √2

       real(dot(ψ⁺', st, ψ⁺)) / real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻))
    end

    function zalm_bell_fidelity(μ::Real, ηT::Real, ηR::Real)::Real

        Ns  = (ηT*μ + 1) / (μ + 1)
        
        N′s = (ηR/Ns + (1 - ηR))^(-1)
        
        Pr_psi_minus = (N′s^4 / 2) * (
            2*(1 - N′s)^2
            - (2*ηR*(3*N′s^3 - 5*N′s^2 + 2*N′s)) / Ns
            + (ηR^2*(5*N′s^4 - 6*N′s^3 + 2*N′s^2)) / Ns^2  
        )
        Pr_Bell = 2*N′s^4 * (
            2*(1 - N′s)^2
            - (2*ηR*(3*N′s^3 - 5*N′s^2 + 2*N′s)) / Ns
            + (ηR^2*(4*N′s^4 - 6*N′s^3 + 2*N′s^2)) / Ns^2
        ) + (ηR^2 * N′s^8) / (2*Ns^2)

        return Pr_psi_minus / Pr_Bell
    end

    F = zalm2_bell_fidelity.(μ, 0.9, 0.01; engine=engine)
    F_numerical = zalm_bell_fidelity.(μ, 0.9, 0.01)
    plot(μ, F_numerical, label="numerical formula", xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", legend=:bottomright,color=1)
    plot!(μ, F, label="Genqo v2", color=2, linestyle=:dash)
end
@time plot_zalm2_bell_fidelity()

function plot_zalm2_Bell_state_fraction()
    μ = logrange(1e-4, 1.5, 100)

    function zalm2_Pload(μ::Float64, ηR::Float64, ηT::Float64)
        st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
        apply!(st, [2,4, 5,7], modeswap(QuadBlockBasis(4)))
        apply!(st, [3,5, 4,6], beamsplitter(QuadBlockBasis(4), 0.5))

        η = [ηR,ηR,ηT,ηT,ηT,ηT,ηR,ηR]
        Π_S = projector([:,:,1,1,0,0,:,:])
        Π_A0 = projector([0,0,1,1,0,0,:,:])
        Π_B0 = projector([:,:,1,1,0,0,0,0])
        Π_S0 = projector([0,0,1,1,0,0,0,0])
        st_S = project(st, Π_S; η=η)
        st_A0 = project(st, Π_A0; η=η)
        st_B0 = project(st, Π_B0; η=η)
        st_S0 = project(st, Π_S0; η=η)

        tr(st_S) - tr(st_A0) - tr(st_B0) + tr(st_S0)
    end

    function zalm2_PBell(μ::Float64, ηR::Float64, ηT::Float64)
        st = eprstate(QuadBlockBasis(8), asinh(√μ), 0.)
        apply!(st, [2,4, 5,7], modeswap(QuadBlockBasis(4)))
        apply!(st, [3,5, 4,6], beamsplitter(QuadBlockBasis(4), 0.5))

        η = [ηR,ηR,ηT,ηT,ηT,ηT,ηR,ηR]
        Π = projector([:,:,1,1,0,0,:,:])
        st = project(st, Π; η=η)

        ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
        ψ⁻ = (clicks([1,0,0,1]) - clicks([0,1,1,0])) / √2
        ϕ⁺ = (clicks([1,0,1,0]) + clicks([0,1,0,1])) / √2
        ϕ⁻ = (clicks([1,0,1,0]) - clicks([0,1,0,1])) / √2

        real(dot(ψ⁺', st, ψ⁺) + dot(ψ⁻', st, ψ⁻) + dot(ϕ⁺', st, ϕ⁺) + dot(ϕ⁻', st, ϕ⁻))
    end

    Pload = zalm2_Pload.(μ, 1., 0.9)
    PBell = zalm2_PBell.(μ, 1., 0.9)

    B = PBell ./ Pload
    plot(μ, B, label="ZALM", xlabel="Mean Photon Number Per Mode", ylabel="Bell-state Fraction", title="ZALM: Bell-state fraction", legend=:bottomright, dpi=300, color=1)
end
@time plot_zalm2_Bell_state_fraction()

## Fidelity
function plot_zalm2_fidelity()
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    states = zalm2.(μ, 1., η')
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    F = similar(states, Float64)
    Threads.@threads for I in CartesianIndices(states)
        F[I] = real(dot(ψ⁺', states[I], ψ⁺)) / tr(states[I])
    end
    F_ground = zalm.fidelity.(μ, 1., 1., η')

    plot(μ, F_ground, label="Genqo v1 (ground truth)", ylim=[0,1], xscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", title="ZALM: Fidelity", legend=:topleft, dpi=300, color=[1 2 3 4])
    plot!(μ, F, label="Genqo v2", linestyle=:dash, color=[1 2 3 4])
end
@time plot_zalm2_fidelity()

## Spin-spin density matrix after Duan-Kimble loading
function check_zalm2_spin_density_matrix()
    μ = logrange(1e-4, 10, 4)
    ηR = 10 .^ -([0, 5, 10, 15]/10)
    ηT = 10 .^ -([0, 3, 6, 9]/10)
    states = zalm2.(μ, ηR', reshape(ηT, (1,1,:)))
    ρ = similar(states, Operator)
    Threads.@threads for I in CartesianIndices(states)
        ρ[I] = duankimble(states[I], [1,0,1,0]) * 4
    end
    ρ_ground = zalm.spin_density_matrix.(μ, ηR', 1., reshape(ηT, (1,1,:)), [[1,0,1,1,0,0,1,0]])

    println("Spin-spin density matrix after Duan-Kimble loading:")
    println("Genqo v2:")
    display(ρ[3,2,2])
    println("Genqo v1 (ground truth):")
    display(ρ_ground[3,2,2])
    println("Equal? ", all(ρi.data ≈ ρi_ground for (ρi, ρi_ground) in zip(ρ, ρ_ground)) ? "✓" : "✗")
end
@time check_zalm2_spin_density_matrix()

## Distillable entanglement rate
function plot_zalm2_distillable_entanglement_rate()
    μ = logrange(1e-4, 10, 100)
    ηR = 0.01
    ηT = 1.0:-0.1:0.6
    states = zalm2.(μ, ηR, ηT')
    ρABD = similar(states, Operator)
    PgenD = similar(states, Float64)
    ρABE = similar(states, Operator)
    PgenE = similar(states, Float64)
    Threads.@threads for I in CartesianIndices(states)
        ρABD[I] = duankimble(states[I], [1,0,1,0]) * 4
        PgenD[I] = tr(ρABD[I])
        ρABD[I] /= PgenD[I]
        ρABE[I] = emissiveload(states[I]) * 4
        PgenE[I] = tr(ρABE[I])
        ρABE[I] /= PgenE[I]
    end

    # Compute Hashing bound
    SρABD = @. entropy_vn(ρABD) |> real
    SρA = @. entropy_vn(ptrace(ρABD, 2)) |> real
    SρB = @. entropy_vn(ptrace(ρABD, 1)) |> real
    I = max.(SρA - SρABD, SρB - SρABD)

    SρABE = @. entropy_vn(ρABE) |> real
    SρA = @. entropy_vn(ptrace(ρABE, 2)) |> real
    SρB = @. entropy_vn(ptrace(ρABE, 1)) |> real
    I = max.(SρA - SρABE, SρB - SρABE)

    # Compute distillable entanglement rate
    RD = max.(I .* PgenD, 1e-20) # send nonpositive numbers to 1e-20 for log scale
    RE = max.(I .* PgenE, 1e-20) # send nonpositive numbers to 1e-20 for log scale

    local p
    for i in eachindex(ηT)
        if i == 1
            p = plot(μ, RD[:,i], label="\\eta_T = $(ηT[i])", xlabel="Mean Photon Number Per Mode", ylabel="Distillable Entanglement Rate", title="ZALM: Distillable entanglement rate", xscale=:log10, yscale=:log10, xticks=10. .^ (-4:1), yticks=10. .^ (-14:2:-4), xlim=[1e-4,1e1], ylim=[1e-14,1e-4], legend=:topleft, dpi=300, color=i)
        else
            plot!(μ, RD[:,i], label="\\eta_T = $(ηT[i])", color=i)
        end
    end
    for i in eachindex(ηT)
        plot!(μ, RE[:,i], label="", linestyle=:dash, color=i)
    end
    plot!([NaN], [NaN], label="Duan-Kimble loading", color=:black, linestyle=:solid)
    plot!([NaN], [NaN], label="Emissive loading", color=:black, linestyle=:dash)
    p
end
@time plot_zalm2_distillable_entanglement_rate()

## Spin-spin fidelity
function plot_zalm2_spin_fidelity()
    μ = logrange(1e-4, 10, 100)
    ηR = 0.01
    ηT = 1.0:-0.1:0.6
    states = zalm2.(μ, ηR, ηT')
    ρD = similar(states, Operator)
    PgenD = similar(states, Float64)
    ρE = similar(states, Operator)
    PgenE = similar(states, Float64)
    Threads.@threads for I in CartesianIndices(states)
        ρD[I] = duankimble(states[I], [1,0,1,0]) * 4
        PgenD[I] = tr(ρD[I])
        ρD[I] /= PgenD[I]
        ρE[I] = emissiveload(states[I]) * 4
        PgenE[I] = tr(ρE[I])
        ρE[I] /= PgenE[I]
    end

    fidelityD(ρi::Operator) = dot([1,0,0,-1]'/√2, ρi.data, [1,0,0,-1]/√2) / tr(ρi) |> real
    fidelityE(ρi::Operator) = dot([0,1,1,0]'/√2, ρi.data, [0,1,1,0]/√2) / tr(ρi) |> real
    FsD = fidelityD.(ρD)
    FsE = fidelityE.(ρE)
    local p
    for i in eachindex(ηT)
        if i == 1
            p = plot(μ, FsD[:,i], label="\\eta_T = $(ηT[i])", xlabel="Mean Photon Number Per Mode", ylabel="Spin-spin Fidelity", title="ZALM: Spin-spin fidelity", xscale=:log10, legend=:bottomleft, dpi=300, color=i)
        else
            plot!(p, μ, FsD[:,i], label="\\eta_T = $(ηT[i])", color=i)
        end
    end
    for i in eachindex(ηT)
        plot!(p, μ, FsE[:,i], label="", linestyle=:dash, color=i)
    end
    plot!(p, [NaN], [NaN], label="Duan-Kimble loading", color=:black, linestyle=:solid)
    plot!(p, [NaN], [NaN], label="Emissive loading", color=:black, linestyle=:dash)
    p
end
@time plot_zalm2_spin_fidelity()

## Emissive loading
function zalm2_spin_density_matrix_emissive()
    μ = 0.2
    ηR = 0.01
    ηT = 0.8
    state = zalm2(μ, ηR, ηT)
    ρ = emissiveload(state) * 4
    
    display(ρ)
end
@time zalm2_spin_density_matrix_emissive()
