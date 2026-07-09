using Genqo
using Gabs

using BenchmarkTools
using Plots


## ZALM calculations

function zalm_metrics(μ::Float64, ηᵗ::Float64, ηᵈ::Float64; proj::HybridProjector, metric::Symbol=:none)
    basis4 = QuadBlockBasis(4)
    basis8 = QuadBlockBasis(8)
    st = eprstate(basis8, asinh(√μ), 0.)
    ms = modeswap(basis4)
    bs = beamsplitter(basis4, 0.5)

    apply!(st, ms, [2,4, 5,7])
    apply!(st, bs, [3,5, 4,6])

    η = [ηᵗ,ηᵗ,ηᵈ,ηᵈ,ηᵈ,ηᵈ,ηᵗ,ηᵗ]
    st_heralded = project(st, proj, [:,:,1,1,0,0,:,:]; η=η) # of some intermediate type just containing detection/loss/Gaussian state information, but not converted to a density matrix yet

    # Probability of success
    Pgen = tr(st_heralded) # shortcuts density matrix calculation
    if metric == :Pgen
        return Pgen
    end

    # Fidelity
    ψ⁺ = (clicks([1,0,0,1]) + clicks([0,1,1,0])) / √2
    F = real(dot(ψ⁺', st_heralded, ψ⁺)) / Pgen # also shortcuts density matrix calculation
    if metric == :F
        return F
    end

    ρ_photon = to_fock(st_heralded; cutoff=1) # user explicitly converts to fock basis and specifies cutoff
    if metric == :ρ_photon
        return ρ_photon
    end

    # memory = DuanKimble()
    # st_loaded = load_state(st_heralded, memory)
    # ρ_spin = 

    if metric == :all
        return (Pgen=Pgen, F=F, ρ_photon=ρ_photon)
    end
end

function benchmark_zalm()
    proj = HybridProjector(8)
    μ = logrange(1e-4, 1e2, 100)
    ηᵗ = 1.0
    ηᵈ = 0.8
    @btime zalm_metrics.($μ, $ηᵗ, $ηᵈ; proj=$proj)
end
benchmark_zalm()

## Verify v2 matches v1

function verify_zalm_Pgen()
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    proj = HybridProjector(8)

    zalm_Pg_ground = zalm.probability_success.(μ, η', 1, η', 0)
    zalm_Pg_new = zalm_metrics.(μ, η', η'; proj=proj, metric=:Pgen)

    plot(μ, zalm_Pg_ground, label="Genqo v1 (ground truth)", xscale=:log10, yscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Probability of Success", legend=:bottomright, color=[1 2 3 4])
    plot!(μ, zalm_Pg_new, label="Genqo v2", linestyle=:dash, color=[1 2 3 4])
    plot!(μ, μ.^2 ./ (μ.+1).^6, linestyle=:dash, color=:black)
end
verify_zalm_Pgen()

function verify_zalm_F()
    μ = logrange(1e-4, 10, 100)
    η = 10 .^ -([0, 3, 6, 9]/10)
    proj = HybridProjector(8)

    zalm_F_ground = zalm.fidelity.(μ, η', 1, η')
    zalm_F_new = zalm_metrics.(μ, η', η'; proj=proj, metric=:F)
    plot(μ, zalm_F_ground, label="Genqo v1 (ground truth)", xscale=:log10, xlabel="Mean Photon Number Per Mode", ylabel="Fidelity", legend=:topleft, color=[1 2 3 4])
    plot!(μ, zalm_F_new, label="Genqo v2", linestyle=:dash, color=[1 2 3 4])
end
verify_zalm_F()
