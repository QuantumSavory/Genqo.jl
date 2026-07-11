@testitem "SIGSAG v2 vs ground truth" setup = [GroundTruth] begin
    using Gabs

    GT = GroundTruth.GT
    P = GroundTruth.PARAMS["sigsag"]
    engine = HybridProjectionEngine(6)
    ψ⁺ = GroundTruth.bell4()

    # Heralding clicks on output modes 1, 2 with efficiency ηᵈ; the photon-photon
    # state lives in the traced-out measured modes 3–6 with efficiency ηᵗ
    Π = projector([1, 1, -1, -1, -1, -1])
    sigsag_η(ηᵗ, ηᵈ) = [ηᵈ, ηᵈ, ηᵗ, ηᵗ, ηᵗ, ηᵗ]

    for i in 1:GroundTruth.GT_NCASES
        μ, ηᵗ, ηᵈ, ηᵇ = P[i, :]
        cov = GT["sigsag/covariance"][:, :, i]

        # Circuit construction: one SPDC source plus two vacuum modes, interfered on
        # beamsplitters between modes (3,5) and (4,6)
        st4 = eprstate(QuadBlockBasis(4), asinh(√μ), Float64(π))
        apply!(st4, modeswap(QuadBlockBasis(2)), [2, 4])
        st = st4 ⊗ vacuumstate(QuadBlockBasis(2))
        bs = beamsplitter(QuadBlockBasis(2), 0.5)
        apply!(st, bs, [3, 5])
        apply!(st, bs, [4, 6])
        @test st.covar ≈ 2 .* cov rtol = 1e-12

        η = sigsag_η(ηᵗ, ηᵈ)
        ps = project(st, Π; engine, η = η)
        @test tr(ps) ≈ GT["sigsag/pgen"][i] rtol = 1e-9
        @test real(fidelity(ψ⁺, ps)) ≈ GT["sigsag/fidelity"][i] rtol = 1e-9

        # Same results from the legacy covariance wrapped directly
        ps_legacy = project(GroundTruth.legacy_state(cov), Π; engine, η = η)
        @test tr(ps_legacy) ≈ GT["sigsag/pgen"][i] rtol = 1e-9
        @test real(fidelity(ψ⁺, ps_legacy)) ≈ GT["sigsag/fidelity"][i] rtol = 1e-9
    end
end
