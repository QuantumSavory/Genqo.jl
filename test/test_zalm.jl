@testitem "ZALM v2 vs ground truth" setup = [GroundTruth] begin
    using Gabs

    GT = GroundTruth.GT
    P = GroundTruth.PARAMS["zalm"]
    engine = HybridProjectionEngine(8)
    ψ⁺ = GroundTruth.bell4()

    # Signal modes (1, 2, 7, 8) are kept for the photon-photon state; BSM heralds on
    # clicks in modes 3, 4 and vacuum in modes 5, 6 (cf. zalm.moment_vector.trc)
    Π = projector([-1, -1, 1, 1, 0, 0, -1, -1])
    zalm_η(ηᵗ, ηᵈ, ηᵇ) = [ηᵗ * ηᵈ, ηᵗ * ηᵈ, ηᵇ, ηᵇ, ηᵇ, ηᵇ, ηᵗ * ηᵈ, ηᵗ * ηᵈ]

    for i in 1:GroundTruth.GT_NCASES
        μ, ηᵗ, ηᵈ, ηᵇ = P[i, :]
        cov = GT["zalm/covariance"][:, :, i]

        # Circuit construction: two SPDC sources (EPR pairs + polarization swaps),
        # BSM beamsplitters between modes (3,5) and (4,6)
        st = eprstate(QuadBlockBasis(8), asinh(√μ), Float64(π))
        ms = modeswap(QuadBlockBasis(2))
        bs = beamsplitter(QuadBlockBasis(2), 0.5)
        apply!(st, ms, [2, 4])
        apply!(st, ms, [5, 7])
        apply!(st, bs, [3, 5])
        apply!(st, bs, [4, 6])
        @test st.covar ≈ 2 .* cov rtol = 1e-12

        η = zalm_η(ηᵗ, ηᵈ, ηᵇ)
        ps = project(st, Π; engine, η = η)
        @test tr(ps) ≈ GT["zalm/pgen"][i] rtol = 1e-9
        @test real(fidelity(ψ⁺, ps)) ≈ GT["zalm/fidelity"][i] rtol = 1e-9

        # Same results from the legacy covariance wrapped directly
        ps_legacy = project(GroundTruth.legacy_state(cov), Π; engine, η = η)
        @test tr(ps_legacy) ≈ GT["zalm/pgen"][i] rtol = 1e-9
        @test real(fidelity(ψ⁺, ps_legacy)) ≈ GT["zalm/fidelity"][i] rtol = 1e-9

        # Colon shorthand for the traced-out signal modes is equivalent
        ps_colon = project(st, projector([:, :, 1, 1, 0, 0, :, :]); engine, η = η)
        @test tr(ps_colon) ≈ tr(ps)
    end
end

@testitem "ZALM Duan-Kimble loading vs ground truth" setup = [GroundTruth] begin
    using Gabs

    GT = GroundTruth.GT
    P = GroundTruth.PARAMS["zalm"]
    engine = HybridProjectionEngine(8)
    nvec = GroundTruth.GT_NVEC_ZALM

    # KNOWN DISCREPANCY (hence broken=true): v2 duankimble is currently a factor of 4
    # too small relative to the legacy ZALM spin_density_matrix — the trailing /4
    # normalization in duankimble matches SPDC's legacy Coef = 1/(4·D1·D2·D3), but the
    # legacy ZALM Coef has no 1/4. Once the implementations agree, drop broken=true.
    for i in 1:GroundTruth.GT_NCASES
        μ, ηᵗ, ηᵈ, ηᵇ = P[i, :]
        st = GroundTruth.legacy_state(GT["zalm/covariance"][:, :, i])
        η = [ηᵗ * ηᵈ, ηᵗ * ηᵈ, ηᵇ, ηᵇ, ηᵇ, ηᵇ, ηᵗ * ηᵈ, ηᵗ * ηᵈ]
        ps = project(st, projector([-1, -1, 1, 1, 0, 0, -1, -1]); engine, η = η)

        # Loaded memories pair the signal modes (1,2) and (7,8); the BSM click
        # pattern comes from the projector (cf. nvec[3:6])
        ρ = duankimble(ps, nvec[[1, 2, 7, 8]])
        @test size(ρ.data) == (4, 4)
        @test ρ.data ≈ GT["zalm/sdm"][:, :, i] rtol = 1e-9 broken = true
    end
end

@testitem "ZALM to_fock density matrix" setup = [GroundTruth] begin
    using Gabs
    using LinearAlgebra: diag

    GT = GroundTruth.GT
    μ, ηᵗ, ηᵈ, ηᵇ = GroundTruth.PARAMS["zalm"][1, :] # lossless case only; to_fock is O(4^2m) dots
    engine = HybridProjectionEngine(8)
    st = GroundTruth.legacy_state(GT["zalm/covariance"][:, :, 1])
    ps = project(st, projector([-1, -1, 1, 1, 0, 0, -1, -1]); engine)

    dm = to_fock(ps; cutoff = 1)
    @test size(dm.data) == (16, 16)
    @test dm.data ≈ dm.data'
    @test all(real.(diag(dm.data)) .>= -1e-15)

    # ⟨ψ⁺|ρ|ψ⁺⟩ assembled from Fock matrix elements matches the direct dot():
    # with cutoff 1, pattern (n₁,n₂,n₃,n₄) sits at index 1 + n₁ + 2n₂ + 4n₃ + 8n₄,
    # so |1001⟩ → 10 and |0110⟩ → 7
    ψ⁺ = GroundTruth.bell4()
    overlap_fock = (dm.data[10, 10] + dm.data[10, 7] + dm.data[7, 10] + dm.data[7, 7]) / 2
    @test overlap_fock ≈ dot(ψ⁺', ps, ψ⁺)
end
