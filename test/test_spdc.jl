@testitem "SPDC v2 vs ground truth" setup = [GroundTruth] begin
    using Gabs

    GT = GroundTruth.GT
    P = GroundTruth.PARAMS["spdc"]
    engine = HybridProjectionEngine(4)
    ψ⁺ = GroundTruth.bell4()

    for i in 1:GroundTruth.GT_NCASES
        μ, ηᵗ, ηᵈ, ηᵇ = P[i, :]
        cov = GT["spdc/covariance"][:, :, i]

        # Circuit construction: two EPR pairs (1,2)(3,4), then swapping modes 2 and 4
        # yields the SPDC polarization pairing (1,4)(2,3)
        st = eprstate(QuadBlockBasis(4), asinh(√μ), Float64(π))
        apply!(st, modeswap(QuadBlockBasis(2)), [2, 4])
        @test st.covar ≈ 2 .* cov rtol = 1e-12

        # Bell-state overlap of the raw (unheralded) source: all four modes kept.
        ps = project(st, projector([-1, -1, -1, -1]); engine, η = fill(ηᵗ * ηᵈ, 4))
        @test real(dot(ψ⁺', ps, ψ⁺)) ≈ GT["spdc/fidelity"][i] rtol = 1e-9
    end
end

@testitem "SPDC Duan-Kimble loading vs ground truth" setup = [GroundTruth] begin
    using Gabs

    GT = GroundTruth.GT
    P = GroundTruth.PARAMS["spdc"]
    engine = HybridProjectionEngine(4)
    nvec = GroundTruth.GT_NVEC_SPDC

    for i in 1:GroundTruth.GT_NCASES
        μ, ηᵗ, ηᵈ, ηᵇ = P[i, :]

        st = eprstate(QuadBlockBasis(4), asinh(√μ), Float64(π))
        apply!(st, modeswap(QuadBlockBasis(2)), [2, 4])
        ps = project(st, projector([-1, -1, -1, -1]); engine, η = fill(ηᵗ * ηᵈ, 4))

        # Duan-Kimble loading of the raw source into two memories, pairing modes
        # (1,2) and (3,4), reproduces the legacy spin-spin density matrix
        ρ = duankimble(ps, nvec)
        @test size(ρ.data) == (4, 4)
        @test ρ.data ≈ GT["spdc/sdm"][:, :, i] rtol = 1e-9
        # The default consecutive pairing of the traced-out modes is explicit here
        @test duankimble(ps, nvec, [(1, 2), (3, 4)]).data ≈ ρ.data
    end
end
