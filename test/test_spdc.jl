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
        # The legacy fidelity carries an extra (ηᵗηᵈ)² prefactor — (ηᵗηᵈ)⁴ in total —
        # relative to v2's per-photon loss weighting η^((bra+ket)/2), which matches the
        # convention used by the ZALM/SIGSAG legacy code.
        ps = project(st, projector([-1, -1, -1, -1]); engine, η = fill(ηᵗ * ηᵈ, 4))
        @test real(dot(ψ⁺', ps, ψ⁺)) * (ηᵗ * ηᵈ)^2 ≈ GT["spdc/fidelity"][i] rtol = 1e-9
    end
end
