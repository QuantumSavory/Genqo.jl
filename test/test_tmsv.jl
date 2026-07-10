@testitem "TMSV v2 vs ground truth" setup = [GroundTruth] begin
    using Gabs

    GT = GroundTruth.GT
    P = GroundTruth.PARAMS["tmsv"]
    proj = HybridProjector(2)

    for i in 1:GroundTruth.GT_NCASES
        μ, ηᵗ, ηᵈ, ηᵇ = P[i, :]
        cov = GT["tmsv/covariance"][:, :, i]

        # Circuit construction: an EPR pair with θ=π matches the legacy squeezing sign
        # (legacy TMSV has +√(μ(μ+1)) in the qq off-diagonal); Gabs uses ħ=2, legacy ħ=1
        st = eprstate(QuadBlockBasis(2), asinh(√μ), Float64(π))
        @test st.covar ≈ 2 .* cov rtol = 1e-12

        # Coincidence probability: both photons detected with efficiency ηᵈ
        outcomes, η = [1, 1], [ηᵈ, ηᵈ]
        @test tr(project(st, proj, outcomes; η = η)) ≈ GT["tmsv/pgen"][i] rtol = 1e-9
        # Same result from the legacy covariance wrapped directly
        st_legacy = GroundTruth.legacy_state(cov)
        @test tr(project(st_legacy, proj, outcomes; η = η)) ≈ GT["tmsv/pgen"][i] rtol = 1e-9
    end
end
