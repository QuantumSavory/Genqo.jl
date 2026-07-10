@testitem "HybridProjector construction" begin
    proj = HybridProjector(3)
    @test proj.mds == 3
    @test length(proj.α) == 3
    @test length(proj.β) == 3
    @test isempty(proj.C_poly_cache)
end

@testitem "project argument validation" begin
    using Gabs

    proj = HybridProjector(2)
    st = eprstate(QuadBlockBasis(2), asinh(√1e-2), Float64(π))

    # State and projector must have the same number of modes
    @test_throws ArgumentError project(vacuumstate(QuadBlockBasis(3)), proj, [1, 1, 1])
    # Detector outcomes must have one entry per mode
    @test_throws ArgumentError project(st, proj, [1])
    @test_throws ArgumentError project(st, proj, [1, 1, 0])
    # Loss vector must have one entry per mode and lie in [0, 1]
    @test_throws ArgumentError project(st, proj, [1, 1]; η = [0.9])
    @test_throws ArgumentError project(st, proj, [1, 1]; η = [0.9, 1.1])
    @test_throws ArgumentError project(st, proj, [1, 1]; η = [-0.1, 0.9])
    # Outcomes are restricted to {-1, 0, 1}
    @test_throws ArgumentError project(st, proj, [2, 1])
    @test_throws ArgumentError project(st, proj, [1, -2])
    # Only pure Gaussian states are supported
    @test_throws ArgumentError project(thermalstate(QuadBlockBasis(2), 2), proj, [1, 1])
    # Colon-form outcomes only accept Int or Colon entries
    @test_throws ArgumentError project(st, proj, [:, 1.5])

    # Valid calls construct a projected state
    @test project(st, proj, [1, 1]) isa ProjectedPureGaussianState
    @test project(st, proj, [:, 1]) isa ProjectedPureGaussianState
end

@testitem "projected state trace" begin
    using Gabs

    proj = HybridProjector(2)
    st = eprstate(QuadBlockBasis(2), asinh(√1e-1), Float64(π))

    # Coincidence probability is a real number in (0, 1]
    p = tr(project(st, proj, [1, 1]; η = [0.9, 0.8]))
    @test p isa Float64
    @test 0 < p < 1

    # Tracing out every mode recovers the full trace of the density matrix
    @test tr(project(st, proj, [-1, -1])) ≈ 1.0

    # Colon shorthand is equivalent to -1 outcomes
    ps_colon = project(st, proj, [:, :])
    @test tr(ps_colon) ≈ 1.0

    # Repeated evaluation hits the compiled-polynomial cache and reproduces the result
    @test !isempty(proj.C_poly_cache)
    @test tr(project(st, proj, [1, 1]; η = [0.9, 0.8])) == p
end

@testitem "projected state matrix elements and fidelity" begin
    using Gabs

    proj = HybridProjector(2)
    st = eprstate(QuadBlockBasis(2), asinh(√1e-1), Float64(π))
    ps = project(st, proj, [-1, -1]; η = [0.9, 0.8])

    # Bra/ket mode count must match the number of traced-out modes
    @test_throws ArgumentError dot(clicks([1])', ps, clicks([1]))
    ps_partial = project(st, proj, [1, -1]; η = [0.9, 0.8])
    @test_throws ArgumentError dot(clicks([1, 0])', ps_partial, clicks([1, 0]))

    # Diagonal elements are real populations; off-diagonal pairs are conjugate
    p11 = dot(clicks([1, 1])', ps, clicks([1, 1]))
    @test imag(p11) ≈ 0 atol = 1e-14
    @test real(p11) > 0
    v00 = clicks([0, 0])
    v11 = clicks([1, 1])
    @test dot(v00', ps, v11) ≈ conj(dot(v11', ps, v00))

    # fidelity is dot normalized by the trace, for both ket and bra targets
    ψ = (v00 + v11) / √2
    @test fidelity(ψ, ps) ≈ dot(ψ', ps, ψ) / tr(ps)
    @test fidelity(ψ', ps) ≈ fidelity(ψ, ps)

    # Colon and Vector{Int} forms give identical physics
    ps_colon = project(st, proj, [:, :]; η = [0.9, 0.8])
    @test dot(ψ', ps_colon, ψ) ≈ dot(ψ', ps, ψ)
end

@testitem "to_fock" begin
    using Gabs
    using LinearAlgebra: ishermitian, diag

    proj = HybridProjector(2)
    st = eprstate(QuadBlockBasis(2), asinh(√1e-1), Float64(π))
    ps = project(st, proj, [-1, -1]; η = [0.9, 0.8])

    dm = to_fock(ps; cutoff = 1)
    @test size(dm.data) == (4, 4)
    @test dm.data ≈ dm.data'
    @test all(real.(diag(dm.data)) .>= 0)

    # Basis ordering: index of pattern (n₁, n₂) with cutoff 1 is 1 + n₁ + 2n₂
    for (i, pat) in enumerate([[0, 0], [1, 0], [0, 1], [1, 1]])
        @test dm.data[i, i] ≈ dot(clicks(pat)', ps, clicks(pat))
    end

    # Truncated trace is bounded by (and for small μ close to) the full trace
    @test real(sum(diag(dm.data))) <= tr(ps) + 1e-12
end
