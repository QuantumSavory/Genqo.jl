@testitem "HybridProjectionEngine construction" begin
    engine = HybridProjectionEngine(3)
    α, βc = get_phase_space_generators_half(engine)
    @test engine.mds == 3
    @test length(α) == 3
    @test length(βc) == 3
    @test isempty(engine.C_poly_cache)
end

@testitem "project argument validation" begin
    using Gabs

    engine = HybridProjectionEngine(2)
    st = eprstate(QuadBlockBasis(2), asinh(√1e-2), Float64(π))

    # State, projector, and loss vector must have the same number of modes
    @test_throws ArgumentError project(st, projector([1]))
    @test_throws ArgumentError project(st, projector([1, 1, 0]))
    @test_throws ArgumentError project(st, projector([1, 1]); η = [0.9])
    # Loss vector must lie in [0, 1]
    @test_throws ArgumentError project(st, projector([1, 1]); η = [0.9, 1.1])
    @test_throws ArgumentError project(st, projector([1, 1]); η = [-0.1, 0.9])
    # Detector outcomes are restricted to {-1, 0, 1} (higher photon numbers are a TODO)
    @test_throws ArgumentError project(st, projector([2, 1]))
    # Only pure Gaussian states are supported
    @test_throws ArgumentError project(thermalstate(QuadBlockBasis(2), 2), projector([1, 1]))

    # Valid calls construct a projected state
    @test project(st, projector([1, 1])) isa ProjectedPureGaussianState
    @test project(st, projector([:, 1])) isa ProjectedPureGaussianState
    # Sums of click projectors are valid detector outcomes
    Π = projector([1, 0]) + projector([0, 1])
    @test project(st, Π) isa ProjectedPureGaussianState
end

@testitem "projected state trace" begin
    using Gabs

    engine = HybridProjectionEngine(2)
    st = eprstate(QuadBlockBasis(2), asinh(√1e-1), Float64(π))
    η = [0.9, 0.8]

    # Coincidence probability is a real number in (0, 1]
    p = tr(project(st, projector([1, 1]); η = η); engine)
    @test p isa Float64
    @test 0 < p < 1

    # Tracing out every mode recovers the full trace of the density matrix
    @test tr(project(st, projector([-1, -1])); engine) ≈ 1.0
    # Colon shorthand is equivalent to -1 outcomes
    @test tr(project(st, projector([:, :])); engine) ≈ 1.0

    # The trace is additive over the patterns of a summed projector
    Pa, Pb = projector([1, 0]), projector([0, 1])
    p_sum = tr(project(st, Pa + Pb; η = η); engine)
    @test p_sum ≈ tr(project(st, Pa; η = η); engine) + tr(project(st, Pb; η = η); engine)
    # Together with the mutually exclusive complements, probabilities sum to 1
    total = p_sum + tr(project(st, projector([0, 0]) + projector([1, 1]); η = η); engine)
    @test total < 1 # cutoff at 1 photon per mode leaves out higher-number events

    # Repeated evaluation hits the compiled-polynomial cache and reproduces the result
    @test !isempty(engine.C_poly_cache)
    @test tr(project(st, projector([1, 1]); η = η); engine) == p
end

@testitem "projected state matrix elements and fidelity" begin
    using Gabs

    engine = HybridProjectionEngine(2)
    st = eprstate(QuadBlockBasis(2), asinh(√1e-1), Float64(π))
    ps = project(st, projector([-1, -1]); η = [0.9, 0.8])

    # Bra/ket mode count must match the number of traced-out modes
    @test_throws ArgumentError dot(clicks([1])', ps, clicks([1]); engine)
    ps_partial = project(st, projector([1, -1]); η = [0.9, 0.8])
    @test_throws ArgumentError dot(clicks([1, 0])', ps_partial, clicks([1, 0]); engine)

    # Diagonal elements are real populations; off-diagonal pairs are conjugate
    p11 = dot(clicks([1, 1])', ps, clicks([1, 1]); engine)
    @test imag(p11) ≈ 0 atol = 1e-14
    @test real(p11) > 0
    v00 = clicks([0, 0])
    v11 = clicks([1, 1])
    @test dot(v00', ps, v11; engine) ≈ conj(dot(v11', ps, v00; engine))

    # fidelity is dot normalized by the trace, for both ket and bra targets
    ψ = (v00 + v11) / √2
    @test fidelity(ψ, ps) ≈ dot(ψ', ps, ψ; engine) / tr(ps; engine)
    @test fidelity(ψ', ps) ≈ fidelity(ψ, ps)

    # Colon and -1 outcome forms give identical physics
    ps_colon = project(st, projector([:, :]); η = [0.9, 0.8])
    @test dot(ψ', ps_colon, ψ; engine) ≈ dot(ψ', ps, ψ; engine)
end

@testitem "duankimble argument validation and basic properties" begin
    using Gabs
    using LinearAlgebra: diag

    engine = HybridProjectionEngine(2)
    st = eprstate(QuadBlockBasis(2), asinh(√1e-1), Float64(π))
    ps = project(st, projector([-1, -1]); η = [0.9, 0.8])

    # One memory formed from the two traced-out modes: 2x2 spin density matrix
    ρ = duankimble(ps, [1, 0]; engine)
    @test size(ρ.data) == (2, 2)
    @test ρ.data ≈ ρ.data'
    @test all(real.(diag(ρ.data)) .≥ 0)
    # Explicit mode pairing matches the default consecutive pairing of free modes
    @test duankimble(ps, [1, 0], [(1, 2)]; engine).data ≈ ρ.data

    # One outcome per loaded mode is required
    @test_throws ArgumentError duankimble(ps, [1, 0, 1]; engine)
    @test_throws ArgumentError duankimble(ps, [1]; engine)
    # All loaded modes must be traced out in the projector
    ps_clicked = project(st, projector([1, -1]))
    @test_throws ArgumentError duankimble(ps_clicked, [1, 0], [(1, 2)]; engine)
    # An odd number of free modes cannot be paired into memories
    engine3 = HybridProjectionEngine(3)
    ps3 = project(vacuumstate(QuadBlockBasis(3)), projector([-1, -1, -1]))
    @test_throws ArgumentError duankimble(ps3, [1, 0]; engine)
end

@testitem "to_fock" begin
    using Gabs
    using LinearAlgebra: diag

    engine = HybridProjectionEngine(2)
    st = eprstate(QuadBlockBasis(2), asinh(√1e-1), Float64(π))
    ps = project(st, projector([-1, -1]); η = [0.9, 0.8])

    dm = to_fock(ps; engine, cutoff = 1)
    @test size(dm.data) == (4, 4)
    @test dm.data ≈ dm.data'
    @test all(real.(diag(dm.data)) .>= 0)

    # Basis ordering: index of pattern (n₁, n₂) with cutoff 1 is 1 + n₁ + 2n₂
    for (i, pat) in enumerate([[0, 0], [1, 0], [0, 1], [1, 1]])
        @test dm.data[i, i] ≈ dot(clicks(pat)', ps, clicks(pat); engine)
    end

    # Truncated trace is bounded by (and for small μ close to) the full trace
    @test real(sum(diag(dm.data))) <= tr(ps; engine) + 1e-12
end

@testitem "block inversion of A" begin
    using Genqo: _invA, _invA_UL
    using Gabs
    using LinearAlgebra: I, Diagonal, norm

    # Rebuild A from σ the long way round, to give the closed forms in _invA /
    # _invA_UL something independent to be checked against.
    function A_matrix(σ, η, n)
        mds = size(σ, 1) ÷ 2
        Γinv = inv(σ + 0.5*I)

        # Views of Γinv blocks
        a  = @view Γinv[1:mds,      1:mds     ]
        c  = @view Γinv[1:mds,      mds+1:2mds]

        cᵀ = @view Γinv[mds+1:2mds, 1:mds     ]
        b  = @view Γinv[mds+1:2mds, mds+1:2mds]

        @assert 0.5 * (a + b - im*(c - cᵀ)) ≈ I # Ã purity check

        C̃ = 0.5 * (a - b + im*(c + cᵀ))
        y = ones(ComplexF64, mds)
        y[n .!== -1] -= η[n .!== -1] # y_i = 1 - η_i for detected modes, y_i = 1 for transmitted modes
        Y = Diagonal(y)

        A = zeros(ComplexF64, 4mds, 4mds)

        copyto!(view(A, 1:mds,       mds+1:2mds ), -Y)
        copyto!(view(A, mds+1:2mds,  1:mds      ), -Y)
        copyto!(view(A, 1:mds,       2mds+1:3mds), I)
        copyto!(view(A, mds+1:2mds,  3mds+1:4mds), I)
        copyto!(view(A, 2mds+1:3mds, 1:mds      ), I)
        copyto!(view(A, 3mds+1:4mds, mds+1:2mds ), I)
        copyto!(view(A, 2mds+1:3mds, 2mds+1:3mds), C̃) # C̃
        conj!(C̃)
        copyto!(view(A, 3mds+1:4mds, 3mds+1:4mds), C̃) # C̃*
        A
    end

    mds = 6
    η = [0.7,0.7,0.05,0.05,0.05,0.05]
    n = [1,1,-1,-1,-1,-1]

    function bsm_cov(phases)
        st = eprstate(QuadBlockBasis(4), asinh(√1e-1), 0.) ⊗ vacuumstate(QuadBlockBasis(2))
        apply!(st, [2,4], modeswap(QuadBlockBasis(2)))
        apply!(st, [3,5, 4,6], beamsplitter(QuadBlockBasis(4), 0.5))
        for (mode, φ) in phases
            apply!(st, [mode], phaseshift(QuadBlockBasis(1), φ))
        end
        st.covar / st.ħ
    end

    # A real C̃ makes the whole of A⁻¹ real, and a C̃ carrying only a global phase
    # has that phase cancel in GC̃YC̃*, so either one hides a mix-up between the
    # conjugate-paired blocks of the closed form. Unequal phase shifts on separate
    # modes leave those blocks genuinely complex, which pins them down.
    phased = (1 => π/5, 3 => π/3)
    for phases in ((), phased)
        σ = bsm_cov(phases)

        invA, denom = _invA(σ, η, n)
        invA_UL, denom_UL = _invA_UL(σ, η, n)

        @test invA[1:2mds, 1:2mds] ≈ invA_UL
        @test denom ≈ denom_UL
        @test A_matrix(σ, η, n) * invA ≈ I
    end

    # Guard the guard: if the phased covariance ever stops making the off-diagonal
    # blocks complex, the case above silently goes back to proving nothing.
    invA, _ = _invA(bsm_cov(phased), η, n)
    @test norm(imag(invA[1:mds, mds+1:2mds])) > 1e-3
end
