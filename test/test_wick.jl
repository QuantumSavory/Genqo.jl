@testitem "wick_partitions" begin
    using Genqo: wick_partitions

    doublefactorial(n) = prod(n:-2:1)
    for N in (2, 4, 6, 8)
        parts = wick_partitions(N)
        @test size(parts) == (doublefactorial(N - 1), 2, N ÷ 2)
        # Each partition is a perfect matching: every index of 1:N appears exactly once
        for p in axes(parts, 1)
            @test sort(vec(parts[p, :, :])) == 1:N
        end
    end
    @test_throws ArgumentError wick_partitions(3)
end

@testitem "wick_out reference contraction" begin
    using StableRNGs

    B = rand(StableRNG(1), ComplexF64, 6, 6)
    A = B + transpose(B) # Wick contraction assumes a symmetric kernel

    # Odd-order moments vanish
    @test wick_out([1, 2, 3], A) == 0
    # Second and fourth order against hand-written Isserlis expansions
    @test wick_out([1, 2], A) ≈ A[1, 2]
    @test wick_out([1, 2, 3, 4], A) ≈ A[1, 2] * A[3, 4] + A[1, 3] * A[2, 4] + A[1, 4] * A[2, 3]
    # Repeated indices are allowed in the reference path
    @test wick_out([1, 1, 2, 2], A) ≈ A[1, 1] * A[2, 2] + 2 * A[1, 2]^2
end

@testitem "W fast path matches symbolic path" begin
    using StableRNGs
    using Nemo

    # HybridProjector(6) provides 24 phase-space variables; its α/β are the standard
    # multilinear building blocks for moment polynomials
    proj = HybridProjector(6)
    α, β = proj.α, proj.β
    CC = Nemo.ComplexField()

    B = rand(StableRNG(2), ComplexF64, 24, 24)
    Ainv = B + transpose(B)

    # Mixed-degree multilinear polynomial: degrees 4 and 8, complex coefficients
    C = (2 + Nemo.onei(CC)) * α[1] * α[2] * β[1] * β[2] +
        CC(0.5) * α[1] * α[2] * α[3] * α[4] * β[1] * β[2] * β[3] * β[4]
    @test W(extract_W_terms(C), Ainv) ≈ W(C, Ainv)

    # Degree 12 exceeds the @generated hafnian's full-unroll limit (K=10), exercising
    # the recursive expansion against the partition-enumerating symbolic path
    C12 = prod(α) * prod(β)
    @test W(extract_W_terms(C12), Ainv) ≈ W(C12, Ainv)

    # Odd-degree monomials contract to zero
    @test W(extract_W_terms(α[1] * α[2] * β[1]), Ainv) ≈ 0 atol = 1e-14
end

@testitem "WTerms algebra" begin
    using StableRNGs

    proj = HybridProjector(2)
    α, β = proj.α, proj.β
    B = rand(StableRNG(3), ComplexF64, 8, 8)
    Ainv = B + transpose(B)

    t = extract_W_terms(α[1] * α[2] * β[1] * β[2])
    w = W(t, Ainv)

    @test W(2.5 * t, Ainv) ≈ 2.5 * w
    @test W(t * (0.0 + 2.0im), Ainv) ≈ 2.0im * w
    @test W(t + t, Ainv) ≈ 2 * w

    # The accumulate-from-zero pattern used by dot(::ClickStateBra, ...)
    acc = zero(WTerms{Tuple{}})
    @test W(acc, Ainv) == 0
    acc += (1.0 + 0.0im) * t
    acc += (0.0 + 1.0im) * t
    @test W(acc, Ainv) ≈ (1 + im) * w
end
