@testitem "modeswap" begin
    using Gabs
    using LinearAlgebra

    for basis in (QuadPairBasis(2), QuadBlockBasis(2))
        ms = modeswap(basis)
        S = ms.symplectic
        # Symplectic and self-inverse
        @test S * symplecticform(basis) * S' ≈ symplecticform(basis)
        @test S * S ≈ I
    end

    # Both orderings represent the same operation
    @test changebasis(QuadBlockBasis, modeswap(QuadPairBasis(2))).symplectic ≈
          modeswap(QuadBlockBasis(2)).symplectic

    # Swapping the modes of an asymmetric product state swaps the covariance blocks
    st = squeezedstate(QuadBlockBasis(1), 0.4, 0.0) ⊗ vacuumstate(QuadBlockBasis(1))
    swapped = copy(st)
    apply!(swapped, modeswap(QuadBlockBasis(2)))
    @test swapped.covar[1, 1] ≈ st.covar[2, 2]
    @test swapped.covar[2, 2] ≈ st.covar[1, 1]
    @test swapped.covar[3, 3] ≈ st.covar[4, 4]
    @test swapped.covar[4, 4] ≈ st.covar[3, 3]
end

@testitem "greenmachine" begin
    using Gabs
    using LinearAlgebra

    for basis in (QuadPairBasis(4), QuadBlockBasis(4))
        gm = greenmachine(basis, 4)
        S = gm.symplectic
        @test S * symplecticform(basis) * S' ≈ symplecticform(basis)
        # Hadamard-transform structure: self-inverse
        @test S * S ≈ I
    end

    @test changebasis(QuadBlockBasis, greenmachine(QuadPairBasis(4), 4)).symplectic ≈
          greenmachine(QuadBlockBasis(4), 4).symplectic

    # A 2-mode Green machine is a 50/50 interferometer: q1 → (q1+q2)/√2
    gm2 = greenmachine(QuadBlockBasis(2), 2)
    @test gm2.symplectic[1, 1:2] ≈ [1 / √2, 1 / √2]

    # Mode count must match the basis and be a power of two
    @test_throws ArgumentError greenmachine(QuadBlockBasis(4), 2)
    @test_throws ArgumentError greenmachine(QuadBlockBasis(3), 3)
    @test_throws ArgumentError greenmachine(QuadPairBasis(3), 3)
end
