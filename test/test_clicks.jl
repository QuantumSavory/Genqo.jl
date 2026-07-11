@testitem "ClickState construction" begin
    c = clicks([1, 0, 1, 0])
    @test c isa ClickStateKet
    @test c.coefs == [1.0 + 0.0im]
    @test c.clicks == [1 0 1 0]
    @test Genqo.nmodes(c) == 4
    @test Genqo.nmodes(c') == 4

    # Kets and bras render as photon-number patterns
    @test occursin("|1,0,1,0⟩", repr(c))
    @test occursin("⟨1,0,1,0|", repr(c'))

    # Number of coefficients must match number of click patterns
    @test_throws ArgumentError ClickStateKet([1.0 + 0im, 0.5 + 0im], [1 0 1 0])
    @test_throws ArgumentError ClickStateBra([1.0 + 0im, 0.5 + 0im], [1 0 1 0])
    # Click states are photon-number patterns: entries must be non-negative
    # (traced-out modes belong to ClickProjector, not to kets/bras)
    @test_throws ArgumentError clicks([1, -1])
    @test_throws ArgumentError ClickStateBra([1.0 + 0im], [-1 0])
end

@testitem "ClickState algebra" begin
    a = clicks([1, 0])
    b = clicks([0, 1])

    s = (a + b) / √2
    @test s.coefs ≈ [1 / √2, 1 / √2]
    @test (a - b).coefs ≈ [1.0, -1.0]
    @test (2.0im * a).coefs == [2.0im]
    @test (a * 2.0im).coefs == [2.0im]
    @test (a + b) == (a + b)
    @test (a + b) ≈ (a + b * (1 + 1e-12))

    # adjoint is an involution and conjugates coefficients
    c = (1.0 + 2.0im) * a
    @test c' isa ClickStateBra
    @test c'.coefs == [1.0 - 2.0im]
    @test (c')' == c

    # bra algebra mirrors ket algebra
    @test (a' + b').coefs ≈ [1.0, 1.0]
    @test ((a' - b') / 2).coefs ≈ [0.5, -0.5]
end

@testitem "ClickState inner products and norm" begin
    using LinearAlgebra: norm

    a = clicks([1, 0, 1, 0])
    b = clicks([0, 1, 0, 1])

    # Distinct patterns are orthogonal
    @test dot(a', b) == 0
    @test dot(a', a) == 1

    # Superpositions: normalization and cross terms
    c1 = (a + im * b) / √2
    @test dot(c1', c1) ≈ 1.0
    c2 = (a - im * b) / √2
    @test dot(c2', c1) ≈ 0.0 atol = 1e-15
    @test dot(a', c1) ≈ 1 / √2

    @test norm(c1) ≈ 1.0
    @test norm(a + b) ≈ √2
    @test norm((a + b)') ≈ √2
end

@testitem "ClickProjector construction" begin
    P = projector([1, 0, 1, 0])
    @test P isa ClickProjector
    @test P.clicks == [1 0 1 0]
    @test Genqo.nmodes(P) == 4

    # Colons mark traced-out modes and are stored as -1
    @test projector([:, 1, 0, :]) == projector([-1, 1, 0, -1])
    # Projectors render one |pattern⟩⟨pattern| per term, with ':' for traced-out modes
    @test occursin("|:,1,0,:⟩⟨:,1,0,:|", repr(projector([:, 1, 0, :])))
    @test occursin("|1,0⟩⟨1,0|\n|0,1⟩⟨0,1|", repr(projector([1, 0]) + projector([0, 1])))
    # Entries must be Int or Colon
    @test_throws ArgumentError projector([1.5, 1])
    @test_throws ArgumentError projector(Any["a", 1])
    # Entries must be non-negative photon numbers or -1 for traceout
    @test_throws ArgumentError projector([1, -2])
    @test_throws ArgumentError ClickProjector([-3 0])
end

@testitem "ClickProjector algebra" begin
    Pa = projector([:, 1, 0])
    Pb = projector([:, 0, 1])

    # Projectors onto distinct click patterns may be summed (the sum is again a projector)
    @test (Pa + Pb) isa ClickProjector
    @test (Pa + Pb).clicks == [-1 1 0; -1 0 1]
    @test Pa + Pb == Pa + Pb
    @test Pa != Pb

    # Linear combinations other than sums are not part of the projector algebra
    @test_throws MethodError 2 * Pa
    @test_throws MethodError Pa * 2
    @test_throws MethodError Pa / 2
    @test_throws MethodError Pa - Pb
    @test_throws MethodError Pa'
end

@testitem "ClickProjector traceout consistency" begin
    # Summed projectors must trace out the same modes in every term
    @test (projector([:, 1, 0]) + projector([:, 0, 1])) isa ClickProjector
    @test_throws ArgumentError projector([:, 1, 0]) + projector([1, :, 0])
    @test_throws ArgumentError projector([1, 0]) + projector([:, 0])

    # Also caught for three or more terms, where only one pair disagrees
    consistent = projector([:, 1, 0]) + projector([:, 0, 1]) + projector([:, 0, 0])
    @test consistent isa ClickProjector
    @test_throws ArgumentError projector([:, 1, 0]) + projector([1, :, 0]) + projector([0, 0, :])
    @test_throws ArgumentError projector([:, 1, 0]) + projector([:, 0, 1]) + projector([1, 0, :])

    # And when constructing the pattern matrix directly
    @test_throws ArgumentError ClickProjector([-1 0; 0 -1])
end
