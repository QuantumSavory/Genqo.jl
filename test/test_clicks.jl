@testitem "ClickState construction" begin
    c = clicks([1, 0, 1, 0])
    @test c isa ClickStateKet
    @test c.coefs == [1.0 + 0.0im]
    @test c.clicks == [1 0 1 0]
    @test Genqo.nmodes(c) == 4
    @test Genqo.nmodes(c') == 4

    # Number of coefficients must match number of click patterns
    @test_throws ArgumentError ClickStateKet([1.0 + 0im, 0.5 + 0im], [1 0 1 0])
    @test_throws ArgumentError ClickStateBra([1.0 + 0im, 0.5 + 0im], [1 0 1 0])
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

@testitem "ClickState inner products" begin
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
end
