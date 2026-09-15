@testitem "LazyDensityMatrix defers and memoizes" begin
    using LinearAlgebra: tr, diag

    calls = Ref(0)
    A = LazyDensityMatrix((i, j) -> (calls[] += 1; ComplexF64(10i + j)), 4)

    @test size(A) == (4, 4)
    @test eltype(A) == ComplexF64
    @test Genqo.ncomputed(A) == 0
    @test calls[] == 0 # construction evaluates nothing

    # A single read evaluates exactly one entry
    @test A[2, 3] == 23
    @test calls[] == 1
    @test Genqo.ncomputed(A) == 1

    # Re-reading is memoized, not recomputed
    @test A[2, 3] == 23
    @test calls[] == 1

    # tr touches the diagonal only: 4 entries, not 16
    calls[] = 0
    @test tr(A) == 11 + 22 + 33 + 44
    @test calls[] == 4
    @test Genqo.ncomputed(A) == 5 # the 4 diagonal entries plus [2,3] from before

    # diag likewise reuses the already-computed diagonal
    calls[] = 0
    @test diag(A) == ComplexF64[11, 22, 33, 44]
    @test calls[] == 0

    # Materializing fills the rest exactly once and yields a plain dense Matrix
    calls[] = 0
    M = Matrix(A)
    @test M isa Matrix{ComplexF64}
    @test M == ComplexF64[10i + j for i in 1:4, j in 1:4]
    @test calls[] == 16 - 5
    @test Genqo.ncomputed(A) == 16
    @test collect(A) == M
end

@testitem "LazyDensityMatrix behaves like a dense matrix" begin
    using LinearAlgebra: tr

    f = (i, j) -> ComplexF64(i == j ? i : 0.5(i + j))
    A = LazyDensityMatrix(f, 3)
    D = ComplexF64[f(i, j) for i in 1:3, j in 1:3]

    @test A == D
    @test A ≈ D
    @test Matrix(A) == D
    @test A[1, :] == D[1, :]
    @test A[:, 2] == D[:, 2]
    @test A[CartesianIndex(2, 3)] == D[2, 3]
    @test A[5] == D[5] # linear indexing over an IndexCartesian array
    @test tr(A) == tr(D)
    @test A' == D'
    @test sum(A) == sum(D)

    @test_throws BoundsError A[4, 1]
    @test_throws BoundsError A[1, 4]
    @test_throws ArgumentError LazyDensityMatrix(f, -1)

    # setindex! overrides an entry and marks it computed
    B = LazyDensityMatrix((i, j) -> error("should not be evaluated"), 2)
    B[1, 1] = 7.0 + 0im
    @test B[1, 1] == 7
    @test Genqo.ncomputed(B) == 1
end

@testitem "LazyDensityMatrix is safe to read concurrently" begin
    # Each entry must be produced exactly once even when many threads race for it.
    calls = Threads.Atomic{Int}(0)
    A = LazyDensityMatrix(16) do i, j
        Threads.atomic_add!(calls, 1)
        ComplexF64(100i + j)
    end

    results = Vector{Vector{ComplexF64}}(undef, 8)
    Threads.@threads for t in 1:8
        results[t] = [A[i, j] for j in 1:16 for i in 1:16]
    end

    expected = [ComplexF64(100i + j) for j in 1:16 for i in 1:16]
    @test all(r == expected for r in results)
    @test calls[] == 16 * 16
    @test Genqo.ncomputed(A) == 16 * 16
end

@testitem "lazily-backed operators support the dense operator API" begin
    using Gabs
    using LinearAlgebra: tr
    using QuantumOpticsBase: DenseOpType, dense, entropy_vn, entropy_renyi, ptrace, expect, variance

    engine = HybridProjectionEngine(8)
    st = eprstate(QuadBlockBasis(8), asinh(√1e-2), Float64(π))
    apply!(st, [2, 4], modeswap(QuadBlockBasis(2)))
    apply!(st, [5, 7], modeswap(QuadBlockBasis(2)))
    apply!(st, [3, 5], beamsplitter(QuadBlockBasis(2), 0.5))
    apply!(st, [4, 6], beamsplitter(QuadBlockBasis(2), 0.5))
    ps = project(st, projector([-1, -1, 1, 1, 0, 0, -1, -1]); η = fill(0.8, 8))
    mk() = duankimble(ps, [1, 0, 1, 0]; engine)

    ρd = dense(mk())
    @test ρd isa DenseOpType
    @test ρd.data == Matrix(mk().data)

    # Arithmetic stays lazy: scaling a density matrix is not necessarily a prelude to reading
    # all of it, so `tr(ρ * 4)` must still cost only the diagonal.
    @test !(mk() isa DenseOpType)
    for ρ in (mk(), mk() * 4, 2 * mk(), mk() / 2.0, mk() + mk(), -mk(), (mk() * 4) / 2)
        @test ρ isa Genqo.LazyOpType
        @test !(ρ isa DenseOpType)
    end
    @test Matrix((mk() * 4).data) == ρd.data * 4
    let ρ = mk() * 4
        ρ /= tr(ρ)
        @test ρ isa Genqo.LazyOpType
        @test Matrix(ρ.data) ≈ (ρd.data * 4) / tr(ρd.data * 4)
    end

    # Broadcasting over a single operator goes through QuantumOpticsBase's own operator broadcast,
    # which allocates a dense result. That is QuantumOpticsBase's behaviour, not ours; recorded here
    # so a change in it is noticed. (Broadcasting over an *array* of operators applies scalar `*`
    # elementwise and stays lazy, which is what the tutorials do.)
    @test (mk() .* 4) isa DenseOpType
    @test [mk(), mk()] .* 4 |> first isa Genqo.LazyOpType

    # Laziness survives the lazy wrappers: tr through them still evaluates only the diagonal
    findlazy(A::LazyDensityMatrix) = A
    findlazy(A) = findlazy(first(filter(a -> a isa AbstractArray, collect(A.args))))
    for build in (() -> mk(), () -> mk() * 4, () -> (mk() * 4) / 2, () -> mk() + mk())
        ρ = build()
        inner = findlazy(ρ.data)
        @test ncomputed(inner) == 0
        tr(ρ)
        @test ncomputed(inner) == 4 # the diagonal only, out of 16
    end

    # Functions QuantumOpticsBase restricts to DenseOpType work on the lazy operator directly
    @test entropy_vn(mk()) ≈ entropy_vn(ρd)
    @test entropy_renyi(mk(), 2) ≈ entropy_renyi(ρd, 2)
    @test exp(mk()).data ≈ exp(ρd).data
    # ...and the ones that were already generic still agree
    @test tr(mk()) == tr(ρd)
    @test ptrace(mk(), 1).data ≈ ptrace(ρd, 1).data
    @test expect(mk(), mk()) ≈ expect(ρd, ρd)
    @test variance(mk(), mk()) ≈ variance(ρd, ρd)
end

@testitem "duankimble and emissiveload return lazily-evaluated operators" begin
    using Gabs
    using LinearAlgebra: tr

    engine = HybridProjectionEngine(2)
    st = eprstate(QuadBlockBasis(2), asinh(√1e-1), Float64(π))
    ps = project(st, projector([-1, -1]); η = [0.9, 0.8])

    ρ = duankimble(ps, [1, 0]; engine)
    @test ρ.data isa LazyDensityMatrix
    @test Genqo.ncomputed(ρ.data) == 0

    # Taking the trace evaluates only the diagonal
    t = tr(ρ)
    @test Genqo.ncomputed(ρ.data) == 2
    @test t ≈ sum(Matrix(ρ.data)[i, i] for i in 1:2)
    @test Genqo.ncomputed(ρ.data) == 4

    # Lazy evaluation reproduces the fully-materialized matrix exactly, in any access order
    ρ_diag_first = duankimble(ps, [1, 0]; engine)
    tr(ρ_diag_first)
    ρ_all_at_once = duankimble(ps, [1, 0]; engine)
    @test Matrix(ρ_diag_first.data) == Matrix(ρ_all_at_once.data)

    # Same for emissive loading, whose engine needs two extra modes per memory
    engine4 = HybridProjectionEngine(4)
    ρe = emissiveload(ps; engine = engine4)
    @test ρe.data isa LazyDensityMatrix
    @test Genqo.ncomputed(ρe.data) == 0
    tr(ρe)
    @test Genqo.ncomputed(ρe.data) == 2

    ρe_all = emissiveload(ps; engine = engine4)
    @test Matrix(ρe.data) == Matrix(ρe_all.data)
end


@testitem "dot only evaluates the entries the sandwiching vectors reach" begin
    using Gabs
    using LinearAlgebra: tr, dot

    engine = HybridProjectionEngine(8)
    st = eprstate(QuadBlockBasis(8), asinh(√1e-2), Float64(π))
    apply!(st, [2, 4], modeswap(QuadBlockBasis(2)))
    apply!(st, [5, 7], modeswap(QuadBlockBasis(2)))
    apply!(st, [3, 5], beamsplitter(QuadBlockBasis(2), 0.5))
    apply!(st, [4, 6], beamsplitter(QuadBlockBasis(2), 0.5))
    ps = project(st, projector([-1, -1, 1, 1, 0, 0, -1, -1]); η = fill(0.8, 8))
    mk() = duankimble(ps, [1, 0, 1, 0]; engine)
    D = Matrix(mk().data) # dense reference

    # A basis-vector sandwich picks out exactly one entry, exactly
    for i in 1:4, j in 1:4
        x = zeros(4); x[i] = 1
        y = zeros(ComplexF64, 4); y[j] = 1
        ρ = mk()
        @test dot(x, ρ.data, y) == D[i, j]
        @test ncomputed(ρ.data) == 1
    end

    # A Bell-state fidelity needs the 4 entries where both sides are nonzero, not all 16.
    # Accumulation order differs from the generic method, so agreement is to a few ulp.
    for v in ([1, 0, 0, -1] / √2, [0, 1, 1, 0] / √2)
        for bra in (v', transpose(v), v)
            ρ = mk()
            @test dot(bra, ρ.data, ComplexF64.(v)) ≈ dot(bra, D, ComplexF64.(v)) rtol = 1e-14
            @test ncomputed(ρ.data) == 4
        end
    end

    # Vectors with no zeros still need everything, and complex bras are conjugated as usual
    let ρ = mk(), x = ComplexF64[0.3 + 0.1im, 0.2, 0.5, -0.2im], y = ComplexF64[0.7, -0.1im, 0.4, 0.4]
        @test dot(x, ρ.data, y) ≈ dot(x, D, y) rtol = 1e-14
        @test ncomputed(ρ.data) == 16
    end

    # Dispatch is on LazyDensityMatrix rather than on any foreign type, so a sandwich against data
    # that has been through lazy arithmetic falls back to the generic dense path: still correct,
    # but it reads every entry. Sandwich the unscaled operator to stay sparse.
    let ρ = mk(), v = [0, 1, 1, 0] / √2
        @test dot(v', (ρ * 4).data, ComplexF64.(v)) ≈ dot(v', 4 * D, ComplexF64.(v)) rtol = 1e-14
        @test ncomputed(ρ.data) == 16
    end
    let ρ = mk(), v = [0, 1, 1, 0] / √2
        @test 4 * dot(v', ρ.data, ComplexF64.(v)) ≈ dot(v', 4 * D, ComplexF64.(v)) rtol = 1e-14
        @test ncomputed(ρ.data) == 4
    end

    @test_throws DimensionMismatch dot(ones(3)', mk().data, ones(ComplexF64, 4))
    @test_throws DimensionMismatch dot(ones(4)', mk().data, ones(ComplexF64, 3))
end
