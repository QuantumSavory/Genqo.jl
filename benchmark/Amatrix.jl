# Benchmark comparing `_invA_UL`'s two internal code paths: the O(N^3)-ish
# Schur-complement shortcut that pivots explicitly on C̃, versus its full 4N×4N
# `A_P` inversion fallback (triggered when C̃ is singular -- structurally the case
# for sources like SIGSAG, which mix genuine vacuum ancillas through a passive
# network; see the "why is C̃ singular for sigsag" discussion this benchmark grew
# out of). Quantifies what such sources give up by needing the fallback.
#
# Runnable directly:
#     julia --project=benchmark benchmark/Amatrix.jl

using Genqo
using LinearAlgebra
using BenchmarkTools
using Random

Random.seed!(1)

# A generic random pure N-mode covariance (QuadBlockBasis/qqpp, ħ=1 convention):
# a random symplectic (squeezing + passive mixing) applied to vacuum. Generically
# gives a well-conditioned C̃, exercising _invA_UL's fast Schur-complement path.
function random_pure_cov(N::Int)
    Ω = zeros(2N, 2N)
    for i in 1:N
        Ω[i, N+i] = 1
        Ω[N+i, i] = -1
    end
    H = randn(2N, 2N)
    H = (H + H') / 2
    S = exp(Ω * H)
    S * Matrix{Float64}(0.5I, 2N, 2N) * S'
end

# A pure N-mode covariance where only 2 modes are squeezed and the rest are exact,
# unmixed vacuum -- C̃ then has N-2 exact zero rows/cols, unconditionally singular
# regardless of squeezing strength. This is the structural mechanism behind
# SIGSAG's singular C̃, minus the beamsplitter rotation that hides which modes it
# is (irrelevant here since we only need *a* singular C̃, not a realistic circuit).
function singular_C_cov(N::Int)
    N >= 2 || throw(ArgumentError("Need at least 2 modes"))
    core = random_pure_cov(2)
    σ = Matrix{Float64}(0.5I, 2N, 2N)
    idx = [1, 2, N + 1, N + 2]
    σ[idx, idx] = core
    σ
end

function C̃_of(σ::Matrix{Float64})
    N = size(σ, 1) ÷ 2
    Γinv = inv(σ + 0.5I)
    a = Γinv[1:N, 1:N]
    b = Γinv[N+1:2N, N+1:2N]
    c = Γinv[1:N, N+1:2N]
    cᵀ = Γinv[N+1:2N, 1:N]
    0.5 * (a - b + im * (c + cᵀ))
end

println(
    rpad("N", 6), rpad("shortcut t", 14), rpad("full t", 14), rpad("t ratio", 10),
    rpad("shortcut mem", 16), rpad("full mem", 16), "mem ratio",
)
for N in (8, 12, 16)
    η = fill(0.8, N)
    clickmat = reshape(fill(0, N), 1, :)
    n = @view clickmat[1, :]

    σ_fast = random_pure_cov(N)
    σ_slow = singular_C_cov(N)

    # Confirm each input actually exercises the path we intend to benchmark.
    @assert cond(C̃_of(σ_fast)) < 1e8 "expected a well-conditioned C̃ for the shortcut-path input"
    @assert !isfinite(cond(C̃_of(σ_slow))) || cond(C̃_of(σ_slow)) > 1e12 "expected a singular C̃ for the fallback-path input"

    b_fast = @benchmark Genqo._invA_UL($σ_fast, $η, $n) seconds = 1
    b_slow = @benchmark Genqo._invA_UL($σ_slow, $η, $n) seconds = 1

    t_fast, t_slow = median(b_fast).time, median(b_slow).time
    m_fast, m_slow = median(b_fast).memory, median(b_slow).memory
    println(
        rpad(string(N), 6),
        rpad(BenchmarkTools.prettytime(t_fast), 14),
        rpad(BenchmarkTools.prettytime(t_slow), 14),
        rpad("$(round(t_slow / t_fast, digits=1))x", 10),
        rpad(BenchmarkTools.prettymemory(m_fast), 16),
        rpad(BenchmarkTools.prettymemory(m_slow), 16),
        "$(round(m_slow / m_fast, digits=1))x",
    )
end
