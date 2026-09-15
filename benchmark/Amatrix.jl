# Benchmark comparing three ways to get the Gaussian contraction kernel A⁻¹ (and
# the accompanying normalization) out of a covariance matrix:
#
#   direct    -- assemble the full 4N×4N A and LU-invert it, O((4N)³)
#   _invA     -- closed-form block inversion, full 4N×4N result
#   _invA_UL  -- same closed form, but only the 2N×2N upper-left block, which is
#                all the Wick contraction in `project` actually consumes
#
# The block formula pivots on G = (I - C̃YC̃*Y)⁻¹ rather than on C̃ itself, so it no
# longer needs C̃ to be invertible. That used to be a hard restriction: sources
# that mix genuine vacuum ancillas through a passive network (SIGSAG, anything
# with unsqueezed modes) have a structurally singular C̃ and had to fall back to
# the direct 4N×4N inversion. Both covariance classes below therefore run through
# the same code path now, and the table shows there is no longer any penalty for
# the singular one.
#
# Runnable directly:
#     julia --project=benchmark benchmark/Amatrix.jl

using Genqo
using LinearAlgebra
using BenchmarkTools
using Random

Random.seed!(1)

# A generic random pure N-mode covariance (QuadBlockBasis/qqpp, ħ=1 convention):
# a random symplectic (squeezing + passive mixing) applied to vacuum. Mixes q and
# p, so C̃ comes out complex and well conditioned.
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
# unmixed vacuum -- C̃ then has N-2 exact zero rows/cols and is unconditionally
# singular regardless of squeezing strength. This is the structural mechanism
# behind SIGSAG's singular C̃, minus the beamsplitter rotation that hides which
# modes it is (irrelevant here since we only need *a* singular C̃).
function vacuum_padded_cov(N::Int)
    N >= 2 || throw(ArgumentError("Need at least 2 modes"))
    σ = Matrix{Float64}(0.5I, 2N, 2N)
    idx = [1, 2, N + 1, N + 2]
    σ[idx, idx] = random_pure_cov(2)
    σ
end

# The pieces the closed form is built out of: the C̃ kernel, the detector-loss
# diagonal Y, and Γ (whose determinant enters the normalization).
function kernel_pieces(σ::Matrix{Float64}, η::Vector{Float64}, n::AbstractArray{Int})
    mds = size(σ, 1) ÷ 2
    Γ = σ + 0.5I
    Γinv = inv(Γ)
    a  = @view Γinv[1:mds,      1:mds     ]
    c  = @view Γinv[1:mds,      mds+1:2mds]
    cᵀ = @view Γinv[mds+1:2mds, 1:mds     ]
    b  = @view Γinv[mds+1:2mds, mds+1:2mds]

    C̃ = 0.5 * (a - b + im * (c + cᵀ))
    y = ones(ComplexF64, mds)
    y[n .!== -1] -= η[n .!== -1] # y_i = 1 - η_i for detected modes, y_i = 1 for traced-out modes
    (C̃, Diagonal(y), Γ)
end

# The full 4N×4N kernel:
# A = [
#   0  -Y   I   0
#  -Y   0   0   I
#   I   0   C̃   0
#   0   I   0   C̃*
# ]
function A_full(C̃::Matrix{ComplexF64}, Y::Diagonal)
    mds = size(C̃, 1)
    A = zeros(ComplexF64, 4mds, 4mds)
    copyto!(view(A, 1:mds,       mds+1:2mds ), -Y)
    copyto!(view(A, mds+1:2mds,  1:mds      ), -Y)
    copyto!(view(A, 1:mds,       2mds+1:3mds), I)
    copyto!(view(A, mds+1:2mds,  3mds+1:4mds), I)
    copyto!(view(A, 2mds+1:3mds, 1:mds      ), I)
    copyto!(view(A, 3mds+1:4mds, mds+1:2mds ), I)
    copyto!(view(A, 2mds+1:3mds, 2mds+1:3mds), C̃)
    copyto!(view(A, 3mds+1:4mds, 3mds+1:4mds), conj(C̃))
    A
end

# The baseline: same signature and return value as `Genqo._invA`, but obtained by
# factorizing A whole. One LU serves both the inverse and det A (which equals
# det(I - C̃YC̃*Y), the quantity the block form gets for free).
function invA_direct(σ::Matrix{Float64}, η::Vector{Float64}, n::AbstractArray{Int})
    C̃, Y, Γ = kernel_pieces(σ, η, n)
    F = lu(A_full(C̃, Y))
    denom = sqrt(real(det(F)) * abs(det(Γ)))
    (inv(F), denom)
end

# Non-uniform η and a mixed detect/traceout pattern, so Y is not a multiple of I
# and the conjugate-symmetric block structure is actually exercised.
detector_config(N::Int) = (collect(range(0.6, 0.95, N)), [isodd(i) ? 0 : -1 for i in 1:N])

# One set of inputs, shared by the verification and the timings below, so the
# numbers in every table describe the same matrices.
const CASES = [(N, label, cov(N), detector_config(N)...)
               for N in (8, 16, 32)
               for (label, cov) in (("generic", random_pure_cov),
                                    ("singular", vacuum_padded_cov))]


# Correctness: every method must reproduce the true inverse of A, for a singular C̃
# just as for a generic one.

println("Verification -- ‖A·A⁻¹ - I‖ and agreement between methods\n")
println(rpad("N", 5), rpad("C̃", 11), rpad("cond(C̃)", 12), rpad("direct", 12),
        rpad("_invA", 12), rpad("_invA_UL Δ", 12), "denom Δ (rel)")
for (N, label, σ, η, n) in CASES
    C̃, Y, _ = kernel_pieces(σ, η, n)
    A = A_full(C̃, Y)

    invA_d, denom_d = invA_direct(σ, η, n)
    invA, denom = Genqo._invA(σ, η, n)
    invA_UL, denom_UL = Genqo._invA_UL(σ, η, n)

    res_d = norm(A * invA_d - I)
    res = norm(A * invA - I)
    ul_Δ = norm(invA[1:2N, 1:2N] - invA_UL)
    # Relative: denom carries a det Γ factor, so its absolute scale varies wildly.
    denom_Δ = max(abs(denom - denom_d), abs(denom - denom_UL)) / abs(denom)

    @assert res < 1e-8 "_invA did not invert A ($label C̃, N=$N)"
    @assert ul_Δ < 1e-12 "_invA_UL disagrees with the upper-left block of _invA"
    @assert denom_Δ < 1e-8 "normalizations disagree ($label C̃, N=$N)"

    println(rpad(N, 5), rpad(label, 11), rpad(round(cond(C̃), sigdigits = 3), 12),
            rpad(round(res_d, sigdigits = 3), 12), rpad(round(res, sigdigits = 3), 12),
            rpad(round(ul_Δ, sigdigits = 3), 12), round(denom_Δ, sigdigits = 3))
end

# Timing. `_invA_UL` returns a quarter of the matrix, so its speedup is not an
# apples-to-apples inversion comparison -- it is the cost of what `project`
# actually needs.

results = map(CASES) do (N, label, σ, η, n)
    (N, label,
     mean(@benchmark invA_direct($σ, $η, $n) seconds = 1),
     mean(@benchmark Genqo._invA($σ, $η, $n) seconds = 1),
     mean(@benchmark Genqo._invA_UL($σ, $η, $n) seconds = 1))
end

function report(title, field, pretty, ratio_header)
    println("\n\n", title, "\n")
    println(rpad("N", 5), rpad("C̃", 11), rpad("direct", 13), rpad("_invA", 13),
            rpad("_invA_UL", 13), rpad("_invA", 9), "_invA_UL")
    for (N, label, t_d, t_f, t_u) in results
        d, f, u = getfield(t_d, field), getfield(t_f, field), getfield(t_u, field)
        println(rpad(N, 5), rpad(label, 11), rpad(pretty(d), 13), rpad(pretty(f), 13),
                rpad(pretty(u), 13), rpad("$(round(d / f, digits = 1))x", 9),
                "$(round(d / u, digits = 1))x")
    end
    println("(last two columns: ", ratio_header, ")")
end

report("Timing -- mean of @benchmark", :time, BenchmarkTools.prettytime,
       "speedup over direct inversion")
report("Allocation -- mean of @benchmark", :memory, BenchmarkTools.prettymemory,
       "allocation reduction vs direct inversion")
