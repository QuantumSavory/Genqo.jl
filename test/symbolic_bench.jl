using Genqo
using Gabs: Gabs, eprstate, QuadBlockBasis, ⊗, apply!, beamsplitter, embed
using SparseArrays

using BenchmarkTools

μs = logrange(1e-4, 1e2, 100);

## N=8

# Option 1: apply matrices directly with no fusion (embedding, current Gabs.jl capability)
print("No fusion (N=8): ")
function gaussian_ops_nofusion_n8(μ)
    tmsv = eprstate(QuadBlockBasis(2), asinh(√μ), 0.)
    basis = QuadBlockBasis(2)
    full_basis = QuadBlockBasis(8)
    st = reduce(⊗, tmsv for _ in 1:4)
    ms1 = embed(full_basis, [2,4], modeswap(basis))
    ms2 = embed(full_basis, [5,7], modeswap(basis))
    bs = beamsplitter(basis, 0.5)
    bs1 = embed(full_basis, [3,5], bs)
    bs2 = embed(full_basis, [4,6], bs)

    apply!(st, ms1)
    apply!(st, ms2)
    apply!(st, bs1)
    apply!(st, bs2)

    st
end
@btime gaussian_ops_nofusion_n8.(μs)

# Option 2: apply matrices directly with no fusion (direct application, no embedding)
print("No fusion, no embed (N=8): ")
function gaussian_ops_nofusion_noembed_n8(μ)
    basis = QuadBlockBasis(2)
    tmsv = eprstate(basis, asinh(√μ), 0.)
    st = reduce(⊗, tmsv for _ in 1:4)
    bs = beamsplitter(basis, 0.5)
    ms = modeswap(basis)

    apply!(st, ms, [2,4])
    apply!(st, ms, [5,7])
    apply!(st, bs, [3,5])
    apply!(st, bs, [4,6])

    st
end
@btime gaussian_ops_nofusion_noembed_n8.(μs)

# Option 3: use Genqo.jl's fused circuit representation and engine
print("With fusion (N=8): ")
q = QCircuit(8)
basis = QuadBlockBasis(2)
modeswap(basis) | q[2,4]
modeswap(basis) | q[5,7]
beamsplitter(basis, 0.5) | q[3,5]
beamsplitter(basis, 0.5) | q[4,6]
q_fused = fuse(q)
engine = HybridGaussianCoherentEngine(8)
function gaussian_ops_fused_n8(μ)
    tmsv = eprstate(QuadBlockBasis(2), asinh(√μ), 0.)
    engine.gaussian_state = reduce(⊗, tmsv for _ in 1:4)
    for gate in q_fused.gates
        apply!(engine.gaussian_state, gate[1].gate, gate[2])
    end
    engine.gaussian_state
end
@btime gaussian_ops_fused_n8.(μs)

print("Fusion overhead (N=8): ")
@btime fuse(q)

cov_nofusion_n8 = gaussian_ops_nofusion_n8.(μs)
cov_nofusion_noembed_n8 = gaussian_ops_nofusion_noembed_n8.(μs)
cov_fused_n8 = gaussian_ops_fused_n8.(μs)
@assert all(cov_nofusion_n8 .≈ cov_fused_n8) && all(cov_nofusion_noembed_n8 .≈ cov_fused_n8)


## N=16

# Option 1: apply matrices directly with no fusion (embedding, current Gabs.jl capability)
print("No fusion (N=16): ")
function gaussian_ops_nofusion_n16(μ)
    tmsv = eprstate(QuadBlockBasis(2), asinh(√μ), 0.)
    basis = QuadBlockBasis(2)
    full_basis = QuadBlockBasis(16)
    st = reduce(⊗, tmsv for _ in 1:8)
    ms = modeswap(basis)
    ms1 = embed(full_basis, [2,4], ms)
    ms2 = embed(full_basis, [5,7], ms)
    ms3 = embed(full_basis, [12,14], ms)
    ms4 = embed(full_basis, [13,15], ms)
    bs = beamsplitter(basis, 0.5)
    bs1 = embed(full_basis, [3,5], bs)
    bs2 = embed(full_basis, [4,6], bs)
    bs3 = embed(full_basis, [11,13], bs)
    bs4 = embed(full_basis, [12,14], bs)

    apply!(st, ms1)
    apply!(st, ms2)
    apply!(st, ms3)
    apply!(st, ms4)
    apply!(st, bs1)
    apply!(st, bs2)
    apply!(st, bs3)
    apply!(st, bs4)

    st
end
@btime gaussian_ops_nofusion_n16.(μs)

# Option 2a: apply matrices directly with no fusion (direct application, no embedding)
print("No fusion, no embed (N=16): ")
function gaussian_ops_nofusion_noembed_n16(μ)
    tmsv = eprstate(QuadBlockBasis(2), asinh(√μ), 0.)
    st = reduce(⊗, tmsv for _ in 1:8)
    basis = QuadBlockBasis(2)
    bs = beamsplitter(basis, 0.5)
    ms = modeswap(basis)

    apply!(st, ms, [2,4])
    apply!(st, ms, [5,7])
    apply!(st, ms, [12,14])
    apply!(st, ms, [13,15])
    apply!(st, bs, [3,5])
    apply!(st, bs, [4,6])
    apply!(st, bs, [11,13])
    apply!(st, bs, [12,14])

    st
end
@btime gaussian_ops_nofusion_noembed_n16.(μs)

# Option 2b: apply matrices directly with no fusion (direct application, no embedding, sparse matrix)
print("No fusion, no embed, sparse (N=16): ")
function gaussian_ops_nofusion_noembed_sparse_n16(μ)
    basis = QuadBlockBasis(2)
    tmsv = eprstate(Vector, SparseMatrixCSC, basis, asinh(√μ), 0.)
    st = reduce(⊗, tmsv for _ in 1:8)
    bs = beamsplitter(Vector, SparseMatrixCSC, basis, 0.5)
    ms = modeswap(basis)

    apply!(st, ms, [2,4])
    apply!(st, ms, [5,7])
    apply!(st, ms, [12,14])
    apply!(st, ms, [13,15])
    apply!(st, bs, [3,5])
    apply!(st, bs, [4,6])
    apply!(st, bs, [11,13])
    apply!(st, bs, [12,14])

    st
end
@btime gaussian_ops_nofusion_noembed_sparse_n16.(μs);

# Option 3: use Genqo.jl's fused circuit representation and engine
print("With fusion (N=16): ")
q = QCircuit(16)
basis = QuadBlockBasis(2)
modeswap(basis) | q[2,4]
modeswap(basis) | q[5,7]
modeswap(basis) | q[12,14]
modeswap(basis) | q[13,15]
beamsplitter(basis, 0.5) | q[3,5]
beamsplitter(basis, 0.5) | q[4,6]
beamsplitter(basis, 0.5) | q[11,13]
beamsplitter(basis, 0.5) | q[12,14]
q_fused = fuse(q)
engine = HybridGaussianCoherentEngine(16)
function gaussian_ops_fused_n16(μ)
    tmsv = eprstate(QuadBlockBasis(2), asinh(√μ), 0.)
    engine.gaussian_state = reduce(⊗, tmsv for _ in 1:8)
    for gate in q_fused.gates
        apply!(engine.gaussian_state, gate[1].gate, gate[2])
    end
    engine.gaussian_state
end
@btime gaussian_ops_fused_n16.(μs)

print("Fusion overhead (N=16): ")
@btime fuse(q)

cov_nofusion_n16 = gaussian_ops_nofusion_n16.(μs)
cov_nofusion_noembed_n16 = gaussian_ops_nofusion_noembed_n16.(μs)
cov_fused_n16 = gaussian_ops_fused_n16.(μs)
@assert all(cov_nofusion_n16 .≈ cov_fused_n16) && all(cov_nofusion_noembed_n16 .≈ cov_fused_n16)
