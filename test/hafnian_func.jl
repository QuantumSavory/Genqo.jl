# using LinearAlgebra
# using Genqo
# using BenchmarkTools
# using Memoization

# # Get optional function filter and output directory from command line arguments
# func_filter = length(ARGS) > 0 ? ARGS[1] : ""
# bench_dir   = length(ARGS) > 1 ? ARGS[2] : ".benchmarks"

# uniform(min_val, max_val) = min_val + (max_val-min_val)*rand(Float64)
# log_uniform(min_exp, max_exp) = 10^uniform(min_exp, max_exp)

# suite = BenchmarkGroup()

# B = randn(ComplexF64, 64, 64)

# A = ComplexF64[ 0 2 0 0;
#                 2 0 0 0;
#                 0 0 0 5;
#                 0 0 5 0;]

#  A = ComplexF64[ 1 0;
#                 0 1;]

# function sparcity_ratio(μ::Real, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real, dark_counts::Real)
#     cov = Genqo.zalm.covariance_matrix(μ)
#     A = Genqo.tools.k_function_matrix(cov) + Genqo.zalm.loss_bsm_matrix_pgen(ηᵗ, ηᵈ, ηᵇ)
#     Ainv = inv(A)
#    return count(!iszero, Ainv) / length(Ainv)
# end

# function recursive_algorithm(A::Matrix{ComplexF64})
#     nb_lines = size(A,1)

#     if nb_lines % 2 != 0
#         return 0
#     end

#     n = nb_lines ÷ 2

#     z = zeros(ComplexF64, n * (2n - 1), n + 1)

#     for j in 1:(2n - 1)
#         ind = j * (j - 1) ÷ 2
#         for k in 0:(j - 1)
#             z[ind + k + 1, 1] = A[j + 1, k + 1]
#         end
#     end

#     g = zeros(ComplexF64, n + 1)
#     g[1] = one(ComplexF64)

#     return solve_recursive(z, 2n, 1, g, n)
# end


# function solve_recursive(
#     b::Matrix{ComplexF64},
#     s::Int,
#     w::Int,
#     g::AbstractVector{ComplexF64},
#     n::Int,
# )

#     if s == 0
#         return w * g[n + 1]
#     end

#     c = zeros(ComplexF64, ((s - 2) * (s - 3)) ÷ 2, n + 1)

#     i = 1

#     for j in 1:(s - 3)
#         for k in 0:(j - 1)
#             src_row = ((j + 1) * (j + 2)) ÷ 2 + k + 3
#             c[i, :] .= b[src_row, :]
#             i += 1
#         end
#     end

#     h = solve_recursive(c, s - 2, -w, g, n)

#     e = copy(g)

#     for u in 0:(n - 1)
#         for v in 0:(n - u - 1)
#             e[u + v + 2] += g[u + 1] * b[1, v + 1]
#         end
#     end

#     for j in 1:(s - 3)
#         for k in 0:(j - 1)
#             c_row = j * (j - 1) ÷ 2 + k + 1

#             for u in 0:(n - 1)
#                 for v in 0:(n - u - 1)
#                     c[c_row, u + v + 2] +=
#                         b[((j + 1) * (j + 2)) ÷ 2 + 1, u + 1] *
#                         b[((k + 1) * (k + 2)) ÷ 2 + 2, v + 1] +
#                         b[((k + 1) * (k + 2)) ÷ 2 + 1, u + 1] *
#                         b[((j + 1) * (j + 2)) ÷ 2 + 2, v + 1]
#                 end
#             end
#         end
#     end

#     return h + solve_recursive(c, s - 2, w, e, n)
# end
# function hafnian_sparse(A::AbstractMatrix{ComplexF64}, D::Set{Int}=nothing; loop::Bool=false)
#     n = size(A, 1)

#     Dset = D === nothing ? Set(1:n) : Set(D)

#     B = copy(A)
#     if !loop
#         for i in 1:n
#             B[i, i] = zero(ComplexF64)
#         end
#     end

#     # Python: if np.allclose(A, 0): return 0.0
#     if all(x -> isapprox(x, zero(ComplexF64); atol=1e-12, rtol=1e-12), B)
#         return zero(ComplexF64)
#     end

#     # Memoization cache.
#     # Use sorted tuples as keys because Set is mutable and unsafe as a Dict key.
#     cache = Dict{Tuple{Vararg{Int}}, ComplexF64}()

#     function lhaf(d::Set{Int})::ComplexF64
#         key = Tuple(sort(collect(d)))

#         if haskey(cache, key)
#             return cache[key]
#         end

#         if isempty(d)
#             return one(T)
#         end

#         d_without_k = copy(d)
#         k = pop!(d_without_k)

#         # Python: indices(d, k) = d ∩ nonzero(A[k, :])
#         nonzero_cols = findall(j -> !iszero(B[k, j]), 1:n)
#         js = intersect(d, Set(nonzero_cols))

#         result = zero(T)
#         for j in js
#             next_d = setdiff(d_without_k, Set([j]))
#             result += B[j, k] * lhaf(next_d)
#         end

#         cache[key] = result
#         return result
#     end

#     return lhaf(Dset)
# end

# function hafnian_batched(A::Matrix{ComplexF64}, cutoff::Int, mu::vector{}, rtol::Float64 = 1e-05, atol::Float64 = 1e-08, renom::Bool = false, make_tensor::Bool = true)
#    #input validation?

#    n = size(A, 1)

#    if isEmpty(mu)
#        mu = zeros(ComplexF64, n)
#    end

#     return hermite_multidimnetional()
# end

# function hermite_multidimnetional(R::Matrix{ComplexF64}, cutoff::Int, y::vector{}, C::ComplexF64=1, renorm::Bool=False, make_tensor::Bool=True, modified::Bool=False, rtol::Float64=1e-05, atol::Float64=1e-08)
#     #input validation??
#     n = size(R, 1)

#     if !modified && !isempty(y)
#         m = size(y, 1)
#         if m == nameo
#             ym = R * y
#             return hermite_multidimnetional()
#         end
#     end

#     if y === nothing
#         y = zeros(ComplexF64, n)
#     end

#     m = size(y, 1)

#     if m != n
#         ##throw error
#     end

#     num_indices = length(y)

#     cutoff =
#         if type(cutoff) == AbstractArray && length(cutoff) == 1
#             ntuple(_ -> Int(cutoff[begin]), num_indices)
#         elseif cutoff isa Integer\
#             ntuple(_ -> Int(cutoff, num_indices))
#         else
#             Tuple(Int.(cutoff))
#         end

#     Rt = real_if_close(R)
#     yt = real_if_close(y)

#     T = promote_type(Rt, yt, C)
#     array = zeros(T, cutoff)

#     array[ntuple(_ -> 1, num_indices)...] = C

#     values = 
#         if renorm 
#             Array(_hermite_multidimensional_renorm(Rt, yt, array))
#         else 
#             Array(_hermite_multidimensional(Rt, yt, array))
#         end

#     if !make_tensor
#         values = vec(values)
#     end
#     return values

# end

# function _hermite_multidimensional_renorm(R::Matrix{T}, y::Vector{T}, G::Array{T, N}) where {T, N}
#     shape = size(G)
#     shape_arr = collect(shape)
#     D = length(y)
#     # calculate the strides (e.g. (100,10,1) for shape (10,10,10))
    
#     strides = ones(Int, D)

#     for i in (D-1):-1:1
#         strides[i] = strides[i + 1] * shape_arr[i + 1]
#     end

#     total_size = prod(shape)
#     Gflat = zeros(eltype(G), total_size)

#     # Copy G into the row-major flat buffer
#     for flat_index in 0:(total_size - 1)
#         idx0 = rowmajor_index(flat_index, shape_arr, strides)
#         idx1 = Tuple(i + 1 for i in idx0)
#         Gflat[flat_index + 1] = G[idx1...]
#     end

#      # Iterate over the indices smaller than max(strides) with pivot bound check.
#     # The check is needed only if the flat index is smaller than the largest stride.
#     # Afterwards it will be safe to get the pivot by subtracting the first (largest) stride.
#     for flat_index in 1:(strides[1] - 1)
#         index = rowmajor_index(flat_index, shape_arr, strides)
#         i = 1

#         for s in strides
#             pivot = flat_index - s
#             if pivot >= 0 
#                 break
#             end
#             i += 1
#         end
#         value_at_index = y[i] * Gflat[pivot + 1]

#         # Contribution from pivot's lower neighbours
#         value_at_index -= R[i, i] * sqrt(index[i] - 1) * Gflat[pivot - strides[i] + 1]
#         for j in i+1:D 
#             value_at_index -= R[i,j] * sqrt(index[j]) * Gflat[pivot - strides[j] + 1]
#         end

#         Gflat[flat_index + 1] = value_at_index/sqrt(index[i])

#     end

#     # Iterate over the rest of the indices.
#     for flat_index in strides[1]:(total_size - 1)
#         index = rowmajor_index(flat_index, shape_arr, strides)

#         pivot = flat_index - strides[1]

#         value_at_index = y[1] * Gflat[pivot + 1]
#         #Contribution from pivot's lower neighbours
#         value_at_index -= R[1, 1] *sqrt(index[1] - 1) * Gflat[pivot - strides[1] + 1]
#         for j in 2:D
#             value_at_index -= R[1, j] * sqrt(index[j]) * Gflat[pivot - strides[j] + 1]
#         end

#         Gflat[flat_index + 1] = value_at_index / sqrt(index[1])
#     end
#     out = similar(G)

#     for flat_index in 0:(total_size - 1)
#         idx0 = rowmajor_index(flat_index, shape_arr, strides)
#         idx1 = Tuple(i + 1 for i in idx0)
#         out[idx1...] = Gflat[flat_index + 1]
#     end

#     return out
# end

# function rowmajor_index(flat_index::Int, shape_arr::Vector{Int}, strides::Vector{Int})
#     D = length(shape_arr)
#     index = Vector{Int}(undef, D)

#     for k in 1:D
#         index[k] = div(flat_index, strides[k]) % shape_arr[k]
#     end

#     return index
# end

# function _hermite_multidimensional(R::Matrix{T}, y::Vector{T}, G::Array{T, N}) where {T, N}
#     shape = size(G) 
#     D = length(shape)
#     indices = CartesianIndices(D)

#     for idx in Iterators.drop(indices, 1)
#         idx_tuple = Tuple(idx)

#         i = 1
#         for j in 1:D
#             if idx_tuple[j] > 1
#                 i = j
#                 break
#             end
#         end
#         ki = ntuple(j -> j == i ? idx_tuple[j] - 1 : idx_tuple[j], D)

#         u = y[i] * G[ki...]
#          for l in 1:D
#             if ki[l] > 1
#                 kl = ntuple(j -> j == l ? ki[j] - 1 : ki[j], D)
#                 u -= (ki[l] - 1) * R[i, l] * G[kl...]
#             end
#         end

#         G[idx_tuple...] = u
#     end

#     return G
# end

# #for sparce matrix

# function hafnian_trace()

# suite["hafnian_recursive for 64x64"] = @benchmarkable recursive_algorithm(B) setup=(B=randn(ComplexF64, 64, 64))
# suite["hafnian_batched for 64x64"] = @benchmarkable hafnian_batched(B) setup=(B=randn(ComplexF64, 64, 64))

# results = run(suite)
# for (func, trial) in results
#     println("$func:")
#     display(trial)
#     println()
# end
# BenchmarkTools.save(joinpath(bench_dir, "jl-bench.json"), results)
