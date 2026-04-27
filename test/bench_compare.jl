using Genqo
using BenchmarkTools
using Nemo
using LinearAlgebra
using BlockDiagonals


suite = BenchmarkGroup()

uniform(min_val, max_val) = min_val + (max_val-min_val)*rand(Float64)
log_uniform(min_exp, max_exp) = 10^uniform(min_exp, max_exp)

rand_sigsag() = sigsag.SIGSAG(
    log_uniform(-5, 1),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
)

# Global canonical position and momentum variables
const mds = 6 # Number of modes for our system

_qai = ["qa$i" for i in 1:mds]
_pai = ["pa$i" for i in 1:mds]
_qbi = ["qb$i" for i in 1:mds]
_pbi = ["pb$i" for i in 1:mds]
all_qps = hcat(_qai, _pai, _qbi, _pbi)
CC = ComplexField()
i = onei(CC) # Imaginary unit in CC ring
R, generators = polynomial_ring(CC, all_qps)
(qai, pai, qbi, pbi) = (generators[:,i] for i in 1:4)

# Define the alpha and beta vectors
α = (qai + i .* pai) / sqrt(2)
β = (qbi - i .* pbi) / sqrt(2)

function probability_success_new(μ::Real, ηᵗ::Real, ηᵈ::Real)
    cov = sigsag.covariance_matrix(μ)
    A = tools.k_function_matrix(cov) + sigsag.loss_bsm_matrix_pgen(ηᵗ, ηᵈ)
    lu!(A)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = 1/(D1*D2*D3)

    C = ηᵈ^2 * (α[1]*α[2]) * (β[1]*β[2]) # moment_vector([0,0,0,0], [0,0,0,0], ηᵗ, ηᵈ)

    return real(Coef * tools.W(C, Ainv))
end
probability_success_new(sigsag::sigsag.SIGSAG) = probability_success_new(sigsag.mean_photon, sigsag.outcoupling_efficiency, sigsag.detection_efficiency)


# suite["sigsag.probability_success_old"] = @benchmarkable sigsag.probability_success(s) setup=(s=rand_sigsag())
# suite["sigsag.probability_success_new"] = @benchmarkable probability_success_new(s) setup=(s=rand_sigsag())

#probability of sucess zalm
 
rand_zalm() = zalm.ZALM(
    log_uniform(-5, 1),
    #[one(Float64)],
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    uniform(0.5, 1.0),
    zero(Float64),
    #one(Float64)
)
nvec = [1,0,1,1,0,0,1,0]

#Not sucessful
function zalm_probability_success_new(μ::Real, ηᵗ::Real, ηᵈ::Real, ηᵇ::Real, dark_counts::Real)
    cov = zalm.covariance_matrix(μ)
    A = zalm.k_function_matrix(cov) + zalm.loss_bsm_matrix_pgen(ηᵗ, ηᵈ, ηᵇ)
    lu!(A)
    Ainv = inv(A)
    Γ = cov + (1/2)*I
    detΓ = det(Γ)

    D1 = sqrt(det(A))
    D2 = detΓ^(1/4)
    D3 = conj(detΓ)^(1/4)
    Coef = 1/(D1*D2*D3)

    # TODO: should this indexing be changed to 1-based? Or is there some mathematical meaning to the 0 index?
    C1 = zalm.moment_terms[0]
    C2 = zalm.moment_terms[9]
    C3 = zalm.moment_terms[10]
    C4 = zalm.moment_terms[14]

    return real(Coef * (
        ηᵇ^2 * (1-dark_counts)^4 * zalm.W(C1, Ainv) +
        ηᵇ * dark_counts * (1-dark_counts)^3 * zalm.W(C2, Ainv) +
        ηᵇ * dark_counts * (1-dark_counts)^3 * zalm.W(C3, Ainv) +
        dark_counts^2 * (1-dark_counts)^2 * zalm.W(C4, Ainv)
    ))
end
 zalm_probability_success_new(zalm::zalm.ZALM) =  zalm_probability_success_new(zalm.mean_photon, zalm.outcoupling_efficiency, zalm.detection_efficiency, zalm.bsm_efficiency, zalm.dark_counts)


# suite["zalm.probability_success_old"]    = @benchmarkable zalm.probability_success(z)         setup=(z=rand_zalm())
# suite["zalm.probability_success_new"]    = @benchmarkable  zalm_probability_success_new(z)         setup=(z=rand_zalm())

#Not consistent
function zalm_covariance_matrix_old(μ::Real)
   # Initial ZALM covariance matrix in qpqp ordering
    spdc_covar = spdc.covariance_matrix(μ)
    covar_qpqp = Matrix(BlockDiagonal([spdc_covar, spdc_covar]))

    # Reorder qpqp → qqpp and apply beamsplitters
    covar_qqpp = tools.reorder(covar_qpqp) 

    return zalm._S46 * zalm._S35 * covar_qqpp * zalm._S35' * zalm._S46' # or similar pattern
end
zalm_covariance_matrix_old(zalm::zalm.ZALM) = zalm_covariance_matrix_old(zalm.mean_photon)

suite["zalm.covariance_matrix_new"]      = @benchmarkable zalm.covariance_matrix(z)           setup=(z=rand_zalm())
suite["zalm.covariance_matrix_old"]      = @benchmarkable zalm_covariance_matrix_old(z)           setup=(z=rand_zalm())

results = run(suite)
for (func, trial) in results
    println("$func:")
    display(trial)
    println()
end
 