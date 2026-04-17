module states

using LinearAlgebra

export QuantumState, GaussianState, FockState, TMSVState, CoherentState, ThermalState, VacuumState, wigner
export wigner
export TMSVState


abstract type QuantumState end

# TODO: make numerical types user-specifiable
mutable struct GaussianState <: QuantumState
    means::Vector{Float64}
    covariance::Matrix{Float64}
end

function wigner(state::GaussianState, x::Vector{Float64})::Float64
    n = length(x) ÷ 2
    V = state.covariance
    return 1/(π^n*sqrt(det(V))) * exp(-(x'*inv(V)*x) / 2)
end

mutable struct FockState <: QuantumState
    density::Array{ComplexF64}
    cutoffs::Vector{Int}
end

# All states in qqpp ordering

TMSVState(μ::Real) = GaussianState(
    zeros(4),
    [
        1+2μ         2√(μ*(μ+1))  0             0            ;
        2√(μ*(μ+1))  1+2μ         0             0            ;
        0            0            1+2μ          -2√(μ*(μ+1)) ;
        0            0            -2√(μ*(μ+1))  1+2μ         ;
    ] / 2
)

CoherentState(α::Complex) = GaussianState(
    [real(α), imag(α)],
    Matrix{Float64}(I, 2, 2) / 2
)

ThermalState(μ::Real) = GaussianState(
    zeros(2),
    Matrix{Float64}(I, 2, 2) * (1 + 2μ) / 2
)

VacuumState(mds::Int) = GaussianState(
    zeros(2mds),
    Matrix{Float64}(I, 2mds, 2mds) / 2
)

end # module