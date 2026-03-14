module states

export QuantumState, GaussianState, FockState


abstract type QuantumState end

mutable struct GaussianState <: QuantumState
    covariance::Matrix{Float64}
end

mutable struct FockState <: QuantumState
    density::Array{ComplexF64}
    cutoffs::Vector{Int}
end

# All states in qqpp ordering

TMSVState(μ::Real)::GaussianState = GaussianState(
    [
        1+2μ         2√(μ*(μ+1))  0             0            ;
        2√(μ*(μ+1))  1+2μ         0             0            ;
        0            0            1+2μ          -2√(μ*(μ+1)) ;
        0            0            -2√(μ*(μ+1))  1+2μ         ;
    ] / 2
)

end # module