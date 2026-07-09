using Gabs: SymplecticBasis, QuadPairBasis, QuadBlockBasis, GaussianUnitary

export modeswap, greenmachine


function modeswap(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}; ħ = 2) where {Td,Ts,N<:Int}
    disp, symplectic = _modeswap(basis)
    return GaussianUnitary(basis, Td(disp), Ts(symplectic); ħ = ħ)
end
modeswap(::Type{T}, basis::SymplecticBasis{N}; ħ = 2) where {T,N<:Int} = modeswap(T, T, basis; ħ = ħ)
function modeswap(basis::SymplecticBasis{N}; ħ = 2) where {N<:Int}
    disp, symplectic = _modeswap(basis)
    return GaussianUnitary(basis, disp, symplectic; ħ = ħ)
end
function _modeswap(basis::QuadPairBasis{N}) where {N<:Int}
    nmodes = basis.nmodes
    disp = falses(2*nmodes)
    symplectic = falses(2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes÷2)
        symplectic[4*i-3, 4*i-1] = 1
        symplectic[4*i-2, 4*i  ] = 1
        symplectic[4*i-1, 4*i-3] = 1
        symplectic[4*i  , 4*i-2] = 1
    end
    return disp, symplectic
end
function _modeswap(basis::QuadBlockBasis{N}) where {N<:Int}
    nmodes = basis.nmodes
    disp = falses(2*nmodes)
    symplectic = falses(2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes÷2)
        symplectic[4*i-3, 4*i-2] = 1
        symplectic[4*i-2, 4*i-3] = 1
        symplectic[4*i-1, 4*i  ] = 1
        symplectic[4*i  , 4*i-1] = 1
    end
    return disp, symplectic
end

function greenmachine(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, m::M; ħ = 2) where {Td,Ts,N<:Int,M<:Int}
    disp, symplectic = _greenmachine(basis, m)
    return GaussianUnitary(basis, Td(disp), Ts(symplectic); ħ = ħ)
end
greenmachine(::Type{T}, basis::SymplecticBasis{N}, m::M; ħ = 2) where {T,N<:Int,M<:Int} = greenmachine(T, T, basis, m; ħ = ħ)
function greenmachine(basis::SymplecticBasis{N}, m::M; ħ = 2) where {N<:Int,M<:Int}
    disp, symplectic = _greenmachine(basis, m)
    return GaussianUnitary(basis, disp, symplectic; ħ = ħ)
end
function _greenmachine(basis::QuadPairBasis{N}, m::M) where {N<:Int,M<:Int}
    nmodes = basis.nmodes
    m == nmodes || throw(ArgumentError("Mode mismatch: cannot construct $m mode Green machine in $nmodes mode basis")) # TODO: support embedding multiple green machines in one matrix, mirroring behavior of beamsplitter()
    local p
    try
        p = Int(log2(m))
    catch e
        if e isa InexactError; throw(ArgumentError("m must be a power of 2")) else rethrow(e) end
    end
    disp = zeros(2*nmodes)
    Hs = vcat(repeat([[1 1; 1 -1]/√2], p), [[1 0; 0 1]])
    symplectic = kron(Hs...)
    return disp, symplectic
end
function _greenmachine(basis::QuadBlockBasis{N}, m::M) where {N<:Int,M<:Int}
    nmodes = basis.nmodes
    m == nmodes || throw(ArgumentError("Mode mismatch: cannot construct $m mode Green machine in $nmodes mode basis"))
    local p
    try
        p = Int(log2(m))
    catch e
        if e isa InexactError; throw(ArgumentError("m must be a power of 2")) else rethrow(e) end
    end
    disp = zeros(2*nmodes)
    Hs = vcat([[1 0; 0 1]], repeat([[1 1; 1 -1]/√2], p))
    symplectic = kron(Hs...)
    return disp, symplectic
end
