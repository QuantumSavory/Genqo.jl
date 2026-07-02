using Gabs: SymplecticBasis, GaussianUnitary

export modeswap


modeswap(basis::SymplecticBasis) = GaussianUnitary(
    basis,
    zeros(2*2),
    [
        0   1   0   0 ;
        1   0   0   0 ;
        0   0   0   1 ;
        0   0   1   0 ;
    ]
)
