# [Wick Contraction](@id api_wick)

The computational core. Gaussian moments are evaluated by Wick contraction: a symbolic
moment polynomial is compiled once into [`WTerms`](@ref), a type-stable representation
grouped by monomial degree, and then contracted against the inverse kernel by
[`W`](@ref).

Most users never call these directly — [`project`](@ref) and the metrics built on it drive
them, and [`HybridProjectionEngine`](@ref) caches the compiled polynomials. They are
documented because they are the extension point for new moment calculations.

```@docs
Genqo.extract_W_terms
Genqo.WTerms
Genqo.W
Genqo.wick_out
```
