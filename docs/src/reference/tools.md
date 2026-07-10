# Tools

The core mathematical machinery shared by all source modules: Gaussian moment evaluation via Wick's theorem and hybrid Gaussian/non-Gaussian projection (top-level `Genqo`), plus the covariance matrix utilities and K-matrix construction in the `tools` module.

## Wick Contraction

```@docs
Genqo.wick_out
Genqo.W
Genqo.WTerms
Genqo.extract_W_terms
```

## Hybrid Projection

```@docs
Genqo.tr(::Genqo.ProjectedPureGaussianState)
Genqo.to_fock
```

## Covariance Matrix Utilities

```@docs
Genqo.tools.permutation_matrix
Genqo.tools.reorder
Genqo.tools.k_function_matrix
```
