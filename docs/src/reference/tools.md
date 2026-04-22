# Tools

The `tools` module provides the core mathematical machinery shared by all source modules: Gaussian moment evaluation via Wick's theorem, covariance matrix utilities, and the K-matrix construction used to form the contraction kernel.

## Wick Contraction

```@docs
Genqo.tools.wick_out
Genqo.tools.W
Genqo.tools.extract_W_terms
```

## Covariance Matrix Utilities

```@docs
Genqo.tools.permutation_matrix
Genqo.tools.reorder
Genqo.tools.k_function_matrix
```
