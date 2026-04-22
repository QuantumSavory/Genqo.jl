# [TMSV](@id tmsv_ref)

The `tmsv` module models a Two-Mode Squeezed Vacuum (TMSV) state — the simplest Gaussian entangled state. Both modes are produced by a single parametric interaction and characterized by the mean photon number μ. This source is primarily used as a building block for the SPDC and ZALM modules and as a validation baseline since it does not produce dual-rail Bell states.

## Type

```@docs
Genqo.tmsv.TMSV
```

## Source Metrics

```@docs
Genqo.tmsv.probability_success
```

## Internal Matrices

```@docs
Genqo.tmsv.covariance_matrix
Genqo.tmsv.loss_matrix_pgen
```
