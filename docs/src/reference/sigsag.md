# [SIGSAG](@id sigsag_ref)

The `sigsag` module models a Sagnac-loop (single-pass) entanglement source. A single SPDC
crystal is placed inside a Sagnac interferometer; the counter-propagating modes are routed
through a BSM beamsplitter network. Compared to ZALM, the Sagnac architecture uses fewer
optical components and has no cascaded BSM stage.

## Type

```@docs
Genqo.sigsag.SIGSAG
```

## Source Metrics

```@docs
Genqo.sigsag.fidelity
Genqo.sigsag.probability_success
```

## Internal Matrices

```@docs
Genqo.sigsag.covariance_matrix
Genqo.sigsag.loss_bsm_matrix_fid
Genqo.sigsag.loss_bsm_matrix_pgen
```

## Moment Polynomials

```@docs
Genqo.sigsag.moment_vector
```
