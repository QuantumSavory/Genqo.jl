# [ZALM](@id zalm_ref)

The `zalm` module models a Zero-Added-Loss Multiplexing (ZALM) cascaded entanglement source, in which two SPDC sources are combined at a central Bell-state measurement (BSM) station. A coincidence click at the BSM heralds a remote entangled spin-spin state. Dark counts and all three efficiency channels (detection, outcoupling, BSM) can be modeled.

## Type

```@docs
Genqo.zalm.ZALM
```

## Source Metrics

```@docs
Genqo.zalm.fidelity
Genqo.zalm.probability_success
Genqo.zalm.spin_density_matrix
```

## Internal Matrices

```@docs
Genqo.zalm.covariance_matrix
Genqo.zalm.loss_bsm_matrix_fid
Genqo.zalm.loss_bsm_matrix_pgen
Genqo.zalm.dmijZ
```

## Moment Polynomials

```@docs
Genqo.zalm.moment_vector
Genqo.zalm.moment_terms
```
