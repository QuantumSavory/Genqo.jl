# [SPDC](@id spdc_ref)

The `spdc` module models a Spontaneous Parametric Down-Conversion entanglement source. A single
nonlinear crystal generates entangled photon pairs; the heralding photons pass through a
Bell-state measurement that projects the remote spin qubits into an entangled state.

## Type

```@docs
Genqo.spdc.SPDC
```

## Source Metrics

```@docs
Genqo.spdc.fidelity
Genqo.spdc.spin_density_matrix
```

## Internal Matrices

```@docs
Genqo.spdc.covariance_matrix
Genqo.spdc.loss_bsm_matrix_fid
Genqo.spdc.dmijZ
```

## Moment Polynomials

```@docs
Genqo.spdc.moment_vector
Genqo.spdc.moment_terms
```
