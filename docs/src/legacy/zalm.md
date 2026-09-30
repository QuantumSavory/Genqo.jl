# [ZALM](@id zalm_ref)

The `zalm` module models a Zero-Added-Loss Multiplexing (ZALM) cascaded entanglement source. The ZALM architecture uses two SPDC sources, interfering half of the modes from each source on a pair of 50/50 beamsplitters to perform a Bell-state measurement (BSM). A heralding click pattern signifies a probabilistic photon-photon Bell state between the output modes. Dark counts and all three efficiency channels can be modeled.

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

## Circuit

```@docs
Genqo.zalm.covariance_matrix
```
