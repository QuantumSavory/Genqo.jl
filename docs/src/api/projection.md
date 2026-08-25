# [Projection and Metrics](@id api_projection)

[`project`](@ref) pairs a pure Gaussian state with a detection outcome and detector
efficiencies. It computes nothing on its own — the result is a description of a
measurement, and the work happens when a metric is requested from it.

```julia
ρ = project(st, projector([1, 1]); η = [0.9, 0.8])

tr(ρ)                      # heralding probability
fidelity(ψ, ρ)             # fidelity against a click-state target
dot(ψ', ρ, φ)              # one unnormalized density-matrix element
to_fock(ρ)                 # the whole heralded density matrix
```

`tr` and `dot` are additive over the click patterns of a summed projector, so a herald that
several patterns satisfy needs no special handling.

## Projecting

```@docs
Genqo.project
Genqo.ProjectedPureGaussianState
Genqo.AbstractProjectedPureGaussianState
Genqo.AbstractProjectedState
```

## Metrics

```@docs
Genqo.tr(::Genqo.ProjectedPureGaussianState)
Genqo.dot(::Genqo.ClickStateBra, ::Genqo.ProjectedPureGaussianState, ::Genqo.ClickStateKet)
Genqo.fidelity(::Genqo.ClickStateKet, ::Genqo.ProjectedPureGaussianState)
Genqo.to_fock
```

## Engines

Every metric accepts an `engine` keyword. Supplying one explicitly reuses its compiled
moment-polynomial cache across calls; omitting it falls back to a process-wide engine for
that mode count.

```@docs
Genqo.HybridProjectionEngine
Genqo.get_default_engine
Genqo.get_phase_space_generators_half
Genqo.get_phase_space_generators_full
Genqo.AbstractProjectionEngine
```
