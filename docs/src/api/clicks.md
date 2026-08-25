# [Click States and Projectors](@id api_clicks)

Two distinct things are described in the photon-number basis, and they are not
interchangeable.

A **projector** is what the detectors did. It is an outcome, built with
[`projector`](@ref), and it may leave modes undetected (`:`). It is idempotent, so it can
only be added to other projectors, never scaled.

A **click state** is a target to compare against. It is built with [`clicks`](@ref), lives
only on the free modes, and carries amplitudes, so it has a full bra-ket algebra. Click
states are what you pass to [`fidelity`](@ref) and [`dot`](@ref).

```julia
Π = projector([1, 0, :])            # detected 1 photon, 0 photons, one mode left free
ψ = (clicks([1]) + clicks([0]))/√2  # a target over that free mode
```

## Projectors

```@docs
Genqo.projector(::Vector{Int})
Genqo.ClickProjector
Genqo.freemodes(::Genqo.ClickProjector)
Genqo.nfreemodes(::Genqo.ClickProjector)
Genqo.AbstractClickOperator
```

## Click States

```@docs
Genqo.clicks(::Vector{Int})
Genqo.ClickStateKet
Genqo.ClickStateBra
Genqo.dot(::Genqo.ClickStateBra, ::Genqo.ClickStateKet)
Genqo.norm(::Genqo.ClickStateKet)
Genqo.AbstractClickState
```
