# [Memory Loading](@id api_memory)

[`duankimble`](@ref) and [`emissiveload`](@ref) load a heralded photonic state into quantum
memories and return the resulting spin density matrix as a
`QuantumOpticsBase.Operator`.

Both return an operator backed by a [`LazyDensityMatrix`](@ref): entries are contracted on
first access and memoized thereafter. This behaves like an ordinary dense operator, but
skips the entries a calculation never reads — `tr(ρ)` touches only the `M` diagonal entries
out of `M²`. Any whole-matrix operation materializes it, so arithmetic such as `ρ * 4` or
`ρ ./= p` yields an ordinary dense operator rather than another lazy wrapper.

```julia
ρ = duankimble(project(st, Π), [1, 1])

tr(ρ)              # contracts only the diagonal
ncomputed(ρ.data)  # how many entries have actually been evaluated
```

## Loading Models

```@docs
Genqo.duankimble
Genqo.emissiveload
```

## Lazy Density Matrices

```@docs
Genqo.LazyDensityMatrix
Genqo.ncomputed
```
