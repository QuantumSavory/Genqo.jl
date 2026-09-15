# [API](@id api)

The Genqo API models an entanglement source as a **pure Gaussian circuit followed by
photon-number-resolved detection**. Gaussian states and unitaries come from
[Gabs.jl](https://github.com/QuantumSavory/Gabs.jl); Genqo supplies the detection layer and
evaluates the resulting non-Gaussian state exactly, without truncating at low mean photon
number.

The modeling process follows four general steps:

1. Build a pure Gaussian state with Gabs, optionally using Genqo's extra [unitaries](@ref api_unitaries).
2. Describe the detector clicks with a [`projector`](@ref).
3. Combine the two with [`project`](@ref). Nothing is computed yet. If loss is desired, pass a mode-wise loss vector as the `η` keyword.
4. Ask for the quantity you want: [`tr`](@ref) for the heralding probability, [`fidelity`](@ref) against a target, [`dot`](@ref) for a single density-matrix element, or [`to_fock`](@ref) for a truncated photon-photon density matrix. Loading into quantum memories is also supported through [`duankimble`](@ref) and [`emissiveload`](@ref), which return spin-spin density matrices.

```julia
using Genqo, Gabs

# 1. A two-mode squeezed vacuum state
st = eprstate(QuadBlockBasis(2), asinh(√1e-2), Float64(π))

# 2-3. Herald on a coincidence click, with per-mode detector efficiency
ρ = project(st, projector([1, 1]); η = [0.9, 0.8])

# 4. Probability that the herald fires
tr(ρ)
```

Modes that are not detected are marked with a colon and stay in the output state:

```julia
# Detect mode 1; mode 2 is left free and survives into the density matrix
ρ = project(st, projector([1, :]))

freemodes(ρ)          # [2]
fidelity(clicks([1]), ρ)
to_fock(ρ)            # a QuantumOpticsBase.Operator over the free modes
```

A detection outcome that several click patterns satisfy is a sum of projectors:

```julia
Π = projector([1, 0]) + projector([0, 1])
tr(project(st, Π))    # tr is additive over the summed patterns
```

## Performance

Evaluation compiles a symbolic moment polynomial per click pattern and contracts it by Wick's theorem. That compilation is memoized in a [`HybridProjectionEngine`](@ref). By default, Genqo automatically manages a global dictionary of engines of various sizes. For finer control of the scratch space used by Genqo, an `engine` keyword may also be passed explicitly:

```julia
engine = HybridProjectionEngine(2)
μ = 0.01:0.01:0.5
st = eprstate(QuadBlockBasis(2), asinh.(√.μ), Float64(π))
Pg = tr.(project.(st, projector([1,1])); engine)
```

[`duankimble`](@ref) and [`emissiveload`](@ref) go further and return a matrix whose
entries are contracted only when read (see [Memory Loading](@ref api_memory)).

## Contents

```@contents
Pages = ["clicks.md", "projection.md", "memory.md", "unitaries.md", "wick.md"]
Depth = 2
```
