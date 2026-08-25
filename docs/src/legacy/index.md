# [Legacy API](@id legacy)

The `Genqo.tools`, `Genqo.tmsv`, `Genqo.spdc`, `Genqo.zalm` and `Genqo.sigsag` submodules
are the first generation of Genqo. Each wraps a single hard-coded source model: a parameter
struct, a hand-built covariance matrix, and a fixed set of metrics computed from it.

They are retained because they are the numerical oracle the current framework is validated
against, and because the Python wrapper is built on them. They are not where new work
should start.

!!! warning "Superseded"
    New models should be built with the [v2 API](@ref api), which composes arbitrary
    Gaussian circuits out of [Gabs.jl](https://github.com/QuantumSavory/Gabs.jl) states and
    unitaries instead of requiring a bespoke module per source. Everything the legacy
    modules compute — success probability, fidelity, spin density matrices — has a
    direct equivalent there.

The legacy modules build covariance matrices in qpqp ordering and reorder to qqpp with
[`Genqo.tools.reorder`](@ref) before contraction, and they use the ħ=1 convention rather
than the ħ=2 convention Gabs uses.

## Contents

```@contents
Pages = ["zalm.md", "spdc.md", "tmsv.md", "sigsag.md", "tools.md"]
Depth = 2
```
