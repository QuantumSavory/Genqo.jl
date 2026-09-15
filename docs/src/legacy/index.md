# [Legacy API](@id legacy)

The `Genqo.tools`, `Genqo.tmsv`, `Genqo.spdc`, `Genqo.zalm` and `Genqo.sigsag` submodules
are the first generation of Genqo. Each wraps a single hard-coded source model: a parameter
struct, a fixed Gaussian circuit, and a fixed set of metrics computed from it.

They are retained because the Python wrapper is built on them, and because they are a
convenient shorthand for the four source architectures Genqo was originally written for.
They are not where new work should start.

!!! warning "Superseded"
    New models should be built with the [v2 API](@ref api), which composes arbitrary
    Gaussian circuits out of [Gabs.jl](https://github.com/QuantumSavory/Gabs.jl) states and
    unitaries instead of requiring a bespoke module per source. Everything the legacy
    modules compute — success probability, fidelity, spin density matrices — has a
    direct equivalent there.

Each module is now a thin wrapper: it builds its source as a Gabs `GaussianState`, applies
the detection outcome with [`Genqo.project`](@ref), and reads the metric off the resulting
[`Genqo.ProjectedPureGaussianState`](@ref). The hand-rolled A-matrix construction and Wick
bookkeeping that used to live here are gone, so the legacy entry points and their v2
equivalents are the same calculation to within floating-point round-off.

Two conventions survive from the original code. `tmsv` and `spdc` report their covariance
matrices in qpqp ordering (convert with [`Genqo.tools.reorder`](@ref)), while `zalm` and
`sigsag` report qqpp; and all four use the ħ=1 convention rather than the ħ=2 convention
Gabs uses, so a reported covariance matrix is half the `covar` field of the underlying
Gabs state.

## Contents

```@contents
Pages = ["zalm.md", "spdc.md", "tmsv.md", "sigsag.md", "tools.md"]
Depth = 2
```
