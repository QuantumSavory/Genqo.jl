# Genqo.jl

**Genqo.jl** [richardson_full-stack_2026,gunnell_computational_2026](@cite) is a package for efficiently modeling hybrid Gaussian / non-Gaussian CV quantum optics. It is useful for problems such as modeling entangled photon sources, general GBS-based multimode state preparation, and photonic quantum computing.

Genqo provides an interface for working with non-Gaussian projections of Gaussian states in a computationally efficient manner. In particular, the [`project`](@ref) function takes a [`Gabs.jl`](https://github.com/QuantumSavory/Gabs.jl) `::GaussianState` and produces an intermediate of type `::ProjectedPureGaussianState`, which performs no actual computations until functions such as [`tr`](@ref), [`dot`](@ref), [`duankimble`](@ref) etc. are called on it. Each method for this type implements a closed form involving a matrix Hafnian, which is handed off to [`TheEggman.jl`](https://github.com/QuantumSavory/TheEggman.jl) for fast evaluation. This approach sidesteps any computations involving truncated infinite-dimensional density operators, which are slow, memory-intensive, and inexact. Presently, only projectors in the Fock basis are supported, corresponding to the outcomes of PNRs (photon number resolving detectors).

Genqo also provides pre-assembled implementations of four common entangled photon source architectures:

| Source | Description |
|--------|-------------|
| [`spdc`](@ref spdc_ref) | Unheralded SPDC Source |
| [`zalm`](@ref zalm_ref) | Cascaded source [dhara_heralded-multiplexed_2022](@cite) (single-frequency-mode ZALM [chen_zero-added-loss_2023](@cite)) |
| [`sigsag`](@ref sigsag_ref) | Alternative heralded Bell pair source [chahine_heralded_2026](@cite) |
| [`tmsv`](@ref tmsv_ref) | Two-Mode Squeezed Vacuum |

These fixed architectures are the original interface and are documented under [Legacy](@ref legacy). The current [API](@ref api) generalizes them: it models an arbitrary pure Gaussian circuit followed by photon-number-resolved detection, so a new source design no longer needs a module of its own.

For an in-depth presentation of the mathematical foundations of this package, see [richardson_full-stack_2026](@cite).

## Getting Started

See the [Getting Started](@ref getting_started) page for installation and a quick-start example.

## API Reference

The [API](@ref api) section documents the current interface: build a Gaussian circuit, specify detection outcomes with a projector, and read off the metric you need.

The per-source modules in the table above are documented under [Legacy](@ref legacy). They remain the numerical oracle the current framework is validated against.

## Bibliography

```@bibliography
```
