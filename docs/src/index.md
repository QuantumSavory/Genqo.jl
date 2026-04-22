# Genqo.jl

**Genqo.jl** [richardson_full-stack_2026](@cite) is a hybrid Gaussian / non-Gaussian quantum optics state engine for modeling entanglement sources used in quantum networking.

It numerically computes performance metrics, such as Bell-state fidelity, probability of success, and spin-spin density matrices, for four photonic entanglement source architectures:

| Source | Description |
|--------|-------------|
| [`tmsv`](@ref tmsv_ref) | Two-Mode Squeezed Vacuum (the simplest Gaussian entangled state) |
| [`spdc`](@ref spdc_ref) | Single SPDC Source |
| [`zalm`](@ref zalm_ref) | Cascaded source [dhara_heralded-multiplexed_2022](@cite) (equivalent to single frequency mode of a ZALM source [chen_zero-added-loss_2023](@cite)) |
| [`sigsag`](@ref sigsag_ref) | Alternative heralded Bell pair source proposed by Yousef Chahine et al. [chahine_heralded_2026](@cite) |

All sources support loss modeling through bell-state measurement efficiency, detection efficiency, outcoupling efficiency, and (for ZALM) dark counts.
A Python wrapper is provided for users who prefer Python; it exposes the same API via [juliacall](https://github.com/JuliaPy/PythonCall.jl).

For an in-depth presentation of the mathematical foundations of this package, see [richardson_full-stack_2026](@cite).

## Getting Started

See the [Getting Started](@ref getting_started) page for installation and a quick-start example.

## API Reference

Full API documentation is in the [Reference](@ref zalm_ref) section.

## Bibliography

```@bibliography
```
