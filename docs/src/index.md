# Genqo.jl

**Genqo.jl** is a hybrid Gaussian / non-Gaussian quantum optics state engine for modeling entanglement sources used in quantum networking.

It numerically computes performance metrics, such as Bell-state fidelity, probability of success, and spin-spin density matrices, for four photonic entanglement source architectures:

| Source | Description |
|--------|-------------|
| [`zalm`](@ref zalm_ref) | Cascaded ZALM source (two SPDC sources + central heralding BSM) |
| [`spdc`](@ref spdc_ref) | Single SPDC source with direct Bell-state measurement |
| [`tmsv`](@ref tmsv_ref) | Two-Mode Squeezed Vacuum (the simplest Gaussian entangled state) |
| [`sigsag`](@ref sigsag_ref) | Single Sagnac source |

All sources support loss modeling through bell-state measurement efficiency, detection efficiency, outcoupling efficiency, and (for ZALM) dark counts.
A Python wrapper is provided for users who prefer Python; it exposes the same API via [juliacall](https://github.com/JuliaPy/PythonCall.jl).

## Getting Started

See the [Getting Started](@ref getting_started) page for installation and a quick-start example.

## Reference

Full API documentation is in the [Reference](@ref zalm_ref) section.
