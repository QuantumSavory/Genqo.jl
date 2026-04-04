# Genqo.jl

**Genqo.jl** is a hybrid Gaussian / non-Gaussian quantum optics state engine for modeling entanglement sources used in quantum networking.

It computes physically exact performance metrics — Bell-state fidelity, probability of success, and spin-spin density matrices — for four photonic entanglement source architectures:

| Source | Description |
|--------|-------------|
| [`zalm`](@ref zalm_ref) | Cascaded ZALM source (two SPDC sources + central BSM) |
| [`spdc`](@ref spdc_ref) | Single SPDC source with direct Bell-state measurement |
| [`tmsv`](@ref tmsv_ref) | Two-Mode Squeezed Vacuum — the simplest Gaussian entangled state |
| [`sigsag`](@ref sigsag_ref) | Sagnac-loop single-pass source |

All sources support loss modeling through detection efficiency, outcoupling efficiency, and (for ZALM) dark counts.
A Python wrapper is provided for users who prefer Python; it exposes the same API via [juliacall](https://github.com/JuliaPy/PythonCall.jl).

See the [Getting Started](@ref) page for installation and a quick-start example.

## Reference

Full API documentation is in the [Reference](@ref zalm_ref) section.
