# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Genqo.jl models quantum entanglement sources (SPDC, ZALM, etc.) using a hybrid Gaussian/non-Gaussian framework that avoids low-mean-photon-number truncation. It is a Julia rewrite of an earlier Python package (kept as a reference implementation in `test/genqo_old_pkg/`), with performance as a primary design goal (~100-1000x over the Python original). A Python wrapper via juliacall lives in `python/`.

Requires Julia 1.12. The repo uses Julia workspaces: the root `Project.toml` declares `test/`, `docs/`, and `benchmark/` as workspace projects.

## Commands

Common tasks are in the `justfile`:

- `just install` — instantiate root/test/docs/benchmark Julia projects and set up the Python venv (`python/.venv`) with the wrapper and reference package
- `just test` — Julia test suite (what CI's Julia job runs via `Pkg.test()`): TestItemRunner `@testitem`s in `test/test_*.jl` validating the v2 framework against v1 ground truth in `test/data/ground_truth.jld2`
- `just ground-truth` — regenerate that JLD2 with the v1 legacy code (deterministic, fixed `StableRNG` seed in `test/gt_common.jl`); rerun and commit whenever the parameter sets change
- `just bench [func]` — Julia regression benchmarks (`benchmark/benchmarks.jl`), e.g. `just bench project.tr`; `func` is a substring filter
- `just asv <rev>` — AirspeedVelocity.jl regression benchmarks against previous commits (same suite)
- `just test-py` — pytest comparison suite validating the v1 legacy code against the reference Python implementation, plus a precision report (writes to `.benchmarks/`)
- `just bench-py [func]` — Julia vs Python benchmarks (`test/python/bench.jl`), e.g. `just bench-py spdc.spin_density_matrix`
- `just build-docs` — build Documenter docs
- `just bump patch|minor|major` — version bump via bump-my-version

To run a subset of test items without the full suite, use a filter with TestItemRunner in the test project, e.g.:

```sh
julia --project=test -e 'using TestItemRunner; TestItemRunner.run_tests("."; filter=ti->occursin("ZALM", ti.name))'
```

CI (`.github/workflows/ci.yml`) runs both the Julia tests and the Python comparison tests on PRs to main.

## Architecture

`src/Genqo.jl` contains two generations of code:

### v2 generalized framework (top-level `Genqo`, under active development)

Built on [Gabs.jl](https://github.com/apkille/Gabs.jl) `GaussianState`/`GaussianUnitary` types (note: Gabs uses the ħ=2 convention; covariance matrices are rescaled by `st.ħ` before use). Three files:

- `src/wick.jl` — the computational core: evaluates Gaussian moments by Wick contraction. Symbolic moment polynomials (Nemo `MPoly` over `ComplexField`) are precompiled by `extract_W_terms` into `WTerms`, a heterogeneous tuple of `WBucket{N}`s grouped by monomial degree so the evaluator `W(::WTerms, Ainv)` is fully type-stable. Each monomial's contraction is a hafnian computed by a `@generated`, fully unrolled recursive expansion (`_hafnian`), which is far cheaper than enumerating `wick_partitions`.
- `src/projectors.jl` — `project(state, ::ClickProjector; engine::HybridProjectionEngine, η)` implements photon-number-resolved detection on a pure Gaussian state. `ClickProjector` (built with `projector`, `Colon` = traced-out mode) is a sum of click-pattern projectors: only `+` is allowed (idempotence-preserving; no scalar multiples), traceout placement must agree across summed patterns, and `tr`/`dot` are additive over patterns. `ClickStateKet`/`ClickStateBra` (built with `clicks`, non-negative patterns only) are superpositions with a bra-ket algebra (`dot`, `adjoint`, `+`, scalar `*`, `norm`) used as targets for matrix elements. `tr` gives success probability, `dot(bra, projected_state, ket)` gives density-matrix elements, `to_fock` builds a QuantumOpticsBase `Operator`. `A_matrix` constructs the Gaussian contraction kernel from the covariance matrix, folding detector loss (η) and traced-out modes into the `G` block. Compiled moment polynomials are memoized in the engine's `C_poly_cache` behind a `ReentrantLock`.
- `src/unitaries.jl` — extra Gaussian unitaries (`modeswap`, `greenmachine`) in the Gabs style, each implemented for both `QuadPairBasis` (qpqp ordering) and `QuadBlockBasis` (qqpp ordering).

### v1 legacy code (`src/legacy/`, submodules `tools`, `tmsv`, `spdc`, `zalm`, `sigsag`)

Each source model is a self-contained submodule with a parameter struct (e.g. `zalm.ZALM`) and functions like `covariance_matrix`, `spin_density_matrix`, `probability_success`, `fidelity`. These build covariance matrices by hand (qpqp ordering, reordered to qqpp by `tools.reorder` before `tools.k_function_matrix`) and share the Wick machinery from `src/wick.jl`. This is the code path the Python wrapper and the comparison suites exercise; it is validated numerically against `test/genqo_old_pkg` and serves as the ground-truth oracle for the v2 Julia tests (legacy covariances are ħ=1; wrap with `legacy_state` from `test/gt_common.jl` to feed them to v2 `project`).

### Performance conventions

Hot paths avoid heap allocation and dynamic dispatch: degree-typed `NTuple` indices, `Val`-based constructors, recursive tuple iteration instead of `for` over heterogeneous tuples, `@generated` unrolling, `@inbounds` in inner loops, and caching of Wick partitions and compiled polynomials. When touching `wick.jl` or moment-polynomial code, preserve type stability and check for regressions with `just asv <rev>` or `just bench`.

## Conventions

- Docs are built with `checkdocs = :exports` — every exported symbol needs a docstring or the docs build fails. New modules must be added to `modules` in `docs/make.jl`.
- `CHANGELOG.md` follows Keep a Changelog; record notable changes under `[Unreleased]`.
- Version lives in `Project.toml` and `python/pyproject.toml`, kept in sync by bump-my-version (`just bump`).
- Tutorials in `docs/tutorial/` are plain Julia scripts and notebooks used as working examples during development.
