# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- All-Julia test suite (TestItemRunner) validating the v2 generalized framework (`HybridProjectionEngine`/`project`/`tr`/`dot`/`fidelity`/`to_fock`, Wick kernels, click-state algebra, unitaries) for ZALM, SPDC, TMSV, and SIGSAG against ground truth precomputed with the v1 legacy code. Ground truth lives in `test/data/ground_truth.jld2`, generated deterministically (fixed `StableRNG` seed) by `test/generate_ground_truth.jl` (`just ground-truth`).
- `benchmark/benchmarks.jl`: lean regression benchmark suite (Wick contractions, v2 probability/fidelity/density-matrix calculations, legacy per-source headliners) with fixed parameters; used by `just bench` and `just asv`.
- `duankimble` tests and benchmark: v2 Duan-Kimble memory loading is validated against the v1 `spin_density_matrix` ground truth (Julia suite) and the reference Python implementation (`just test-py`) for SPDC and ZALM.

### Changed

- Reformulate A matrix and associated Wick contractions to operate directly in the [α β* α* β] basis instead of [qα pα qβ pβ]. This results in an exponentials speedup because it reduces the number of terms in the expanded moment polynomial from 2^N to 1.
- Modify the A matrix function to directly compute only the upper left 2Nx2N block of A⁻¹, rather than computing the entire 4Nx4N A matrix and inverting. The other blocks of A⁻¹ are not needed, as C polynomials only contain monomials in α and β*. This results in a speedup of 2.9-3.4x on the A matrix function for N=8,12,16.
- Renamed justfile commands: `just test`/`just bench` now run the Julia test suite and the Julia benchmark suite; the Python comparison workflows moved to `just test-py`/`just bench-py` (the Julia half of the comparison benchmarks now lives in `test/python/bench.jl`).
- The Python comparison tests moved out of the routine CI workflow into `.github/workflows/python-comparison.yml`, which runs only when legacy/Python-reference code changes or on manual dispatch.

### Fixed

- The Python wrapper failed to import with recent juliacall releases: `juliacall.Pkg` was removed after 0.9.31 (now activated via Julia's own `Pkg`), and newer juliacall/PythonCall versions force Julia ≤ 1.11, incompatible with Genqo's Julia 1.12 requirement (juliacall is now pinned to 0.9.31).
- `project` with `Vector{Int}` detector outcomes threw a `MethodError`: `nmodes` shadowed `Gabs.nmodes` instead of extending it.
- `fidelity` was ambiguous when both `Genqo` and `Gabs` were loaded; Genqo now extends the shared `QuantumInterface.fidelity` binding re-exported by Gabs/QuantumOpticsBase.
- `tr(::ProjectedPureGaussianState)` asserted an exactly-zero imaginary part, which failed spuriously (imaginary residues ~1e-36) for circuit-built states; the assertion is now a relative tolerance.

## [1.2.0] - 2026-04-30

### Added

- `tools.WTerms` and `tools.WBucket` types for more stable dispatch on precomputed moment polynomial terms by the fast `tools.W()` Wick evaluator. The goal of this is to improve `tools.W()` performance by avoiding heap memory where possible in storing the moment term list. It also allows for dispatch on n, the number of separate q-p variables appearing in a moment polynomial term, which avoids `tools.wick_partitions[n]` Dict lookup on every call to `tools.wick_out()`.

### Changed

- `sigsag` functions now rely on pre-computed terms of moment polynomials for improved speed.
- Wick partitions are now stored in an `Array{Int, 3}` instead of clunky `Vector{Vector{Tuple{Int,Int}}}` for better memory continuity.
- Benchmarking now saves .json files and plots under `.benchmarks/<ISO-timestamp>_<short-commit-hash>/` instead of overwriting `.benchmarks/py-bench.json`, `.benchmarks/jl-bench.json`, and `.benchmarks/benchmark_comparison.svg`.
- `just test` now calculates and reports absolute and relative error figures for each function and combination of parameters.

### Fixed

- `justfile` now updated to reflect Julia workspaces under `docs/` and `test/`.

## [1.1.0] - 2026-04-06

### Added

- Julia implementations of TMSV, SPDC, ZALM, and SIGSAG entangled photon sources
- Python wrapper using juliacall with attrs-based dataclasses
- Comparison test suite validating Julia against reference Python implementation
- Benchmark suite for Julia and Python performance comparison
- CI/CD workflows for testing, releasing, and publishing
- API documentation
