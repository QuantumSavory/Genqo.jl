# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Improved type stability across TMSV, SPDC, ZALM, SIGSAG, and tools modules by giving explicit return types to 

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
