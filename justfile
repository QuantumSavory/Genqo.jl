# Install genqo.jl and its Python wrapper
install:
    julia --project=. -e 'using Pkg; Pkg.instantiate()'
    julia --project=test/ -e 'using Pkg; Pkg.instantiate()'
    julia --project=docs/ -e 'using Pkg; Pkg.instantiate()'

    just venv
    . python/.venv/bin/activate && \
    pip install -e python/[test] && \
    pip install -e test/genqo_old_pkg

# Run tests comparing Julia and Python genqo implementations
test:
    #!/usr/bin/env bash
    set -e
    echo "Running tests comparing Julia and Python genqo"
    BENCH_DIR=".benchmarks/test_$(date -u +%Y-%m-%dT%H:%M:%S)_$(git rev-parse --short HEAD)"
    mkdir -p "$BENCH_DIR"
    . python/.venv/bin/activate
    pytest test/python/test_compare_with_python.py --bench-dir="$BENCH_DIR"
    python test/python/precision_table.py "$BENCH_DIR"

# Run benchmarks for <func>, e.g. just bench spdc.spin_density_matrix (benchmarks all by default)
bench func="":
    #!/usr/bin/env bash
    set -e
    echo "Running benchmarks for Julia and Python genqo"
    BENCH_DIR=".benchmarks/bench_$(date -u +%Y-%m-%dT%H:%M:%S)_$(git rev-parse --short HEAD)"
    mkdir -p "$BENCH_DIR"
    julia --project=test/ test/bench.jl "{{func}}" "$BENCH_DIR"
    . python/.venv/bin/activate
    pytest test/python/test_gqpy_bench.py{{ if func != "" { "::test_" + replace(func, '.', '__') } else { "" } }} --benchmark-json="$BENCH_DIR/py-bench.json"
    python test/python/plot_comparison.py "$BENCH_DIR"

# Run AirspeedVelocity.jl benchmarks to compare with previous commits (requires benchpkg and benchpkgplot on PATH)
asv rev:
    #!/usr/bin/env bash
    set -e
    echo "Running AirspeedVelocity.jl benchmarks for revisions {{rev}}"
    BENCH_DIR=".benchmarks/asv_$(date -u +%Y-%m-%dT%H:%M:%S)_{{rev}}"
    mkdir -p "$BENCH_DIR"
    benchpkg --rev="{{rev}}" -o="$BENCH_DIR" -s="test/bench.jl"
    benchpkgplot Genqo --rev="{{rev}}" -i="$BENCH_DIR" -o="$BENCH_DIR"

# Build documentation
build-docs:
    julia --project=docs/ docs/make.jl

# Bump version: just bump patch | minor | major
bump part:
    bump-my-version bump {{part}}

# Create virtual environment for Python wrapper
venv:
    if [ ! -d python/.venv ]; then \
        python3 -m venv python/.venv; \
    fi
