# Getting Started

## Installation

### Julia

Genqo.jl requires Julia 1.12 or later. Install it from the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/QuantumSavory/Genqo.jl")
```

Once registered in the Julia General registry, it can be installed with:

```julia
Pkg.add("Genqo")
```

### Python

The Python wrapper requires Python ≥ 3.9 and uses [juliacall](https://github.com/JuliaPy/PythonCall.jl) to call Julia under the hood.

```bash
pip install genqo
```

## Quick Start

### Julia

Load the package and access sources through their submodule namespaces:

```julia
using Genqo

# TMSV: probability of coincidence detection
p = tmsv.probability_success(1e-2, 0.9)

# SPDC: Bell-state fidelity and spin-spin density matrix
F = spdc.fidelity(1e-2, 0.8, 0.6)
ρ = spdc.spin_density_matrix(1e-2, 0.8, 0.6, [0, 1, 0, 1])

# ZALM: fidelity, probability of success, and density matrix
F = zalm.fidelity(1e-2, 0.8, 0.6, 0.9)
p = zalm.probability_success(1e-2, 0.8, 0.6, 0.9, 0.0)
ρ = zalm.spin_density_matrix(1e-2, 0.8, 0.6, 0.9, [1, 0, 1, 1, 0, 0, 1, 0])

# SIGSAG: fidelity and probability of success
F = sigsag.fidelity(1e-2, 0.8, 0.6)
p = sigsag.probability_success(1e-2, 0.8, 0.6)
```

You can also use the struct-based API, which stores parameters and lets you call functions without
repeatedly passing arguments:

```julia
src = zalm.ZALM(mean_photon=1e-2, outcoupling_efficiency=0.8,
                detection_efficiency=0.6, bsm_efficiency=0.9, dark_counts=0.01)

zalm.fidelity(src)
zalm.probability_success(src)
```

### Python

```python
import genqo as gq

# TMSV
tmsv = gq.TMSV(mean_photon=1e-2, detection_efficiency=0.9)
p = tmsv.probability_success()

# SPDC
spdc = gq.SPDC(mean_photon=1e-2, outcoupling_efficiency=0.8, detection_efficiency=0.6)
F = spdc.fidelity()
rho = spdc.spin_density_matrix([0, 1, 0, 1])

# ZALM
zalm = gq.ZALM(mean_photon=1e-2, outcoupling_efficiency=0.8,
               detection_efficiency=0.6, bsm_efficiency=0.9, dark_counts=0.01)
F = zalm.fidelity()
p = zalm.probability_success()

# SIGSAG
sigsag = gq.SIGSAG(mean_photon=1e-2, outcoupling_efficiency=0.8, detection_efficiency=0.6)
F = sigsag.fidelity()
```
