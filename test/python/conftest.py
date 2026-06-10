"""Pytest configuration for genqo tests."""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

import genqo_old as gqpy

if TYPE_CHECKING:
    import genqo as gqjl


def pytest_addoption(parser):
    parser.addoption("--bench-dir", default=None, help="Directory to save benchmark artifacts")


@pytest.fixture
def tmsv_test_cases() -> list[dict]:
    """Return a list of test cases (dictionary of parameters)."""
    detection_efficiencies = [1.0, 0.6, 0.34]
    mean_photons = [1e-4, 1e-3, 1e-2]

    test_cases = []
    for detection_efficiency, mean_photon in zip(detection_efficiencies, mean_photons):
        test_cases.append({
            "detection_efficiency": detection_efficiency,
            "mean_photon": mean_photon,
        })
    return test_cases

@pytest.fixture
def spdc_test_cases() -> list[dict]:
    """Return a list of test cases (dictionary of parameters)."""
    bsm_efficiencies = [1.0, 0.5, 0.75]
    outcoupling_efficiencies = [1.0, 0.75, 0.5]
    detection_efficiencies = [1.0, 0.6, 0.34]
    mean_photons = [1e-4, 1e-3, 1e-2]

    test_cases = []
    for bsm_efficiency, outcoupling_efficiency, detection_efficiency, mean_photon in zip(bsm_efficiencies, outcoupling_efficiencies, detection_efficiencies, mean_photons):
        test_cases.append({
            "bsm_efficiency": bsm_efficiency,
            "outcoupling_efficiency": outcoupling_efficiency,
            "detection_efficiency": detection_efficiency,
            "mean_photon": mean_photon,
        })
    return test_cases

@pytest.fixture
def zalm_test_cases() -> list[dict]:
    """Return a list of test cases (dictionary of parameters)."""
    bsm_efficiencies = [1.0, 0.5, 0.75]
    outcoupling_efficiencies = [1.0, 0.75, 0.5]
    detection_efficiencies = [1.0, 0.6, 0.34]
    mean_photons = [1e-4, 1e-3, 1e-2]
    dark_counts = [0.0, 0.01, 0.05]

    test_cases = []
    for bsm_efficiency, outcoupling_efficiency, detection_efficiency, mean_photon, dark_count in zip(bsm_efficiencies, outcoupling_efficiencies, detection_efficiencies, mean_photons, dark_counts):
        test_cases.append({
            "bsm_efficiency": bsm_efficiency,
            "outcoupling_efficiency": outcoupling_efficiency,
            "detection_efficiency": detection_efficiency,
            "mean_photon": mean_photon,
            "dark_counts": dark_count,
        })
    return test_cases

@pytest.fixture
def sigsag_test_cases() -> list[dict]:
    """Return a list of test cases (dictionary of parameters)."""
    mean_photons = [1e-4, 1e-3, 1e-2]
    bsm_efficiencies = [1.0, 0.5, 0.75]
    outcoupling_efficiencies = [1.0, 0.75, 0.5]
    detection_efficiencies = [1.0, 0.6, 0.34]

    test_cases = []
    for mean_photon, bsm_efficiency, outcoupling_efficiency, detection_efficiency in zip(mean_photons, bsm_efficiencies, outcoupling_efficiencies, detection_efficiencies):
        test_cases.append({
            "mean_photon": mean_photon,
            "bsm_efficiency": bsm_efficiency,
            "outcoupling_efficiency": outcoupling_efficiency,
            "detection_efficiency": detection_efficiency,
        })
    return test_cases

@pytest.fixture
def tmsv_test_case_rand() -> dict:
    """Return a random test case (dictionary of parameters)."""
    params = {
        "detection_efficiency": np.random.uniform(0.5, 1.0),
        "mean_photon": 10**np.random.uniform(-5, 1),
    }
    return params

@pytest.fixture
def spdc_test_case_rand() -> dict:
    """Return a random test case (dictionary of parameters)."""
    params = {
        "bsm_efficiency": np.random.uniform(0.5, 1.0),
        "outcoupling_efficiency": np.random.uniform(0.5, 1.0),
        "detection_efficiency": np.random.uniform(0.5, 1.0),
        "mean_photon": 10**np.random.uniform(-5, 1),
    }
    return params

@pytest.fixture
def zalm_test_case_rand() -> dict:
    """Return a random test case (dictionary of parameters)."""
    params = {
        "bsm_efficiency": np.random.uniform(0.5, 1.0),
        "outcoupling_efficiency": np.random.uniform(0.5, 1.0),
        "detection_efficiency": np.random.uniform(0.5, 1.0),
        "mean_photon": 10**np.random.uniform(-5, 1),
        "dark_counts": np.random.uniform(0.0, 0.1),
    }
    return params

@pytest.fixture
def sigsag_test_case_rand() -> dict:
    """Return a random test case (dictionary of parameters)."""
    params = {
        "mean_photon": 10**np.random.uniform(-5, 1),
        "bsm_efficiency": np.random.uniform(0.5, 1.0),
        "outcoupling_efficiency": np.random.uniform(0.5, 1.0),
        "detection_efficiency": np.random.uniform(0.5, 1.0),
    }
    return params

@pytest.fixture
def tmsv_py(tmsv_test_case_rand: dict) -> gqpy.TMSV:
    tmsv = gqpy.TMSV()
    tmsv.params.update(tmsv_test_case_rand)
    return tmsv

@pytest.fixture
def tmsv_jl(tmsv_test_case_rand: dict) -> gqjl.TMSV:
    import genqo as gqjl
    return gqjl.TMSV().set(**tmsv_test_case_rand)

@pytest.fixture
def spdc_py(spdc_test_case_rand: dict) -> gqpy.SPDC:
    spdc = gqpy.SPDC()
    spdc.params.update(spdc_test_case_rand)
    return spdc

@pytest.fixture
def spdc_jl(spdc_test_case_rand: dict) -> gqjl.SPDC:
    import genqo as gqjl
    return gqjl.SPDC().set(**spdc_test_case_rand)

@pytest.fixture
def zalm_py(zalm_test_case_rand: dict) -> gqpy.ZALM:
    zalm = gqpy.ZALM()
    zalm.params.update(zalm_test_case_rand)
    return zalm

@pytest.fixture
def zalm_jl(zalm_test_case_rand: dict) -> gqjl.ZALM:
    import genqo as gqjl
    return gqjl.ZALM().set(**zalm_test_case_rand)

@pytest.fixture
def sigsag_py(sigsag_test_case_rand: dict) -> gqpy.SIGSAG_BS:
    sigsag = gqpy.SIGSAG_BS()
    sigsag.params.update(sigsag_test_case_rand)
    return sigsag

@pytest.fixture
def sigsag_jl(sigsag_test_case_rand: dict) -> gqjl.SIGSAG:
    import genqo as gqjl
    return gqjl.SIGSAG().set(**sigsag_test_case_rand)

@pytest.fixture(scope="session")
def precision_table(request):
    rows = []
    yield rows
    bench_dir = request.config.getoption("--bench-dir")
    if bench_dir:
        os.makedirs(bench_dir, exist_ok=True)
        with open(os.path.join(bench_dir, "precision_raw.json"), "w") as f:
            json.dump(rows, f, indent=2)
