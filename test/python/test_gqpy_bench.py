"""Benchmarks for Python genqo."""

import numpy as np
import genqo_old as gqpy


# TMSV benchmarks

def test_tmsv__probability_success(tmsv_py: gqpy.TMSV, benchmark):
    benchmark(lambda: tmsv_py.run() and tmsv_py.calculate_probability_success())

def test_tmsv__covariance_matrix(tmsv_py: gqpy.TMSV, benchmark):
    benchmark(tmsv_py.calculate_covariance_matrix)


# SPDC benchmarks

def test_spdc__fidelity(spdc_py: gqpy.SPDC, benchmark):
    benchmark(lambda: spdc_py.run() and spdc_py.calculate_fidelity())

def test_spdc__spin_density_matrix(spdc_py: gqpy.SPDC, benchmark):
    benchmark(spdc_py.calculate_density_operator, np.array([1,0,1,1,0,0,1,0]))

def test_spdc__covariance_matrix(spdc_py: gqpy.SPDC, benchmark):
    benchmark(spdc_py.calculate_covariance_matrix)


# ZALM benchmarks

def test_zalm__probability_success(zalm_py: gqpy.ZALM, benchmark):
    benchmark(lambda: zalm_py.run() and zalm_py.calculate_probability_success())

def test_zalm__fidelity(zalm_py: gqpy.ZALM, benchmark):
    benchmark(lambda: zalm_py.run() and zalm_py.calculate_fidelity())

def test_zalm__spin_density_matrix(zalm_py: gqpy.ZALM, benchmark):
    benchmark(zalm_py.calculate_density_operator, np.array([1,0,1,1,0,0,1,0]))

def test_zalm__covariance_matrix(zalm_py: gqpy.ZALM, benchmark):
    benchmark(zalm_py.calculate_covariance_matrix)


# SIGSAG benchmarks

def test_sigsag__covariance_matrix(sigsag_py: gqpy.SIGSAG_BS, benchmark):
    benchmark(sigsag_py.calculate_covariance_matrix)


def test_sigsag__probability_success(sigsag_py: gqpy.SIGSAG_BS, benchmark):
    benchmark(lambda: sigsag_py.run() and sigsag_py.calculate_probability_success())

def test_sigsag__fidelity(sigsag_py: gqpy.SIGSAG_BS, benchmark):
    benchmark(lambda: sigsag_py.run() and sigsag_py.calculate_fidelity())


# Other benchmarks


def test_linsweep_1d(tmsv_py: gqpy.TMSV, benchmark):
    def linsweep_1d():
        probability_success = []
        for mp in np.linspace(1e-4, 1e-2, 100):
            tmsv_py.params["mean_photon"] = mp
            tmsv_py.run()
            tmsv_py.calculate_probability_success()
            probability_success.append(tmsv_py.results["probability_success"])
        return probability_success
    benchmark(linsweep_1d)

def test_linsweep_2d(tmsv_py: gqpy.TMSV, benchmark):
    def linsweep_2d():
        probability_success = np.zeros((100, 5))
        for i, eff in enumerate([0.2, 0.5, 0.6, 0.7, 0.9]):
            tmsv_py.params["detection_efficiency"] = eff
            for mp in np.linspace(1e-4, 1e-2, 100):
                tmsv_py.params["mean_photon"] = mp
                tmsv_py.run()
                tmsv_py.calculate_probability_success()
                probability_success[:, i] = tmsv_py.results["probability_success"]
        return probability_success
    benchmark(linsweep_2d)

def test_logsweep_1d(tmsv_py: gqpy.TMSV, benchmark):
    def logsweep_1d():
        probability_success = []
        for mp in np.logspace(-4, -2, 100):
            tmsv_py.params["mean_photon"] = mp
            tmsv_py.run()
            tmsv_py.calculate_probability_success()
            probability_success.append(tmsv_py.results["probability_success"])
        return probability_success
    benchmark(logsweep_1d)
