from __future__ import annotations

import numpy as np

from derivguard.compute.precision import _torch_prices
from derivguard.development.heston_cf import HestonParameters, heston_price_fixed_quad


def test_benchmark_precision_kernel_matches_fp64_and_measures_fp32() -> None:
    strikes = np.linspace(80.0, 120.0, 9)
    parameters = HestonParameters(0.04, 1.5, 0.04, 0.5, -0.7)
    reference = np.asarray(
        heston_price_fixed_quad(100.0, strikes, 1.0, parameters, rate=0.015, nodes=128)
    )
    fp64 = _torch_prices(strikes, 1.0, parameters, device="cpu", precision="float64", nodes=128)
    fp32 = _torch_prices(strikes, 1.0, parameters, device="cpu", precision="float32", nodes=128)
    assert np.max(np.abs(fp64 - reference)) < 1.0e-9
    assert np.all(np.isfinite(fp32))
    assert np.max(np.abs(fp32 - reference)) < 1.0e-3
