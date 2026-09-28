import importlib.util

import numpy as np
import pytest

from derivguard.development.heston_cf import (
    HestonParameters,
    heston_call_price_torch,
    heston_price_fixed_quad,
)
from derivguard.validation.heston_mc import (
    ValidationHestonParameters,
    heston_mc_numpy,
    heston_mc_torch,
)

torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.gpu


@pytest.mark.skipif(torch_spec is None, reason="Torch unavailable")
def test_torch_fp64_backend_agrees_with_numpy() -> None:
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip(f"Torch installation is not importable: {exc}")

    if not torch.cuda.is_available():
        pytest.skip("qualified CUDA device unavailable")
    strikes = torch.tensor([80.0, 100.0, 120.0], dtype=torch.float64, device="cuda")
    parameters = torch.tensor(
        [[0.04, 2.0, 0.04, 0.5, -0.7], [0.06, 1.2, 0.05, 0.7, -0.3]],
        dtype=torch.float64,
        device="cuda",
    )
    gpu = heston_call_price_torch(100.0, strikes, 1.0, parameters, 0.03, 0.01)
    assert gpu.dtype == torch.float64
    for row, raw in enumerate(parameters.cpu().numpy()):
        p = HestonParameters(*map(float, raw))
        cpu = np.asarray(heston_price_fixed_quad(100.0, strikes.cpu().numpy(), 1.0, p, 0.03, 0.01))
        assert gpu[row].cpu().numpy() == pytest.approx(cpu, abs=5.0e-7, rel=5.0e-7)


@pytest.mark.skipif(torch_spec is None, reason="Torch unavailable")
@pytest.mark.parametrize("scheme", ["full_truncation", "qe"])
def test_gpu_mc_seed_is_reproducible(scheme: str) -> None:
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip(f"Torch installation is not importable: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("qualified CUDA device unavailable")
    params = ValidationHestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    kwargs = dict(
        paths=2_048,
        steps=8,
        scheme=scheme,
        seed=1234,
        antithetic=True,
        chunk_paths=2_048,
    )
    first = heston_mc_torch(100.0, 100.0, 0.5, params, **kwargs)  # type: ignore[arg-type]
    second = heston_mc_torch(100.0, 100.0, 0.5, params, **kwargs)  # type: ignore[arg-type]
    assert first.price == second.price
    assert first.standard_error == second.standard_error
    assert first.oom_retries == 0


@pytest.mark.skipif(torch_spec is None, reason="Torch unavailable")
def test_qem_cpu_gpu_distributional_agreement_and_gpu_parity() -> None:
    """Check QE-M across independent CPU/GPU streams and its martingale parity."""

    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip(f"Torch installation is not importable: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("qualified CUDA device unavailable")
    parameters = ValidationHestonParameters(0.04, 2.0, 0.04, 0.8, -0.7)
    common = dict(
        paths=40_000,
        steps=32,
        scheme="qe_m",
        seed=2718,
        antithetic=True,
        control_variate=False,
    )
    cpu = heston_mc_numpy(100.0, 100.0, 1.0, parameters, 0.03, 0.01, **common)
    gpu_call = heston_mc_torch(
        100.0,
        100.0,
        1.0,
        parameters,
        0.03,
        0.01,
        chunk_paths=40_000,
        **common,
    )
    combined_se = np.hypot(cpu.standard_error, gpu_call.standard_error)
    assert abs(cpu.price - gpu_call.price) <= 4.0 * combined_se + 0.01

    gpu_put = heston_mc_torch(
        100.0,
        100.0,
        1.0,
        parameters,
        0.03,
        0.01,
        option_type="put",
        chunk_paths=40_000,
        **common,
    )
    parity_target = 100.0 * np.exp(-0.01) - 100.0 * np.exp(-0.03)
    parity_se = np.hypot(gpu_call.standard_error, gpu_put.standard_error)
    assert abs((gpu_call.price - gpu_put.price) - parity_target) <= 4.0 * parity_se + 0.01
