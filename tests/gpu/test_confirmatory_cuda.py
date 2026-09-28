import numpy as np
import pytest

from derivguard.synthetic.confirmatory import (
    ConfirmatoryConfig,
    _cuda_family_values,
    _surface_design,
    qualify_confirmatory_engines,
)


@pytest.mark.gpu
def test_all_confirmatory_cuda_truth_families_pass_fp64_spot_checks() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("qualified CUDA device unavailable")
    checks = qualify_confirmatory_engines("cuda")
    assert all(check.status == "PASS" for check in checks)
    design = _surface_design(88_000_002, 4, ood=False)
    config = ConfirmatoryConfig(
        cases_dev=1,
        cases_validation=1,
        cases_locked=1,
        cases_ood=1,
        contracts_per_case=4,
        heston_nodes=32,
        bates_nodes=32,
        device="cuda",
    )
    for family in ("HESTON", "BATES", "LOCALVOL_SSVI", "PIECEWISE_REGIME"):
        values = _cuda_family_values(design, family, config, 88_000_002)  # type: ignore[arg-type]
        assert all(np.all(np.isfinite(value)) for value in values)
