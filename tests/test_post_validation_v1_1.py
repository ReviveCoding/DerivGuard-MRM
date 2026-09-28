from pathlib import Path

import numpy as np
import pandas as pd

from derivguard.post_validation_v1_1 import (
    AnalyticsConfig,
    apply_slices,
    freeze_slice_registry,
    model_winner_tables,
)
from derivguard.synthetic.dve_v1_1 import DVEV11Config, add_scores, build_seed_registry


def _minimal_contract_rows() -> pd.DataFrame:
    records = []
    for index in range(12):
        records.append(
            {
                "quote_date": f"2022-07-{index + 1:02d}",
                "split": "DEV",
                "model_id": "M10_HESTON",
                "source_row_id": str(index),
                "log_moneyness": -0.20 + 0.04 * index,
                "dte": 8 + index,
                "VIXCLS": 15.0 + index,
                "relative_spread": 0.01 + 0.01 * index,
                "volume": float(index),
                "open_interest": float(index * 10),
            }
        )
    return pd.DataFrame(records)


def test_slice_registry_is_dev_frozen_and_keeps_zero_dte() -> None:
    rows = _minimal_contract_rows()
    test_root = Path("build/v1_1_test_slice_registry")
    test_root.mkdir(parents=True, exist_ok=True)
    frozen = freeze_slice_registry(test_root, rows, AnalyticsConfig(bootstrap_draws=10))
    assert frozen["status"] == "FROZEN_FROM_DEV_BEFORE_VALIDATION_OR_LOCKED_SLICE_EVALUATION"
    assert "0DTE" in frozen["maturity_labels"]
    sliced = apply_slices(rows, frozen)
    assert set(sliced["vix_regime"]) <= {"LOW", "MEDIUM", "HIGH"}


def test_fresh_dve_seeds_do_not_overlap_preserved_v1_regions() -> None:
    seeds = build_seed_registry(DVEV11Config())
    all_seeds = (*seeds.dev, *seeds.validation, *seeds.locked, *seeds.ood)
    assert len(all_seeds) == len(set(all_seeds)) == 102
    assert min(all_seeds) >= 117_000_001
    assert max(all_seeds) < 121_000_000


def test_transparent_scores_are_deterministic() -> None:
    frame = pd.DataFrame(
        [
            {
                **{
                    name: float(index + 1)
                    for index, name in enumerate(
                        (
                            "E_data",
                            "E_num",
                            "E_cal",
                            "E_ident",
                            "E_param",
                            "E_form",
                            "E_greek",
                            "E_extra",
                            "E_outcome",
                        )
                    )
                },
                "fit_error": 0.5,
            }
        ]
    )
    scales = {name: 1.0 for name in frame.columns if name.startswith("E_")}
    scored = add_scores(frame, scales, "top_2_evidence_mean")
    assert np.isclose(scored.loc[0, "maximum_evidence_percentile"], 9.0)
    assert np.isclose(scored.loc[0, "top_2_evidence_mean"], 8.5)
    assert np.isclose(scored.loc[0, "FullV1_1DVE"], 8.5)


def test_winner_map_excludes_unqualified_bates_sensitivity() -> None:
    rows = []
    for date_index in range(3):
        for row_index in range(4):
            for model, error in (
                ("M00_BS_FLAT", 2.0),
                ("M02_SVI", 1.0),
                ("M03_SSVI", 1.5),
                ("M10_HESTON", 1.2),
                ("M21_BATES", 0.01),
            ):
                rows.append(
                    {
                        "quote_date": f"2022-07-{date_index + 1:02d}",
                        "split": "DEV",
                        "moneyness_slice": "atm",
                        "maturity_slice": "8-30D",
                        "model_id": model,
                        "residual": error,
                        "normalized_error": error + 0.01 * row_index,
                    }
                )
    winner, pairwise, date_win = model_winner_tables(
        pd.DataFrame(rows), AnalyticsConfig(bootstrap_draws=20)
    )
    assert winner.loc[0, "winner"] == "M02_SVI"
    assert "M21_BATES" not in set(pairwise["model"]) | set(pairwise["opponent"])
    assert "M21_BATES" not in set(date_win["model_id"])
    assert 0.0 <= winner.loc[0, "winner_bootstrap_probability"] <= 1.0
