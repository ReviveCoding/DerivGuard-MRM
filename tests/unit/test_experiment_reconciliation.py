from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from derivguard.governance.experiment_reconciliation import reconcile_experiment_registry


def _write_registry(root: Path, families: dict[str, list[dict[str, object]]]) -> None:
    path = root / "experiments/registry.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "registry_status": "PLANNED",
                "split_policy": {},
                "validation_baselines": [],
                "families": families,
                "calibration_methods": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _read_registry(root: Path) -> dict[str, object]:
    value = yaml.safe_load((root / "experiments/registry.yaml").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_materiality_completion_requires_measured_value_and_greeks(tmp_path: Path) -> None:
    _write_registry(tmp_path, {"MATERIALITY": [{"id": "P01", "status": "PLANNED"}]})
    table = tmp_path / "reports/tables/T11_PORTFOLIO_MATERIALITY.csv"
    table.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "portfolio_id": ["P01", "P01", "P01"],
            "model_id": ["M10", "M03", "M21"],
            "value": [1.0, 1.1, 0.9],
            "delta": [0.1, 0.2, 0.0],
            "gamma": [0.01, 0.02, 0.015],
            "vega": [2.0, 2.1, 1.9],
            "valuation_materiality": [0.0, 0.1, 0.1],
            "delta_materiality": [0.0, 0.1, 0.1],
            "gamma_materiality": [0.0, 0.01, 0.005],
            "vega_materiality": [0.0, 0.1, 0.1],
        }
    ).to_csv(table, index=False)

    manifest = reconcile_experiment_registry(tmp_path)
    registry = _read_registry(tmp_path)
    entry = registry["families"]["MATERIALITY"][0]  # type: ignore[index]
    assert entry["status"] == "COMPLETED"
    assert entry["evidence_predicates"]["value_and_greeks_measured"] is True
    assert manifest["summary"]["COMPLETED"] == 1


def test_pde_attempt_remains_partial_when_frozen_qualification_is_partial(
    tmp_path: Path,
) -> None:
    _write_registry(tmp_path, {"NUMERICAL": [{"id": "N07", "status": "PLANNED"}]})
    output = tmp_path / "results/aggregated"
    output.mkdir(parents=True)
    pd.DataFrame(
        {
            "scheme": ["modified_craig_sneyd", "hundsdorfer_verwer"] * 4,
            "study": [
                "spot_grid",
                "spot_grid",
                "variance_grid",
                "variance_grid",
                "time_grid",
                "time_grid",
                "domain_boundary",
                "domain_boundary",
            ],
        }
    ).to_csv(output / "pde_adi_convergence.csv", index=False)
    (output / "numerical_remediation_qualification.json").write_text(
        json.dumps({"pde_status": "PARTIAL"}), encoding="utf-8"
    )

    reconcile_experiment_registry(tmp_path)
    registry = _read_registry(tmp_path)
    entry = registry["families"]["NUMERICAL"][0]  # type: ignore[index]
    assert entry["status"] == "PARTIAL"
    assert entry["evidence_predicates"]["mcs_and_hv_attempted"] is True
    assert entry["evidence_predicates"]["frozen_pde_qualification_passed"] is False


def test_temporal_outcome_uses_recorded_external_data_blocker(tmp_path: Path) -> None:
    _write_registry(tmp_path, {"REAL_OUTCOMES": [{"id": "R02", "status": "PLANNED"}]})
    path = tmp_path / "results/RESULT_AVAILABILITY.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "phase_id": "31",
                        "status": "UNAVAILABLE",
                        "attempted": True,
                        "blocker": "Only one synchronized prospective date is available.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    reconcile_experiment_registry(tmp_path)
    registry = _read_registry(tmp_path)
    entry = registry["families"]["REAL_OUTCOMES"][0]  # type: ignore[index]
    assert entry["status"] == "UNAVAILABLE"
    assert entry["feasibility"] == "EXTERNAL_DATA_BLOCKER"
    assert "one synchronized prospective date" in entry["status_reason"]


def test_missing_precision_artifact_cannot_complete(tmp_path: Path) -> None:
    _write_registry(tmp_path, {"NUMERICAL": [{"id": "N10", "status": "COMPLETED"}]})

    reconcile_experiment_registry(tmp_path)
    registry = _read_registry(tmp_path)
    entry = registry["families"]["NUMERICAL"][0]  # type: ignore[index]
    assert entry["status"] == "PARTIAL"
    assert entry["feasibility"] == "NO_EXECUTION_EVIDENCE"


def test_variance_reduction_completes_only_from_measured_variants(tmp_path: Path) -> None:
    _write_registry(tmp_path, {"NUMERICAL": [{"id": "N06", "status": "PLANNED"}]})
    output = tmp_path / "results/aggregated"
    output.mkdir(parents=True)
    rows = [
        {
            "method": method,
            "antithetic": antithetic,
            "control_variate": control,
            "seed": seed,
            "standard_error": 0.1,
        }
        for seed in range(5)
        for method, antithetic, control in (
            ("none", False, False),
            ("antithetic", True, False),
            ("antithetic_control", True, True),
        )
    ]
    pd.DataFrame(rows).to_csv(output / "mc_variance_reduction.csv", index=False)
    (output / "mc_variance_reduction.json").write_text(
        json.dumps({"status": "MEASURED", "experiment_id": "N06"}), encoding="utf-8"
    )

    reconcile_experiment_registry(tmp_path)
    registry = _read_registry(tmp_path)
    entry = registry["families"]["NUMERICAL"][0]  # type: ignore[index]
    assert entry["status"] == "COMPLETED"


def test_no_bootstrap_ablation_requires_quantitative_summary(tmp_path: Path) -> None:
    _write_registry(
        tmp_path,
        {"DEVELOPER_METHOD_ABLATION": [{"id": "DA04", "status": "PLANNED"}]},
    )
    output = tmp_path / "results/aggregated"
    output.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ablation_id": "DA04",
                "metric": "mean parameter standard deviation",
                "selected_value": 0.02,
                "ablated_value": 0.0,
                "difference_ablated_minus_selected": -0.02,
            }
        ]
    ).to_csv(output / "developer_method_ablation_summary.csv", index=False)

    reconcile_experiment_registry(tmp_path)
    registry = _read_registry(tmp_path)
    entry = registry["families"]["DEVELOPER_METHOD_ABLATION"][0]  # type: ignore[index]
    assert entry["status"] == "COMPLETED"
    assert entry["evidence_predicates"]["quantitative_result_present"] is True
