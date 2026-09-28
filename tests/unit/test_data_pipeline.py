import pandas as pd

from derivguard.data.pipeline import run_audit_pipeline


def _raw_quotes() -> pd.DataFrame:
    rows = []
    for option_type in ("C", "P"):
        for strike in (90, 95, 100, 105, 110):
            mid = max(102 - strike, 0) + 2 if option_type == "C" else max(strike - 102, 0) + 2
            rows.append(
                {
                    "underlying": "SPXW",
                    "quote_date": "2026-09-01",
                    "expiration": "2026-10-01",
                    "strike": strike,
                    "type": option_type,
                    "bid": mid - 0.1,
                    "ask": mid + 0.1,
                }
            )
    rows.append(
        {
            "underlying": "SPXW",
            "quote_date": "2026-09-01",
            "expiration": "2026-10-01",
            "strike": -1,
            "type": "X",
            "bid": 2,
            "ask": 1,
        }
    )
    return pd.DataFrame(rows)


def test_pipeline_preserves_excluded_rows_and_source_identity() -> None:
    result = run_audit_pipeline(_raw_quotes(), source_id="test", source_file="quotes.csv")
    assert len(result.raw) == 11
    assert len(result.quality_flagged) == 11
    assert len(result.model_ready) == 10
    excluded = result.quality_flagged.query("inclusion_status == 'EXCLUDED'").iloc[0]
    assert len(excluded["source_row_id"]) == 64
    assert "INVALID_STRIKE" in excluded["exclusion_codes"]
    assert "ASK_BELOW_BID" in excluded["exclusion_codes"]
    assert excluded["repair_flag"] == False  # noqa: E712
    assert set(result.normalized["settlement_class"]) == {"PM"}


def test_static_arbitrage_and_sparse_diagnostics_are_non_destructive() -> None:
    raw = _raw_quotes().iloc[:4].copy()
    result = run_audit_pipeline(raw, source_id="test", source_file="small.csv")
    assert result.quality_summary["SPARSE_STRIKE_SLICE"] == 4
    assert len(result.quality_flagged) == len(raw)
    assert set(result.quality_flagged["source_row_id"]) >= set(result.filtered["source_row_id"])
