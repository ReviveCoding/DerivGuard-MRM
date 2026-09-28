import pandas as pd

from derivguard.data.adapters import read_cboe_marking_long


def test_cboe_marking_adapter_preserves_indicative_semantics(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "root": ["SPXW"],
            "expiry": ["2026-10-01"],
            "strike": [6000.0],
            "underlying_symbol": ["^SPX"],
            "call_message_time": ["2026-09-25 15:00:00-04:00"],
            "call_final_indicative_bid": [100.0],
            "call_final_indicative_ask": [101.0],
            "call_final_indicative_bid_size": [2],
            "call_final_indicative_ask_size": [3],
            "call_underlying_value": [6100.0],
            "call_osi_identifier": ["call-id"],
            "put_message_time": ["2026-09-25 15:00:00-04:00"],
            "put_final_indicative_bid": [1.0],
            "put_final_indicative_ask": [2.0],
            "put_final_indicative_bid_size": [4],
            "put_final_indicative_ask_size": [5],
            "put_underlying_value": [6100.0],
            "put_osi_identifier": ["put-id"],
        }
    )
    path = tmp_path / "marking.csv"
    source.to_csv(path, index=False)
    result = read_cboe_marking_long(path)
    assert len(result) == 2
    assert set(result["option_type"]) == {"call", "put"}
    assert set(result["observation_class"]) == {"CBOE_INDICATIVE_MARK_NOT_OPRA_NBBO"}
    assert result["quote_date"].astype(str).unique().tolist() == ["2026-09-25"]
