from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from derivguard.data.settlement import (
    classify_settlement,
    expiration_timestamp,
    year_fraction_to_expiry,
)


def test_settlement_classification_preserves_spx_spxw_distinction() -> None:
    assert classify_settlement("SPXW") == "PM"
    assert classify_settlement("^SPX") == "AM"
    assert classify_settlement("SPX", "PM") == "PM"
    assert classify_settlement("OTHER") == "AMBIGUOUS"


def test_same_day_pm_has_positive_time_after_am_settlement() -> None:
    eastern = ZoneInfo("America/New_York")
    valuation = datetime(2026, 9, 25, 12, 0, tzinfo=eastern)
    expiry = date(2026, 9, 25)
    assert year_fraction_to_expiry(valuation, expiry, "PM") > 0
    assert year_fraction_to_expiry(valuation, expiry, "AM") < 0
    assert expiration_timestamp(expiry, "PM").hour == 16


def test_naive_valuation_and_ambiguous_settlement_rejected() -> None:
    with pytest.raises(ValueError):
        year_fraction_to_expiry(datetime(2026, 1, 1), date(2026, 1, 2), "PM")
    with pytest.raises(ValueError):
        expiration_timestamp(date(2026, 1, 2), "AMBIGUOUS")
