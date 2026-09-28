"""Settlement classification and timestamp-aware time-to-expiry handling."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Literal
from zoneinfo import ZoneInfo

SettlementClass = Literal["AM", "PM", "AMBIGUOUS"]
NEW_YORK = ZoneInfo("America/New_York")


def classify_settlement(root: str | None, settlement_time: str | None = None) -> SettlementClass:
    """Classify SPX settlement without silently conflating SPX and SPXW.

    An explicit AM/PM field wins. SPXW is PM settled. Bare SPX is treated as
    AM only when no contradictory explicit field exists; unknown roots remain
    ambiguous rather than guessed.
    """
    explicit = (settlement_time or "").strip().upper()
    if explicit in {"AM", "A.M.", "MORNING"}:
        return "AM"
    if explicit in {"PM", "P.M.", "AFTERNOON"}:
        return "PM"
    normalized = (root or "").strip().upper().lstrip("^")
    if normalized == "SPXW":
        return "PM"
    if normalized == "SPX":
        return "AM"
    return "AMBIGUOUS"


def expiration_timestamp(
    expiration: date,
    settlement: SettlementClass,
    *,
    timezone: ZoneInfo = NEW_YORK,
) -> datetime:
    if settlement == "AMBIGUOUS":
        raise ValueError("cannot assign expiry timestamp to ambiguous settlement")
    # AM settlement value is based on opening component prices; 09:30 is an
    # explicit public-data approximation, not an assertion of a tradable print.
    clock = time(9, 30) if settlement == "AM" else time(16, 0)
    return datetime.combine(expiration, clock, tzinfo=timezone)


def year_fraction_to_expiry(
    valuation: datetime,
    expiration: date,
    settlement: SettlementClass,
    *,
    basis_days: float = 365.0,
) -> float:
    if valuation.tzinfo is None:
        raise ValueError("valuation timestamp must be timezone-aware")
    if basis_days <= 0:
        raise ValueError("basis_days must be positive")
    expiry = expiration_timestamp(expiration, settlement)
    seconds = (
        expiry.astimezone(ZoneInfo("UTC")) - valuation.astimezone(ZoneInfo("UTC"))
    ).total_seconds()
    return seconds / (basis_days * 86400.0)
