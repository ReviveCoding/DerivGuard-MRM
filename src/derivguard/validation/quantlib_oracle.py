"""QuantLib Heston oracle adapter with explicit convention metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QuantLibHestonParameters:
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float


@dataclass(frozen=True)
class QuantLibOracleResult:
    price: float
    requested_maturity: float
    effective_maturity: float
    maturity_days: int
    quantlib_version: str
    engine: str


def quantlib_heston_price(
    spot: float,
    strike: float,
    maturity: float,
    params: QuantLibHestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    *,
    integration_order: int = 144,
) -> QuantLibOracleResult:
    """Price via QuantLib's analytic Heston implementation.

    QuantLib instruments use calendar dates.  The requested year fraction is
    therefore rounded to a whole number of Actual/365 days, and both requested
    and effective maturities are returned to prevent silent convention drift.
    """

    try:
        import QuantLib as ql  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("QuantLib is required for the external oracle") from exc
    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike, and maturity must be positive")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    if params.v0 < 0.0 or params.kappa <= 0.0 or params.theta < 0.0 or params.sigma_v <= 0.0:
        raise ValueError("invalid Heston parameters")
    if abs(params.rho) > 1.0:
        raise ValueError("rho must lie in [-1,1]")

    evaluation_date = ql.Date(15, ql.January, 2020)
    ql.Settings.instance().evaluationDate = evaluation_date
    days = max(1, round(365.0 * maturity))
    expiry = evaluation_date + days
    day_count = ql.Actual365Fixed()
    risk_free = ql.YieldTermStructureHandle(
        ql.FlatForward(evaluation_date, rate, day_count, ql.Continuous)
    )
    dividend = ql.YieldTermStructureHandle(
        ql.FlatForward(evaluation_date, dividend_yield, day_count, ql.Continuous)
    )
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    process = ql.HestonProcess(
        risk_free,
        dividend,
        spot_handle,
        params.v0,
        params.kappa,
        params.theta,
        params.sigma_v,
        params.rho,
    )
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model, integration_order)
    payoff_type = ql.Option.Call if option_type == "call" else ql.Option.Put
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(payoff_type, strike), ql.EuropeanExercise(expiry)
    )
    option.setPricingEngine(engine)
    return QuantLibOracleResult(
        price=float(option.NPV()),
        requested_maturity=maturity,
        effective_maturity=days / 365.0,
        maturity_days=days,
        quantlib_version=str(ql.__version__),
        engine="QuantLib::AnalyticHestonEngine",
    )
