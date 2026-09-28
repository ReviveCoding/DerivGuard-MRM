"""Standardized option portfolios and separate economic materiality measures."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionLeg:
    strike: float
    maturity: float
    option_type: OptionType
    quantity: float

    def __post_init__(self) -> None:
        if self.strike <= 0.0 or self.maturity <= 0.0:
            raise ValueError("leg strike and maturity must be positive")
        if self.option_type not in {"call", "put"} or not np.isfinite(self.quantity):
            raise ValueError("invalid option leg")


@dataclass(frozen=True)
class StandardizedPortfolio:
    portfolio_id: str
    description: str
    legs: tuple[OptionLeg, ...]
    normalization: str


@dataclass(frozen=True)
class InstrumentMeasures:
    value: float
    delta: float
    gamma: float
    vega: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite((self.value, self.delta, self.gamma, self.vega))):
            raise ValueError("instrument measures must be finite")


@dataclass(frozen=True)
class PortfolioMeasures:
    model_id: str
    portfolio_id: str
    value: float
    delta: float
    gamma: float
    vega: float


@dataclass(frozen=True)
class MaterialityComparison:
    developer_model_id: str
    challenger_model_id: str
    portfolio_id: str
    valuation_materiality: float
    delta_materiality: float
    gamma_materiality: float
    vega_materiality: float


def standardized_portfolios(
    spot: float,
    front_maturity: float,
    back_maturity: float,
    *,
    wing_fraction: float = 0.10,
) -> tuple[StandardizedPortfolio, ...]:
    """Construct the six pre-specified portfolio shapes with unit notionals.

    These are shape definitions.  A caller may subsequently vega-normalize or
    delta-neutralize using DEV model Greeks, but that transformation must be
    recorded separately to avoid using locked-test challenger information.
    """

    if spot <= 0.0 or front_maturity <= 0.0 or back_maturity <= front_maturity:
        raise ValueError("require positive spot and ordered positive maturities")
    if not 0.0 < wing_fraction < 0.5:
        raise ValueError("wing_fraction must lie in (0, 0.5)")
    low = spot * (1.0 - wing_fraction)
    high = spot * (1.0 + wing_fraction)
    atm = spot
    return (
        StandardizedPortfolio(
            "P01",
            "ATM straddle",
            (
                OptionLeg(atm, front_maturity, "call", 1.0),
                OptionLeg(atm, front_maturity, "put", 1.0),
            ),
            "unit legs",
        ),
        StandardizedPortfolio(
            "P02",
            "risk reversal",
            (
                OptionLeg(high, front_maturity, "call", 1.0),
                OptionLeg(low, front_maturity, "put", -1.0),
            ),
            "unit long call / short put",
        ),
        StandardizedPortfolio(
            "P03",
            "butterfly",
            (
                OptionLeg(low, front_maturity, "call", 1.0),
                OptionLeg(atm, front_maturity, "call", -2.0),
                OptionLeg(high, front_maturity, "call", 1.0),
            ),
            "1:-2:1 call weights",
        ),
        StandardizedPortfolio(
            "P04",
            "calendar spread",
            (
                OptionLeg(atm, back_maturity, "call", 1.0),
                OptionLeg(atm, front_maturity, "call", -1.0),
            ),
            "long back / short front",
        ),
        StandardizedPortfolio(
            "P05",
            "skew portfolio",
            (
                OptionLeg(low, front_maturity, "put", 1.0),
                OptionLeg(atm, front_maturity, "put", -1.0),
                OptionLeg(atm, front_maturity, "call", -1.0),
                OptionLeg(high, front_maturity, "call", 1.0),
            ),
            "unit wing / short ATM legs",
        ),
        StandardizedPortfolio(
            "P06",
            "mixed standardized book",
            (
                OptionLeg(low, front_maturity, "put", 0.75),
                OptionLeg(atm, front_maturity, "call", 1.0),
                OptionLeg(high, front_maturity, "call", -0.5),
                OptionLeg(atm, back_maturity, "put", -0.5),
            ),
            "fixed pre-specified unit weights",
        ),
    )


def evaluate_portfolio(
    portfolio: StandardizedPortfolio,
    model_id: str,
    evaluator: Callable[[OptionLeg], InstrumentMeasures],
) -> PortfolioMeasures:
    value = delta = gamma = vega = 0.0
    for leg in portfolio.legs:
        measures = evaluator(leg)
        value += leg.quantity * measures.value
        delta += leg.quantity * measures.delta
        gamma += leg.quantity * measures.gamma
        vega += leg.quantity * measures.vega
    return PortfolioMeasures(model_id, portfolio.portfolio_id, value, delta, gamma, vega)


def vega_normalize(
    portfolio: StandardizedPortfolio,
    evaluator: Callable[[OptionLeg], InstrumentMeasures],
    *,
    target_absolute_vega: float = 1.0,
) -> StandardizedPortfolio:
    """Scale all legs using one declared reference model's portfolio vega."""

    if target_absolute_vega <= 0.0:
        raise ValueError("target_absolute_vega must be positive")
    reference = evaluate_portfolio(portfolio, "normalization_reference", evaluator)
    if abs(reference.vega) <= 1.0e-12:
        raise ValueError("portfolio has insufficient reference vega to normalize")
    scale = target_absolute_vega / abs(reference.vega)
    return StandardizedPortfolio(
        portfolio.portfolio_id,
        portfolio.description,
        tuple(
            OptionLeg(leg.strike, leg.maturity, leg.option_type, leg.quantity * scale)
            for leg in portfolio.legs
        ),
        f"vega normalized to {target_absolute_vega:g} using declared reference model",
    )


def delta_neutralize(
    portfolio: StandardizedPortfolio,
    evaluator: Callable[[OptionLeg], InstrumentMeasures],
    *,
    hedge_leg_index: int = 0,
) -> StandardizedPortfolio:
    """Adjust one declared option leg to zero reference-model portfolio delta.

    The reference evaluator must be selected on DEV.  This function does not
    inspect challenger or locked-test results, preventing normalization leakage.
    """

    if not 0 <= hedge_leg_index < len(portfolio.legs):
        raise ValueError("hedge_leg_index is outside the portfolio")
    reference = evaluate_portfolio(portfolio, "normalization_reference", evaluator)
    hedge_leg = portfolio.legs[hedge_leg_index]
    unit_hedge = OptionLeg(hedge_leg.strike, hedge_leg.maturity, hedge_leg.option_type, 1.0)
    unit_delta = evaluator(unit_hedge).delta
    if abs(unit_delta) <= 1.0e-12:
        raise ValueError("selected hedge leg has insufficient reference delta")
    quantities = [leg.quantity for leg in portfolio.legs]
    quantities[hedge_leg_index] -= reference.delta / unit_delta
    legs = tuple(
        OptionLeg(leg.strike, leg.maturity, leg.option_type, quantity)
        for leg, quantity in zip(portfolio.legs, quantities, strict=True)
    )
    return StandardizedPortfolio(
        portfolio.portfolio_id,
        portfolio.description,
        legs,
        f"delta neutralized using declared reference model and leg {hedge_leg_index}",
    )


def compare_materiality(
    developer: PortfolioMeasures,
    challenger: PortfolioMeasures,
    *,
    value_exposure: float = 1.0,
    delta_exposure: float = 1.0,
    gamma_exposure: float = 1.0,
    vega_exposure: float = 1.0,
) -> MaterialityComparison:
    """Keep valuation and Greek materiality as separate non-negative outputs."""

    if developer.portfolio_id != challenger.portfolio_id:
        raise ValueError("portfolio IDs must match")
    exposures = np.asarray(
        (value_exposure, delta_exposure, gamma_exposure, vega_exposure), dtype=np.float64
    )
    if np.any(~np.isfinite(exposures)) or np.any(exposures < 0.0):
        raise ValueError("materiality exposures must be finite and non-negative")
    differences = np.abs(
        np.asarray(
            (
                developer.value - challenger.value,
                developer.delta - challenger.delta,
                developer.gamma - challenger.gamma,
                developer.vega - challenger.vega,
            )
        )
    )
    scaled = differences * exposures
    return MaterialityComparison(
        developer.model_id,
        challenger.model_id,
        developer.portfolio_id,
        *map(float, scaled),
    )


def evidence_materiality_quadrant(
    risk_evidence: float,
    materiality: float,
    *,
    evidence_threshold: float,
    materiality_threshold: float,
) -> str:
    """Classify without collapsing risk evidence and economic materiality."""

    values = (risk_evidence, materiality, evidence_threshold, materiality_threshold)
    if not np.all(np.isfinite(values)) or evidence_threshold < 0.0 or materiality_threshold < 0.0:
        raise ValueError("quadrant inputs must be finite with non-negative thresholds")
    evidence_label = "high evidence" if risk_evidence >= evidence_threshold else "low evidence"
    materiality_label = (
        "high materiality" if materiality >= materiality_threshold else "low materiality"
    )
    return f"{evidence_label} / {materiality_label}"


def evaluate_models(
    portfolios: Sequence[StandardizedPortfolio],
    evaluators: Mapping[str, Callable[[OptionLeg], InstrumentMeasures]],
) -> tuple[PortfolioMeasures, ...]:
    if not portfolios or not evaluators:
        raise ValueError("at least one portfolio and model evaluator are required")
    return tuple(
        evaluate_portfolio(portfolio, model_id, evaluator)
        for portfolio in portfolios
        for model_id, evaluator in evaluators.items()
    )
