import pytest

from derivguard.materiality.portfolio import (
    InstrumentMeasures,
    compare_materiality,
    delta_neutralize,
    evaluate_models,
    evidence_materiality_quadrant,
    standardized_portfolios,
    vega_normalize,
)


def _evaluator(scale: float):
    def evaluate(leg):
        sign = 1.0 if leg.option_type == "call" else -1.0
        return InstrumentMeasures(
            scale * (10.0 + 0.01 * leg.strike),
            scale * 0.5 * sign,
            scale * 0.02,
            scale * (5.0 + 0.1 * leg.maturity),
        )

    return evaluate


def test_standardized_portfolios_and_model_evaluation() -> None:
    portfolios = standardized_portfolios(100.0, 0.25, 0.75)
    assert [portfolio.portfolio_id for portfolio in portfolios] == [
        "P01",
        "P02",
        "P03",
        "P04",
        "P05",
        "P06",
    ]
    results = evaluate_models(
        portfolios, {"developer": _evaluator(1.0), "challenger": _evaluator(1.1)}
    )
    assert len(results) == 12
    comparison = compare_materiality(results[0], results[1], value_exposure=100.0)
    assert comparison.valuation_materiality > 0.0
    assert comparison.delta_materiality >= 0.0


def test_vega_normalization_and_evidence_materiality_matrix() -> None:
    portfolio = standardized_portfolios(100.0, 0.25, 0.75)[0]
    normalized = vega_normalize(portfolio, _evaluator(1.0), target_absolute_vega=10.0)
    result = evaluate_models([normalized], {"model": _evaluator(1.0)})[0]
    assert abs(result.vega) == pytest.approx(10.0)
    directional = standardized_portfolios(100.0, 0.25, 0.75)[1]
    neutral = delta_neutralize(directional, _evaluator(1.0), hedge_leg_index=0)
    neutral_result = evaluate_models([neutral], {"model": _evaluator(1.0)})[0]
    assert neutral_result.delta == pytest.approx(0.0, abs=1.0e-14)
    assert (
        evidence_materiality_quadrant(2.0, 0.2, evidence_threshold=1.0, materiality_threshold=0.5)
        == "high evidence / low materiality"
    )
