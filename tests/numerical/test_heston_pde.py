import pytest

from derivguard.development.heston_cf import HestonParameters, heston_price_adaptive
from derivguard.validation.heston_pde import (
    PDEGrid,
    PDEHestonParameters,
    heston_pde_price,
)


@pytest.mark.numerical
def test_pde_spatial_refinement_moves_toward_independent_cf() -> None:
    pde_params = PDEHestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    reference = float(
        heston_price_adaptive(
            100.0, 100.0, 1.0, HestonParameters(0.04, 2.0, 0.04, 0.5, -0.7), 0.03, 0.01
        )
    )
    coarse = heston_pde_price(
        100.0,
        100.0,
        1.0,
        pde_params,
        0.03,
        0.01,
        grid=PDEGrid(spot_nodes=81, variance_nodes=41, time_steps=100, variance_max=0.5),
    )
    fine = heston_pde_price(
        100.0,
        100.0,
        1.0,
        pde_params,
        0.03,
        0.01,
        grid=PDEGrid(spot_nodes=161, variance_nodes=81, time_steps=200, variance_max=0.5),
    )
    assert abs(fine.price - reference) < abs(coarse.price - reference)
    assert abs(fine.price - reference) < 0.03
    assert fine.minimum_grid_value > -5.0e-5


@pytest.mark.numerical
def test_two_time_schemes_are_consistent_on_bounded_grid() -> None:
    p = PDEHestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    grid = PDEGrid(spot_nodes=121, variance_nodes=61, time_steps=160, variance_max=0.5)
    cn = heston_pde_price(100.0, 100.0, 1.0, p, 0.03, 0.01, scheme="crank_nicolson", grid=grid)
    implicit = heston_pde_price(
        100.0, 100.0, 1.0, p, 0.03, 0.01, scheme="implicit_euler", grid=grid
    )
    assert cn.price == pytest.approx(implicit.price, abs=0.02)


@pytest.mark.numerical
def test_nonuniform_mcs_and_hv_are_independently_consistent() -> None:
    p = PDEHestonParameters(0.04, 2.0, 0.04, 0.5, -0.7)
    reference = float(
        heston_price_adaptive(
            100.0, 100.0, 1.0, HestonParameters(0.04, 2.0, 0.04, 0.5, -0.7), 0.03, 0.01
        )
    )
    grid = PDEGrid(
        spot_nodes=81,
        variance_nodes=41,
        time_steps=120,
        variance_max=0.5,
        nonuniform=True,
        rannacher_steps=2,
    )
    mcs = heston_pde_price(
        100.0, 100.0, 1.0, p, 0.03, 0.01, scheme="modified_craig_sneyd", grid=grid
    )
    hv = heston_pde_price(100.0, 100.0, 1.0, p, 0.03, 0.01, scheme="hundsdorfer_verwer", grid=grid)
    assert abs(mcs.price - reference) < 0.02
    assert abs(hv.price - reference) < 0.02
    assert mcs.price == pytest.approx(hv.price, abs=5.0e-4)


@pytest.mark.numerical
def test_nonuniform_mcs_refines_toward_cf_with_feller_violation() -> None:
    p = PDEHestonParameters(0.02, 0.8, 0.03, 0.8, -0.5)
    reference = float(
        heston_price_adaptive(
            100.0, 110.0, 1.5, HestonParameters(0.02, 0.8, 0.03, 0.8, -0.5), 0.03, 0.01
        )
    )
    coarse = heston_pde_price(
        100.0,
        110.0,
        1.5,
        p,
        0.03,
        0.01,
        scheme="modified_craig_sneyd",
        grid=PDEGrid(61, 31, 80, variance_max=1.5, nonuniform=True),
    )
    fine = heston_pde_price(
        100.0,
        110.0,
        1.5,
        p,
        0.03,
        0.01,
        scheme="modified_craig_sneyd",
        grid=PDEGrid(121, 61, 180, variance_max=1.5, nonuniform=True),
    )
    assert abs(fine.price - reference) < abs(coarse.price - reference)
    # This bounded test establishes convergence direction, not the research
    # registry tolerance; the three-level research runner owns qualification.
    assert abs(fine.price - reference) < 0.01
