"""Independent two-dimensional finite-difference Heston validator.

The solver is intentionally independent of the developer CF kernel.  It uses
a sparse full-operator discretization including the mixed derivative and
offers legacy full-operator reference schemes plus Modified Craig--Sneyd
(MCS) and Hundsdorfer--Verwer (HV) ADI schemes on nonuniform grids.  The ADI
splitting follows in 't Hout and Foulon (2010): the mixed derivative is the
explicit A0 part while spot and variance operators are solved implicitly.
Convergence and boundary sensitivity must accompany reported values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csc_matrix, eye, lil_matrix
from scipy.sparse.linalg import splu

PDEScheme = Literal[
    "implicit_euler", "crank_nicolson", "modified_craig_sneyd", "hundsdorfer_verwer"
]
OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class PDEHestonParameters:
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float

    def validate(self) -> None:
        if self.v0 < 0.0 or self.kappa <= 0.0 or self.theta < 0.0 or self.sigma_v <= 0.0:
            raise ValueError("invalid Heston parameters")
        if abs(self.rho) > 1.0:
            raise ValueError("rho must lie in [-1,1]")


@dataclass(frozen=True)
class PDEGrid:
    spot_nodes: int = 161
    variance_nodes: int = 81
    time_steps: int = 200
    spot_max_multiple: float = 4.0
    variance_max: float | None = None
    nonuniform: bool = False
    rannacher_steps: int = 2

    def validate(self) -> None:
        if self.spot_nodes < 21 or self.variance_nodes < 11 or self.time_steps < 1:
            raise ValueError("PDE grid is too small")
        if self.spot_max_multiple <= 1.0:
            raise ValueError("spot_max_multiple must exceed one")
        if self.rannacher_steps < 0 or self.rannacher_steps > self.time_steps:
            raise ValueError("rannacher_steps must be between zero and time_steps")


@dataclass(frozen=True)
class PDEResult:
    price: float
    scheme: PDEScheme
    spot_nodes: int
    variance_nodes: int
    time_steps: int
    spot_max: float
    variance_max: float
    minimum_grid_value: float


def _index(j: int, i: int, ns: int) -> int:
    return j * ns + i


def _operator(
    spots: np.ndarray,
    variances: np.ndarray,
    p: PDEHestonParameters,
    rate: float,
    dividend_yield: float,
) -> csc_matrix:
    ns = len(spots)
    nv = len(variances)
    ds = float(spots[1] - spots[0])
    dv = float(variances[1] - variances[0])
    matrix = lil_matrix((ns * nv, ns * nv), dtype=np.float64)
    # Spot and upper-variance boundary rows remain zero in L.  At v=0 the
    # degenerate first-order Heston PDE is discretized with a second-order
    # forward variance derivative rather than an artificial reflecting wall.
    for i in range(1, ns - 1):
        s = spots[i]
        row = _index(0, i, ns)
        as1 = (rate - dividend_yield) * s / (2.0 * ds)
        av1 = p.kappa * p.theta / (2.0 * dv)
        matrix[row, _index(0, i - 1, ns)] += -as1
        matrix[row, _index(0, i + 1, ns)] += as1
        matrix[row, row] += -3.0 * av1 - rate
        matrix[row, _index(1, i, ns)] += 4.0 * av1
        matrix[row, _index(2, i, ns)] += -av1
    # Interior rows use centered second and first terms.
    for j in range(1, nv - 1):
        v = variances[j]
        for i in range(1, ns - 1):
            s = spots[i]
            row = _index(j, i, ns)
            ass = 0.5 * v * s * s / (ds * ds)
            as1 = (rate - dividend_yield) * s / (2.0 * ds)
            avv = 0.5 * p.sigma_v * p.sigma_v * v / (dv * dv)
            av1 = p.kappa * (p.theta - v) / (2.0 * dv)
            cross = p.rho * p.sigma_v * v * s / (4.0 * ds * dv)
            matrix[row, _index(j, i - 1, ns)] += ass - as1
            matrix[row, _index(j, i + 1, ns)] += ass + as1
            matrix[row, _index(j - 1, i, ns)] += avv - av1
            matrix[row, _index(j + 1, i, ns)] += avv + av1
            matrix[row, _index(j + 1, i + 1, ns)] += cross
            matrix[row, _index(j + 1, i - 1, ns)] -= cross
            matrix[row, _index(j - 1, i + 1, ns)] -= cross
            matrix[row, _index(j - 1, i - 1, ns)] += cross
            matrix[row, row] += -2.0 * ass - 2.0 * avv - rate
    return matrix.tocsc()


def _boundary_value(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
) -> float:
    if option_type == "call":
        if spot == 0.0:
            return 0.0
        return float(max(spot * np.exp(-dividend_yield * tau) - strike * np.exp(-rate * tau), 0.0))
    if spot == 0.0:
        return float(strike * np.exp(-rate * tau))
    return 0.0


def _apply_boundary_rows(
    lhs: lil_matrix,
    rhs_operator: lil_matrix,
    ns: int,
    nv: int,
) -> None:
    # Spot boundaries: Dirichlet. Upper variance boundary: zero-gradient. At
    # the corners Dirichlet takes precedence. The v=0 row is a PDE row.
    for j in range(nv):
        for i in (0, ns - 1):
            row = _index(j, i, ns)
            lhs.rows[row] = [row]
            lhs.data[row] = [1.0]
            rhs_operator.rows[row] = []
            rhs_operator.data[row] = []
    for i in range(1, ns - 1):
        high = _index(nv - 1, i, ns)
        lhs.rows[high] = [high, _index(nv - 2, i, ns)]
        lhs.data[high] = [1.0, -1.0]
        rhs_operator.rows[high] = []
        rhs_operator.data[high] = []


def _centered_nonuniform_grid(
    lower: float,
    center: float,
    upper: float,
    nodes: int,
    left_fraction: float,
    scale: float,
) -> np.ndarray:
    """Return an increasing sinh grid containing ``center`` exactly."""

    left_intervals = max(2, min(nodes - 3, round((nodes - 1) * left_fraction)))
    right_intervals = nodes - 1 - left_intervals
    left_x = np.linspace(np.arcsinh((lower - center) / scale), 0.0, left_intervals + 1)
    right_x = np.linspace(0.0, np.arcsinh((upper - center) / scale), right_intervals + 1)
    left = center + scale * np.sinh(left_x)
    right = center + scale * np.sinh(right_x)
    result = np.concatenate((left, right[1:]))
    result[0], result[left_intervals], result[-1] = lower, center, upper
    return result


def _first_second_weights(grid: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Three-point first/second derivative weights on a nonuniform grid."""

    hm = float(grid[index] - grid[index - 1])
    hp = float(grid[index + 1] - grid[index])
    first = np.asarray((-hp / (hm * (hm + hp)), (hp - hm) / (hm * hp), hm / (hp * (hm + hp))))
    second = np.asarray((2.0 / (hm * (hm + hp)), -2.0 / (hm * hp), 2.0 / (hp * (hm + hp))))
    return first, second


def _forward_first_weights(grid: np.ndarray) -> np.ndarray:
    """Second-order one-sided derivative weights at the variance origin."""

    offsets = grid[:3] - grid[0]
    system = np.vstack((np.ones(3), offsets, offsets * offsets))
    return np.asarray(np.linalg.solve(system, np.asarray((0.0, 1.0, 0.0))), dtype=np.float64)


def _split_operators(
    spots: np.ndarray,
    variances: np.ndarray,
    p: PDEHestonParameters,
    rate: float,
    dividend_yield: float,
) -> tuple[csc_matrix, csc_matrix, csc_matrix]:
    """Build A0 (mixed), A1 (spot), and A2 (variance) Heston operators."""

    ns, nv = len(spots), len(variances)
    size = ns * nv
    mixed = lil_matrix((size, size), dtype=np.float64)
    spot_op = lil_matrix((size, size), dtype=np.float64)
    variance_op = lil_matrix((size, size), dtype=np.float64)

    spot_first: dict[int, np.ndarray] = {}
    spot_second: dict[int, np.ndarray] = {}
    for i in range(1, ns - 1):
        spot_first[i], spot_second[i] = _first_second_weights(spots, i)
    variance_first: dict[int, np.ndarray] = {}
    variance_second: dict[int, np.ndarray] = {}
    for j in range(1, nv - 1):
        variance_first[j], variance_second[j] = _first_second_weights(variances, j)

    # The v=0 boundary is the degenerate Heston PDE, not a reflecting
    # condition.  A second-order forward derivative represents kappa*theta*V_v.
    forward = _forward_first_weights(variances)
    for i in range(1, ns - 1):
        row = _index(0, i, ns)
        s = float(spots[i])
        for offset, coefficient in zip((-1, 0, 1), spot_first[i], strict=True):
            spot_op[row, _index(0, i + offset, ns)] += (rate - dividend_yield) * s * coefficient
        spot_op[row, row] -= rate
        for j, coefficient in enumerate(forward):
            variance_op[row, _index(j, i, ns)] += p.kappa * p.theta * coefficient

    for j in range(1, nv - 1):
        v = float(variances[j])
        for i in range(1, ns - 1):
            s = float(spots[i])
            row = _index(j, i, ns)
            for offset, (first, second) in enumerate(
                zip(spot_first[i], spot_second[i], strict=True), start=-1
            ):
                spot_op[row, _index(j, i + offset, ns)] += (
                    rate - dividend_yield
                ) * s * first + 0.5 * v * s * s * second
            spot_op[row, row] -= rate
            for offset, (first, second) in enumerate(
                zip(variance_first[j], variance_second[j], strict=True), start=-1
            ):
                variance_op[row, _index(j + offset, i, ns)] += (
                    p.kappa * (p.theta - v) * first + 0.5 * p.sigma_v * p.sigma_v * v * second
                )
            cross_scale = p.rho * p.sigma_v * v * s
            for js, v_weight in enumerate(variance_first[j], start=-1):
                for is_, s_weight in enumerate(spot_first[i], start=-1):
                    mixed[row, _index(j + js, i + is_, ns)] += cross_scale * v_weight * s_weight
    return mixed.tocsc(), spot_op.tocsc(), variance_op.tocsc()


def _set_adi_boundary_matrix(matrix: lil_matrix, ns: int, nv: int) -> None:
    """Set Dirichlet spot and upper-variance Neumann constraint rows."""

    for j in range(nv):
        for i in (0, ns - 1):
            row = _index(j, i, ns)
            matrix.rows[row] = [row]
            matrix.data[row] = [1.0]
    for i in range(1, ns - 1):
        row = _index(nv - 1, i, ns)
        matrix.rows[row] = [row, _index(nv - 2, i, ns)]
        matrix.data[row] = [1.0, -1.0]


def _enforce_adi_boundaries(
    values: np.ndarray,
    spots: np.ndarray,
    strike: float,
    tau: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
) -> np.ndarray:
    """Apply financial spot boundaries and an upper-variance zero gradient."""

    nv, ns = len(values) // len(spots), len(spots)
    shaped = np.asarray(values, dtype=np.float64).reshape(nv, ns).copy()
    shaped[:, 0] = _boundary_value(0.0, strike, tau, rate, dividend_yield, option_type)
    shaped[:, -1] = _boundary_value(
        float(spots[-1]), strike, tau, rate, dividend_yield, option_type
    )
    shaped[-1, 1:-1] = shaped[-2, 1:-1]
    return shaped.reshape(-1)


def _heston_pde_adi(
    spot: float,
    strike: float,
    maturity: float,
    params: PDEHestonParameters,
    rate: float,
    dividend_yield: float,
    option_type: OptionType,
    scheme: PDEScheme,
    grid: PDEGrid,
) -> PDEResult:
    """MCS/HV ADI solve with explicit mixed derivative and Rannacher damping."""

    smax = grid.spot_max_multiple * max(spot, strike)
    stationary_std = params.sigma_v * np.sqrt(max(params.theta, 1.0e-8) / (2.0 * params.kappa))
    natural_vmax = max(0.5, params.v0 + 8.0 * stationary_std)
    vmax = grid.variance_max if grid.variance_max is not None else natural_vmax
    if vmax <= params.v0:
        raise ValueError("variance_max must exceed v0")

    if grid.nonuniform:
        spots = _centered_nonuniform_grid(
            0.0,
            spot,
            smax,
            grid.spot_nodes,
            0.5,
            max(0.04 * spot, 1.0e-4),
        )
        variances = _centered_nonuniform_grid(
            0.0,
            params.v0,
            vmax,
            grid.variance_nodes,
            0.3,
            max(0.25 * max(params.v0, params.theta), 1.0e-4),
        )
    else:
        spots = np.linspace(0.0, smax, grid.spot_nodes)
        variances = np.linspace(0.0, vmax, grid.variance_nodes)
    ns, nv = len(spots), len(variances)
    a0, a1, a2 = _split_operators(spots, variances, params, rate, dividend_yield)
    total = a0 + a1 + a2
    dt = maturity / grid.time_steps
    theta = 1.0 / 3.0 if scheme == "modified_craig_sneyd" else 0.5 + np.sqrt(3.0) / 6.0
    identity = eye(ns * nv, format="csc", dtype=np.float64)

    direction_factors = []
    for operator in (a1, a2):
        matrix = (identity - theta * dt * operator).tolil()
        _set_adi_boundary_matrix(matrix, ns, nv)
        direction_factors.append(splu(matrix.tocsc()))

    half_dt = 0.5 * dt
    rannacher_matrix = (identity - half_dt * total).tolil()
    _set_adi_boundary_matrix(rannacher_matrix, ns, nv)
    rannacher_factor = splu(rannacher_matrix.tocsc())

    terminal = (
        np.maximum(spots - strike, 0.0)
        if option_type == "call"
        else np.maximum(strike - spots, 0.0)
    )
    values = np.tile(terminal, (nv, 1)).reshape(-1)

    def solve_direction(factor: object, rhs: np.ndarray, tau: float) -> np.ndarray:
        constrained = _enforce_adi_boundaries(
            rhs, spots, strike, tau, rate, dividend_yield, option_type
        )
        shaped_rhs = constrained.reshape(nv, ns)
        shaped_rhs[-1, 1:-1] = 0.0
        # SuperLU's solve is used through its stable public method; the local
        # annotation avoids coupling the validator API to SciPy internals.
        solved = factor.solve(shaped_rhs.reshape(-1))  # type: ignore[attr-defined]
        return _enforce_adi_boundaries(
            np.asarray(solved), spots, strike, tau, rate, dividend_yield, option_type
        )

    tau = 0.0
    for step in range(grid.time_steps):
        tau_new = (step + 1) * dt
        if step < grid.rannacher_steps:
            for half in range(2):
                tau_half = tau + (half + 1) * half_dt
                values = solve_direction(rannacher_factor, values, tau_half)
            tau = tau_new
            continue

        previous = values
        y0 = _enforce_adi_boundaries(
            previous + dt * (total @ previous),
            spots,
            strike,
            tau_new,
            rate,
            dividend_yield,
            option_type,
        )
        y = y0
        for operator, factor in zip((a1, a2), direction_factors, strict=True):
            y = solve_direction(factor, y - theta * dt * (operator @ previous), tau_new)

        if scheme == "modified_craig_sneyd":
            corrected = y0 + theta * dt * (a0 @ (y - previous))
            corrected += (0.5 - theta) * dt * (total @ (y - previous))
        else:
            corrected = y0 + 0.5 * dt * (total @ (y - previous))
        corrected = _enforce_adi_boundaries(
            corrected, spots, strike, tau_new, rate, dividend_yield, option_type
        )
        baseline = previous if scheme == "modified_craig_sneyd" else y
        for operator, factor in zip((a1, a2), direction_factors, strict=True):
            corrected = solve_direction(
                factor, corrected - theta * dt * (operator @ baseline), tau_new
            )
        values = corrected
        tau = tau_new

    surface = values.reshape(nv, ns)
    interpolator = RegularGridInterpolator(
        (variances, spots), surface, method="linear", bounds_error=True
    )
    price = float(interpolator(np.asarray([[params.v0, spot]]))[0])
    return PDEResult(
        price=price,
        scheme=scheme,
        spot_nodes=ns,
        variance_nodes=nv,
        time_steps=grid.time_steps,
        spot_max=smax,
        variance_max=vmax,
        minimum_grid_value=float(np.min(surface)),
    )


def heston_pde_price(
    spot: float,
    strike: float,
    maturity: float,
    params: PDEHestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    option_type: OptionType = "call",
    *,
    scheme: PDEScheme = "crank_nicolson",
    grid: PDEGrid | None = None,
) -> PDEResult:
    """Solve the backward Heston PDE with the requested independent scheme.

    ``implicit_euler`` and ``crank_nicolson`` preserve the original uniform
    full-operator reference implementation.  MCS and HV use split operators
    and support nonuniform grids; the caller must set ``grid.nonuniform``
    explicitly so convergence studies can compare grid constructions.
    """

    params.validate()
    if grid is None:
        grid = PDEGrid()
    grid.validate()
    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike, and maturity must be positive")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    if scheme not in {
        "implicit_euler",
        "crank_nicolson",
        "modified_craig_sneyd",
        "hundsdorfer_verwer",
    }:
        raise ValueError("unsupported PDE scheme")
    if scheme in {"modified_craig_sneyd", "hundsdorfer_verwer"}:
        return _heston_pde_adi(
            spot,
            strike,
            maturity,
            params,
            rate,
            dividend_yield,
            option_type,
            scheme,
            grid,
        )

    smax = grid.spot_max_multiple * max(spot, strike)
    stationary_std = params.sigma_v * np.sqrt(max(params.theta, 1.0e-8) / (2.0 * params.kappa))
    natural_vmax = max(0.5, params.v0 + 8.0 * stationary_std)
    vmax = grid.variance_max if grid.variance_max is not None else natural_vmax
    if vmax <= params.v0:
        raise ValueError("variance_max must exceed v0")
    spots = np.linspace(0.0, smax, grid.spot_nodes)
    variances = np.linspace(0.0, vmax, grid.variance_nodes)
    ns, nv = len(spots), len(variances)
    operator = _operator(spots, variances, params, rate, dividend_yield)
    dt = maturity / grid.time_steps
    identity = eye(ns * nv, format="csc", dtype=np.float64)
    weight = 1.0 if scheme == "implicit_euler" else 0.5
    lhs = (identity - weight * dt * operator).tolil()
    rhs_operator = (identity + (1.0 - weight) * dt * operator).tolil()
    _apply_boundary_rows(lhs, rhs_operator, ns, nv)
    factor = splu(lhs.tocsc())
    rhs_matrix = rhs_operator.tocsc()

    if option_type == "call":
        terminal = np.maximum(spots - strike, 0.0)
    else:
        terminal = np.maximum(strike - spots, 0.0)
    values = np.tile(terminal, (nv, 1)).reshape(-1)
    for step in range(1, grid.time_steps + 1):
        tau = step * dt
        rhs = np.asarray(rhs_matrix @ values).reshape(-1)
        for j in range(nv):
            rhs[_index(j, 0, ns)] = _boundary_value(
                0.0, strike, tau, rate, dividend_yield, option_type
            )
            rhs[_index(j, ns - 1, ns)] = _boundary_value(
                smax, strike, tau, rate, dividend_yield, option_type
            )
        # Homogeneous Neumann constraint at the truncated upper variance edge.
        for i in range(1, ns - 1):
            rhs[_index(nv - 1, i, ns)] = 0.0
        values = factor.solve(rhs)

    surface = values.reshape(nv, ns)
    interpolator = RegularGridInterpolator(
        (variances, spots), surface, method="linear", bounds_error=True
    )
    price = float(interpolator(np.asarray([[params.v0, spot]]))[0])
    return PDEResult(
        price=price,
        scheme=scheme,
        spot_nodes=ns,
        variance_nodes=nv,
        time_steps=grid.time_steps,
        spot_max=smax,
        variance_max=vmax,
        minimum_grid_value=float(np.min(surface)),
    )
