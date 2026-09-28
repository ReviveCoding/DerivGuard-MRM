"""Raw SVI and surface SVI (SSVI) total-variance parameterizations.

These utilities operate on log-forward-moneyness ``k = log(K/F)`` and total
implied variance ``w = sigma_IV**2 * T``.  The diagnostics expose explicit
positivity, wing-slope, density, and calendar checks; they do not silently
repair an invalid fitted surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SVIParameters:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def validate(self) -> None:
        values = (self.a, self.b, self.rho, self.m, self.sigma)
        if not all(np.isfinite(x) for x in values):
            raise ValueError("SVI parameters must be finite")
        if self.b < 0.0 or self.sigma <= 0.0 or abs(self.rho) >= 1.0:
            raise ValueError("SVI requires b >= 0, sigma > 0, and |rho| < 1")
        minimum = self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho * self.rho)
        if minimum < 0.0:
            raise ValueError("SVI minimum total variance must be non-negative")


@dataclass(frozen=True)
class SurfaceDiagnostics:
    valid: bool
    minimum_total_variance: float
    minimum_density_factor: float
    minimum_calendar_increment: float
    maximum_left_wing_slope: float
    maximum_right_wing_slope: float
    messages: tuple[str, ...]


def svi_total_variance(log_moneyness: ArrayLike, parameters: SVIParameters) -> float | FloatArray:
    parameters.validate()
    k = np.asarray(log_moneyness, dtype=np.float64)
    x = k - parameters.m
    value = parameters.a + parameters.b * (
        parameters.rho * x + np.sqrt(x * x + parameters.sigma * parameters.sigma)
    )
    return float(value) if value.ndim == 0 else value


def svi_derivatives(
    log_moneyness: ArrayLike, parameters: SVIParameters
) -> tuple[float | FloatArray, float | FloatArray]:
    """Return first and second derivatives of total variance with respect to k."""

    parameters.validate()
    k = np.asarray(log_moneyness, dtype=np.float64)
    x = k - parameters.m
    root = np.sqrt(x * x + parameters.sigma * parameters.sigma)
    first = parameters.b * (parameters.rho + x / root)
    second = parameters.b * parameters.sigma * parameters.sigma / (root * root * root)
    if k.ndim == 0:
        return float(first), float(second)
    return first, second


def density_factor(
    log_moneyness: ArrayLike,
    total_variance: ArrayLike,
    first_derivative: ArrayLike,
    second_derivative: ArrayLike,
) -> float | FloatArray:
    """Return the Gatheral-Jacquier risk-neutral density factor g(k)."""

    k, w, wp, wpp = np.broadcast_arrays(
        *[
            np.asarray(x, dtype=np.float64)
            for x in (log_moneyness, total_variance, first_derivative, second_derivative)
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (1.0 - k * wp / (2.0 * w)) ** 2 - 0.25 * wp * wp * (1.0 / w + 0.25) + 0.5 * wpp
    result = np.where(w > 0.0, result, -np.inf)
    return float(result) if result.ndim == 0 else result


def check_svi_slice(
    parameters: SVIParameters,
    *,
    log_moneyness_grid: ArrayLike | None = None,
    tolerance: float = 1.0e-10,
) -> SurfaceDiagnostics:
    """Numerically check positivity and butterfly/wing conditions on a slice."""

    parameters.validate()
    grid = np.asarray(
        np.linspace(-5.0, 5.0, 2001) if log_moneyness_grid is None else log_moneyness_grid,
        dtype=np.float64,
    )
    if grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid)):
        raise ValueError("log_moneyness_grid must be a finite one-dimensional grid")
    w = np.asarray(svi_total_variance(grid, parameters))
    wp, wpp = svi_derivatives(grid, parameters)
    g = np.asarray(density_factor(grid, w, wp, wpp))
    left_slope = parameters.b * (1.0 - parameters.rho)
    right_slope = parameters.b * (1.0 + parameters.rho)
    messages: list[str] = []
    if np.min(w) < -tolerance:
        messages.append("negative total variance")
    if np.min(g) < -tolerance:
        messages.append("negative density factor on diagnostic grid")
    if left_slope > 2.0 + tolerance or right_slope > 2.0 + tolerance:
        messages.append("Lee wing-slope bound exceeded")
    return SurfaceDiagnostics(
        valid=not messages,
        minimum_total_variance=float(np.min(w)),
        minimum_density_factor=float(np.min(g)),
        minimum_calendar_increment=np.nan,
        maximum_left_wing_slope=float(left_slope),
        maximum_right_wing_slope=float(right_slope),
        messages=tuple(messages),
    )


@dataclass(frozen=True)
class SSVIParameters:
    """Power-law SSVI shape parameters.

    ``phi(theta) = eta / (theta**gamma * (1+theta)**(1-gamma))``.
    The parameterization makes phi non-increasing and theta*phi
    non-decreasing for ``0 <= gamma <= 1``.
    """

    rho: float
    eta: float
    gamma: float

    def validate(self) -> None:
        if not all(np.isfinite(x) for x in (self.rho, self.eta, self.gamma)):
            raise ValueError("SSVI parameters must be finite")
        if abs(self.rho) >= 1.0 or self.eta <= 0.0 or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("SSVI requires |rho| < 1, eta > 0, and 0 <= gamma <= 1")

    def phi(self, theta: ArrayLike) -> float | FloatArray:
        self.validate()
        theta_array = np.asarray(theta, dtype=np.float64)
        if np.any(~np.isfinite(theta_array)) or np.any(theta_array <= 0.0):
            raise ValueError("ATM total variance theta must be finite and positive")
        value = self.eta / (
            np.power(theta_array, self.gamma) * np.power(1.0 + theta_array, 1.0 - self.gamma)
        )
        return float(value) if value.ndim == 0 else value


def ssvi_total_variance(
    log_moneyness: ArrayLike,
    theta: ArrayLike,
    parameters: SSVIParameters,
) -> float | FloatArray:
    parameters.validate()
    k, theta_array = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=np.float64), np.asarray(theta, dtype=np.float64)
    )
    phi = np.asarray(parameters.phi(theta_array))
    rho = parameters.rho
    phi_k = phi * k
    value = 0.5 * theta_array * (1.0 + rho * phi_k + np.sqrt((phi_k + rho) ** 2 + 1.0 - rho * rho))
    return float(value) if value.ndim == 0 else value


def ssvi_derivatives(
    log_moneyness: ArrayLike,
    theta: ArrayLike,
    parameters: SSVIParameters,
) -> tuple[float | FloatArray, float | FloatArray]:
    """Return analytic first and second log-moneyness derivatives of SSVI."""

    parameters.validate()
    k, theta_array = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=np.float64), np.asarray(theta, dtype=np.float64)
    )
    phi = np.asarray(parameters.phi(theta_array))
    rho = parameters.rho
    shifted = phi * k + rho
    root = np.sqrt(shifted * shifted + 1.0 - rho * rho)
    first = 0.5 * theta_array * phi * (rho + shifted / root)
    second = 0.5 * theta_array * phi * phi * (1.0 - rho * rho) / (root * root * root)
    if first.ndim == 0:
        return float(first), float(second)
    return first, second


@dataclass(frozen=True)
class SSVISurface:
    """SSVI surface with a non-decreasing ATM total-variance term structure."""

    maturities: FloatArray
    atm_total_variances: FloatArray
    parameters: SSVIParameters

    def __post_init__(self) -> None:
        maturities = np.asarray(self.maturities, dtype=np.float64)
        theta = np.asarray(self.atm_total_variances, dtype=np.float64)
        if maturities.ndim != 1 or theta.ndim != 1 or maturities.size != theta.size:
            raise ValueError("maturities and ATM total variances must be equal-length vectors")
        if maturities.size < 1 or np.any(~np.isfinite(maturities)) or np.any(maturities <= 0.0):
            raise ValueError("maturities must be finite and positive")
        if np.any(np.diff(maturities) <= 0.0):
            raise ValueError("maturities must be strictly increasing")
        if np.any(~np.isfinite(theta)) or np.any(theta <= 0.0):
            raise ValueError("ATM total variances must be finite and positive")
        if np.any(np.diff(theta) < 0.0):
            raise ValueError("ATM total variance must be non-decreasing with maturity")
        self.parameters.validate()
        object.__setattr__(self, "maturities", maturities.copy())
        object.__setattr__(self, "atm_total_variances", theta.copy())

    def theta(self, maturity: ArrayLike) -> float | FloatArray:
        t = np.asarray(maturity, dtype=np.float64)
        if np.any(~np.isfinite(t)) or np.any(t <= 0.0):
            raise ValueError("maturity must be finite and positive")
        if np.any(t < self.maturities[0]) or np.any(t > self.maturities[-1]):
            raise ValueError("maturity extrapolation is disabled")
        value = np.interp(t, self.maturities, self.atm_total_variances)
        return float(value) if value.ndim == 0 else value

    def total_variance(self, log_moneyness: ArrayLike, maturity: ArrayLike) -> float | FloatArray:
        return ssvi_total_variance(log_moneyness, self.theta(maturity), self.parameters)

    def implied_volatility(
        self, log_moneyness: ArrayLike, maturity: ArrayLike
    ) -> float | FloatArray:
        t = np.asarray(maturity, dtype=np.float64)
        variance = np.asarray(self.total_variance(log_moneyness, t))
        result = np.sqrt(variance / t)
        return float(result) if result.ndim == 0 else result

    def diagnostics(
        self,
        *,
        log_moneyness_grid: ArrayLike | None = None,
        tolerance: float = 1.0e-10,
    ) -> SurfaceDiagnostics:
        """Check SSVI sufficient conditions plus gridded calendar monotonicity."""

        k = np.asarray(
            np.linspace(-5.0, 5.0, 1001) if log_moneyness_grid is None else log_moneyness_grid,
            dtype=np.float64,
        )
        if k.ndim != 1 or k.size < 3 or np.any(~np.isfinite(k)):
            raise ValueError("log_moneyness_grid must be a finite one-dimensional grid")
        theta = self.atm_total_variances
        phi = np.asarray(self.parameters.phi(theta))
        rho_abs = abs(self.parameters.rho)
        condition_one = theta * phi * (1.0 + rho_abs)
        condition_two = theta * phi * phi * (1.0 + rho_abs)
        matrix = np.vstack(
            [np.asarray(ssvi_total_variance(k, value, self.parameters)) for value in theta]
        )
        density_rows: list[FloatArray] = []
        for value, variance_row in zip(theta, matrix, strict=True):
            first, second = ssvi_derivatives(k, value, self.parameters)
            density_rows.append(
                np.asarray(density_factor(k, variance_row, first, second), dtype=np.float64)
            )
        density_matrix = np.vstack(density_rows)
        calendar_increment = np.diff(matrix, axis=0)
        minimum_calendar = float(np.min(calendar_increment)) if calendar_increment.size else np.inf
        min_variance = float(np.min(matrix))
        messages: list[str] = []
        if np.max(condition_one) > 4.0 + tolerance:
            messages.append("SSVI wing-slope sufficient condition exceeded")
        if np.max(condition_two) > 4.0 + tolerance:
            messages.append("SSVI butterfly sufficient condition exceeded")
        if minimum_calendar < -tolerance:
            messages.append("calendar total variance decreases on diagnostic grid")
        if min_variance < -tolerance:
            messages.append("negative total variance")
        minimum_density = float(np.min(density_matrix))
        if minimum_density < -tolerance:
            messages.append("negative density factor on diagnostic grid")
        # SSVI formula has asymptotic total-variance slopes theta*phi*(1-/+rho)/2.
        left = 0.5 * theta * phi * (1.0 - self.parameters.rho)
        right = 0.5 * theta * phi * (1.0 + self.parameters.rho)
        return SurfaceDiagnostics(
            valid=not messages,
            minimum_total_variance=min_variance,
            minimum_density_factor=minimum_density,
            minimum_calendar_increment=minimum_calendar,
            maximum_left_wing_slope=float(np.max(left)),
            maximum_right_wing_slope=float(np.max(right)),
            messages=tuple(messages),
        )
