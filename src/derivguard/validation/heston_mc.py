"""Independent Heston Monte Carlo validator.

This module intentionally does not import the developer characteristic-
function implementation.  It supplies full-truncation Euler and Andersen's
quadratic-exponential (QE) and martingale-corrected QE-M variance schemes,
NumPy and lazy Torch execution,
uncertainty estimates, antithetic sampling, and an optional discounted-stock
control variate.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Scheme = Literal["full_truncation", "qe", "qe_m"]
OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class ValidationHestonParameters:
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float

    def validate(self) -> None:
        x = np.asarray((self.v0, self.kappa, self.theta, self.sigma_v, self.rho))
        if not np.all(np.isfinite(x)):
            raise ValueError("parameters must be finite")
        if self.v0 < 0.0 or self.kappa <= 0.0 or self.theta < 0.0 or self.sigma_v <= 0.0:
            raise ValueError("invalid non-positive variance-process parameter")
        if abs(self.rho) > 1.0:
            raise ValueError("rho must lie in [-1,1]")


@dataclass(frozen=True)
class MonteCarloEstimate:
    price: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    paths: int
    steps: int
    scheme: Scheme
    backend: str
    antithetic: bool
    control_variate: bool
    seed: int
    oom_retries: int = 0


def _qe_variance_numpy(
    variance: NDArray[np.float64],
    normal: NDArray[np.float64],
    uniform: NDArray[np.float64],
    dt: float,
    p: ValidationHestonParameters,
    psi_cutoff: float = 1.5,
) -> NDArray[np.float64]:
    exp_kdt = np.exp(-p.kappa * dt)
    mean = p.theta + (variance - p.theta) * exp_kdt
    sigma2 = variance * p.sigma_v**2 * exp_kdt * (
        1.0 - exp_kdt
    ) / p.kappa + p.theta * p.sigma_v**2 * (1.0 - exp_kdt) ** 2 / (2.0 * p.kappa)
    mean = np.maximum(mean, np.finfo(np.float64).tiny)
    psi = sigma2 / (mean * mean)
    quadratic = psi <= psi_cutoff
    psi_q = np.maximum(psi, np.finfo(np.float64).eps)
    two_over = 2.0 / psi_q
    b2 = two_over - 1.0 + np.sqrt(np.maximum(two_over * (two_over - 1.0), 0.0))
    a = mean / (1.0 + b2)
    q_value = a * (np.sqrt(np.maximum(b2, 0.0)) + normal) ** 2
    p0 = np.clip((psi - 1.0) / (psi + 1.0), 0.0, 1.0 - 1.0e-14)
    beta = (1.0 - p0) / mean
    u = np.clip(uniform, 1.0e-14, 1.0 - 1.0e-14)
    e_value = np.where(u <= p0, 0.0, np.log((1.0 - p0) / (1.0 - u)) / beta)
    return np.asarray(np.where(quadratic, q_value, e_value), dtype=np.float64)


def _qe_conditional_log_mgf_numpy(
    variance: NDArray[np.float64],
    dt: float,
    p: ValidationHestonParameters,
    argument: float,
    psi_cutoff: float = 1.5,
) -> NDArray[np.float64]:
    """Log conditional MGF of the QE variance approximation.

    For ``V(t+dt)=a(b+Z)^2`` the MGF is a shifted non-central chi-square
    MGF.  For the atom-plus-exponential branch it is the corresponding
    mixture MGF.  QE-M uses this quantity to make the one-step discounted
    stock price a conditional martingale.  Invalid MGF domains are rejected
    rather than clipped because clipping would silently invalidate the
    correction.
    """

    exp_kdt = np.exp(-p.kappa * dt)
    mean = p.theta + (variance - p.theta) * exp_kdt
    sigma2 = variance * p.sigma_v**2 * exp_kdt * (
        1.0 - exp_kdt
    ) / p.kappa + p.theta * p.sigma_v**2 * (1.0 - exp_kdt) ** 2 / (2.0 * p.kappa)
    mean = np.maximum(mean, np.finfo(np.float64).tiny)
    psi = sigma2 / (mean * mean)
    safe_psi = np.maximum(psi, np.finfo(np.float64).eps)
    two_over = 2.0 / safe_psi
    b2 = two_over - 1.0 + np.sqrt(np.maximum(two_over * (two_over - 1.0), 0.0))
    a = mean / (1.0 + b2)
    quadratic_denom = 1.0 - 2.0 * argument * a

    p0 = np.clip((psi - 1.0) / (psi + 1.0), 0.0, 1.0 - 1.0e-14)
    beta = (1.0 - p0) / mean
    exponential_denom = beta - argument
    quadratic = psi <= psi_cutoff
    if np.any(quadratic & (quadratic_denom <= 0.0)) or np.any(
        (~quadratic) & (exponential_denom <= 0.0)
    ):
        raise FloatingPointError("QE-M conditional MGF does not exist for this step")

    log_quadratic = -0.5 * np.log(quadratic_denom) + argument * a * b2 / quadratic_denom
    exponential_mgf = p0 + (1.0 - p0) * beta / exponential_denom
    # np.where evaluates both branches, so protect inactive quadratic-branch
    # entries from a spurious log warning.  Active exponential entries were
    # domain-checked above and are not altered by this guard.
    log_exponential = np.log(np.maximum(exponential_mgf, np.finfo(np.float64).tiny))
    return np.asarray(np.where(quadratic, log_quadratic, log_exponential), dtype=np.float64)


def _payoff(
    terminal: NDArray[np.float64], strike: float, option_type: OptionType
) -> NDArray[np.float64]:
    if option_type == "call":
        return np.asarray(np.maximum(terminal - strike, 0.0), dtype=np.float64)
    if option_type == "put":
        return np.asarray(np.maximum(strike - terminal, 0.0), dtype=np.float64)
    raise ValueError("option_type must be 'call' or 'put'")


def heston_mc_numpy(
    spot: float,
    strike: float,
    maturity: float,
    params: ValidationHestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    paths: int = 100_000,
    steps: int = 128,
    scheme: Scheme = "full_truncation",
    option_type: OptionType = "call",
    seed: int = 17,
    antithetic: bool = True,
    control_variate: bool = True,
    confidence_level: float = 0.95,
) -> MonteCarloEstimate:
    """Price with a bounded NumPy reference simulation.

    QE uses Andersen's conditional moment-matching variance transition and
    the standard gamma1=gamma2=1/2 log-stock update.  QE-M uses the same
    variance transition but replaces the constant drift adjustment by the
    exact conditional-MGF correction for the *approximated* transition.  It
    is a discrete martingale correction, not an exact Heston simulation;
    convergence must still be checked.
    """

    params.validate()
    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike, and maturity must be positive")
    if paths < 2 or steps < 1:
        raise ValueError("paths >= 2 and steps >= 1 are required")
    if scheme not in {"full_truncation", "qe", "qe_m"}:
        raise ValueError("unknown scheme")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0,1)")

    rng = np.random.default_rng(seed)
    if antithetic:
        base_paths = (paths + 1) // 2
        simulated_paths = base_paths * 2
    else:
        base_paths = paths
        simulated_paths = paths
    log_s: NDArray[np.float64] = np.full(simulated_paths, np.log(spot), dtype=np.float64)
    variance: NDArray[np.float64] = np.full(simulated_paths, params.v0, dtype=np.float64)
    dt = maturity / steps
    sqrt_dt = np.sqrt(dt)
    orthogonal_scale = np.sqrt(max(1.0 - params.rho * params.rho, 0.0))

    for _ in range(steps):
        zv_base = rng.standard_normal(base_paths)
        zs_base = rng.standard_normal(base_paths)
        u_base = rng.random(base_paths)
        if antithetic:
            zv = np.concatenate((zv_base, -zv_base))
            zs = np.concatenate((zs_base, -zs_base))
            uniform = np.concatenate((u_base, 1.0 - u_base))
        else:
            zv, zs, uniform = zv_base, zs_base, u_base

        old_v = variance
        if scheme == "full_truncation":
            positive_v = np.maximum(old_v, 0.0)
            new_v = (
                old_v
                + params.kappa * (params.theta - positive_v) * dt
                + (params.sigma_v * np.sqrt(positive_v) * sqrt_dt * zv)
            )
            log_s += (rate - dividend_yield - 0.5 * positive_v) * dt + np.sqrt(
                positive_v
            ) * sqrt_dt * (params.rho * zv + orthogonal_scale * zs)
            variance = np.maximum(new_v, 0.0)
        else:
            new_v = _qe_variance_numpy(old_v, zv, uniform, dt, params)
            gamma1 = 0.5
            gamma2 = 0.5
            k1 = gamma1 * dt * (params.kappa * params.rho / params.sigma_v - 0.5) - (
                params.rho / params.sigma_v
            )
            k2 = gamma2 * dt * (params.kappa * params.rho / params.sigma_v - 0.5) + (
                params.rho / params.sigma_v
            )
            k3 = gamma1 * dt * (1.0 - params.rho**2)
            k4 = gamma2 * dt * (1.0 - params.rho**2)
            if scheme == "qe_m":
                mgf_argument = k2 + 0.5 * k4
                log_mgf = _qe_conditional_log_mgf_numpy(old_v, dt, params, mgf_argument)
                # Conditional on V_t, integrating first over the independent
                # stock normal and then V_{t+dt} gives E[exp(increment)]=1.
                k0_term: NDArray[np.float64] | float = -(k1 + 0.5 * k3) * old_v - log_mgf
            else:
                k0_term = -params.rho * params.kappa * params.theta * dt / params.sigma_v
            log_s += (
                (rate - dividend_yield) * dt
                + k0_term
                + k1 * old_v
                + k2 * new_v
                + np.sqrt(np.maximum(k3 * old_v + k4 * new_v, 0.0)) * zs
            )
            variance = new_v

    terminal_all = np.exp(log_s)
    discount = np.exp(-rate * maturity)
    payoff_all = discount * _payoff(terminal_all, strike, option_type)
    if antithetic:
        # Each antithetic pair is one independent sampling unit.  Averaging
        # the pair before estimating the variance avoids an artificially
        # narrow confidence interval caused by treating negatively correlated
        # members as independent observations.
        discounted_payoff = 0.5 * (payoff_all[:base_paths] + payoff_all[base_paths:])
        control = 0.5 * discount * (terminal_all[:base_paths] + terminal_all[base_paths:])
    else:
        discounted_payoff = payoff_all[:paths]
        control = discount * terminal_all[:paths]
    used_control = False
    if control_variate:
        expected_control = spot * np.exp(-dividend_yield * maturity)
        control_variance = float(np.var(control, ddof=1))
        if control_variance > 0.0:
            beta = float(np.cov(discounted_payoff, control, ddof=1)[0, 1]) / control_variance
            discounted_payoff = discounted_payoff - beta * (control - expected_control)
            used_control = True
    price = float(np.mean(discounted_payoff))
    stderr = float(np.std(discounted_payoff, ddof=1) / np.sqrt(discounted_payoff.size))
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return MonteCarloEstimate(
        price=price,
        standard_error=stderr,
        confidence_low=price - z * stderr,
        confidence_high=price + z * stderr,
        paths=paths,
        steps=steps,
        scheme=scheme,
        backend="numpy-cpu",
        antithetic=antithetic,
        control_variate=used_control,
        seed=seed,
    )


def heston_mc_torch(
    spot: float,
    strike: float,
    maturity: float,
    params: ValidationHestonParameters,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    paths: int = 1_000_000,
    steps: int = 128,
    scheme: Scheme = "full_truncation",
    option_type: OptionType = "call",
    seed: int = 17,
    antithetic: bool = True,
    control_variate: bool = True,
    device: str = "cuda",
    chunk_paths: int = 262_144,
) -> MonteCarloEstimate:
    """Research-scale Torch simulation with adaptive path chunking.

    Chunks accumulate first and second payoff moments on the host; no complete
    path tensor is persisted.  CUDA OOM halves the chunk and retries after
    releasing cached allocations.  Full truncation, QE, and QE-M are coded
    independently of the developer characteristic-function implementation.
    """

    if scheme not in {"full_truncation", "qe", "qe_m"}:
        raise ValueError("unknown scheme")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Torch is required for GPU Monte Carlo") from exc
    params.validate()
    if paths < 2 or steps < 1 or chunk_paths < 2:
        raise ValueError("invalid path, step, or chunk count")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    generator = torch.Generator(device=target)
    generator.manual_seed(seed)
    remaining = paths
    moment_count = 0
    sum_x = 0.0
    sum_x2 = 0.0
    dt = maturity / steps
    sqrt_dt = np.sqrt(dt)
    rho_orth = np.sqrt(max(1.0 - params.rho**2, 0.0))
    active_chunk = min(chunk_paths, paths)
    oom_retries = 0

    while remaining > 0:
        count = min(active_chunk, remaining)
        if antithetic and count > 2:
            count = min(count, remaining)
        try:
            base = (count + 1) // 2 if antithetic else count
            simulated = base * 2 if antithetic else count
            log_s = torch.full((simulated,), np.log(spot), dtype=torch.float64, device=target)
            variance = torch.full((simulated,), params.v0, dtype=torch.float64, device=target)
            for _ in range(steps):
                zv0 = torch.randn(base, dtype=torch.float64, device=target, generator=generator)
                zs0 = torch.randn(base, dtype=torch.float64, device=target, generator=generator)
                u0 = torch.rand(base, dtype=torch.float64, device=target, generator=generator)
                if antithetic:
                    zv = torch.cat((zv0, -zv0))
                    zs = torch.cat((zs0, -zs0))
                    uniform = torch.cat((u0, 1.0 - u0))
                else:
                    zv, zs, uniform = zv0, zs0, u0
                if scheme == "full_truncation":
                    positive = torch.clamp_min(variance, 0.0)
                    log_s += (rate - dividend_yield - 0.5 * positive) * dt + torch.sqrt(
                        positive
                    ) * sqrt_dt * (params.rho * zv + rho_orth * zs)
                    variance = torch.clamp_min(
                        variance
                        + params.kappa * (params.theta - positive) * dt
                        + params.sigma_v * torch.sqrt(positive) * sqrt_dt * zv,
                        0.0,
                    )
                else:
                    old_v = variance
                    exp_kdt = np.exp(-params.kappa * dt)
                    mean = params.theta + (old_v - params.theta) * exp_kdt
                    variance_of_v = old_v * params.sigma_v**2 * exp_kdt * (
                        1.0 - exp_kdt
                    ) / params.kappa + params.theta * params.sigma_v**2 * (1.0 - exp_kdt) ** 2 / (
                        2.0 * params.kappa
                    )
                    mean = torch.clamp_min(mean, torch.finfo(torch.float64).tiny)
                    psi = variance_of_v / (mean * mean)
                    safe_psi = torch.clamp_min(psi, torch.finfo(torch.float64).eps)
                    two_over = 2.0 / safe_psi
                    b2 = (
                        two_over
                        - 1.0
                        + torch.sqrt(torch.clamp_min(two_over * (two_over - 1.0), 0.0))
                    )
                    a = mean / (1.0 + b2)
                    quadratic_value = a * (torch.sqrt(torch.clamp_min(b2, 0.0)) + zv) ** 2
                    p0 = torch.clamp((psi - 1.0) / (psi + 1.0), 0.0, 1.0 - 1.0e-14)
                    beta = (1.0 - p0) / mean
                    safe_u = torch.clamp(uniform, 1.0e-14, 1.0 - 1.0e-14)
                    exponential_value = torch.where(
                        safe_u <= p0,
                        torch.zeros_like(safe_u),
                        torch.log((1.0 - p0) / (1.0 - safe_u)) / beta,
                    )
                    new_v = torch.where(psi <= 1.5, quadratic_value, exponential_value)
                    k1 = 0.5 * dt * (params.kappa * params.rho / params.sigma_v - 0.5) - (
                        params.rho / params.sigma_v
                    )
                    k2 = 0.5 * dt * (params.kappa * params.rho / params.sigma_v - 0.5) + (
                        params.rho / params.sigma_v
                    )
                    k34 = 0.5 * dt * (1.0 - params.rho**2)
                    if scheme == "qe_m":
                        mgf_argument = k2 + 0.5 * k34
                        quadratic_denom = 1.0 - 2.0 * mgf_argument * a
                        exponential_denom = beta - mgf_argument
                        quadratic = psi <= 1.5
                        if bool(torch.any(quadratic & (quadratic_denom <= 0.0))) or bool(
                            torch.any((~quadratic) & (exponential_denom <= 0.0))
                        ):
                            raise FloatingPointError(
                                "QE-M conditional MGF does not exist for this step"
                            )
                        log_quadratic = (
                            -0.5 * torch.log(quadratic_denom)
                            + mgf_argument * a * b2 / quadratic_denom
                        )
                        exponential_mgf = p0 + (1.0 - p0) * beta / exponential_denom
                        log_mgf = torch.where(quadratic, log_quadratic, torch.log(exponential_mgf))
                        # Torch is optional and imported lazily; Any here is a
                        # narrow local union between scalar and Tensor drift.
                        k0_term_torch: Any = -(k1 + 0.5 * k34) * old_v - log_mgf
                    else:
                        k0_term_torch = (
                            -params.rho * params.kappa * params.theta * dt / params.sigma_v
                        )
                    log_s += (
                        (rate - dividend_yield) * dt
                        + k0_term_torch
                        + k1 * old_v
                        + k2 * new_v
                        + torch.sqrt(torch.clamp_min(k34 * (old_v + new_v), 0.0)) * zs
                    )
                    variance = new_v
            terminal_all = torch.exp(log_s)
            if option_type == "call":
                payoff_all = torch.clamp_min(terminal_all - strike, 0.0)
            elif option_type == "put":
                payoff_all = torch.clamp_min(strike - terminal_all, 0.0)
            else:
                raise ValueError("option_type must be call or put")
            payoff_all *= np.exp(-rate * maturity)
            if antithetic:
                payoff = 0.5 * (payoff_all[:base] + payoff_all[base : 2 * base])
                terminal = 0.5 * (terminal_all[:base] + terminal_all[base : 2 * base])
                independent_count = base
            else:
                payoff = payoff_all[:count]
                terminal = terminal_all[:count]
                independent_count = count
            if control_variate:
                control = np.exp(-rate * maturity) * terminal
                centered_control = control - spot * np.exp(-dividend_yield * maturity)
                denom = torch.sum(centered_control * centered_control)
                if float(denom) > 0.0:
                    beta = torch.sum((payoff - torch.mean(payoff)) * centered_control) / denom
                    payoff = payoff - beta * centered_control
            sum_x += float(torch.sum(payoff).cpu())
            sum_x2 += float(torch.sum(payoff * payoff).cpu())
            moment_count += independent_count
            remaining -= count
        except torch.cuda.OutOfMemoryError:
            # Clear every potentially live chunk tensor before releasing the
            # allocator cache; otherwise halving the next chunk would retain
            # references from the failed attempt and could repeatedly OOM.
            log_s = variance = None  # type: ignore[assignment]
            zv0 = zs0 = u0 = zv = zs = uniform = None  # type: ignore[assignment]
            positive = old_v = mean = variance_of_v = None  # type: ignore[assignment]
            psi = safe_psi = two_over = b2 = a = None  # type: ignore[assignment]
            quadratic_value = p0 = beta = safe_u = None  # type: ignore[assignment]
            exponential_value = new_v = terminal_all = payoff_all = None  # type: ignore[assignment]
            terminal = payoff = None  # type: ignore[assignment]
            control = centered_control = denom = None
            torch.cuda.empty_cache()
            active_chunk //= 2
            oom_retries += 1
            if active_chunk < 1_024:
                raise RuntimeError("CUDA OOM persisted below the safe minimum path chunk") from None

    mean = sum_x / moment_count
    sample_variance = max((sum_x2 - moment_count * mean * mean) / (moment_count - 1), 0.0)
    stderr = float(np.sqrt(sample_variance / moment_count))
    z = NormalDist().inv_cdf(0.975)
    return MonteCarloEstimate(
        price=mean,
        standard_error=stderr,
        confidence_low=mean - z * stderr,
        confidence_high=mean + z * stderr,
        paths=paths,
        steps=steps,
        scheme=scheme,
        backend=str(target),
        antithetic=antithetic,
        control_variate=control_variate,
        seed=seed,
        oom_retries=oom_retries,
    )
