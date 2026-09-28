"""Synthetic bid/ask observation model with auditable missingness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from derivguard.synthetic.generation import FloatArray, SyntheticTruth


@dataclass(frozen=True)
class ObservationConfig:
    relative_spread_median: float = 0.04
    relative_spread_log_sigma: float = 0.55
    minimum_spread: float = 0.02
    microprice_noise_fraction: float = 0.20
    base_missing_probability: float = 0.04
    wing_missing_probability: float = 0.18
    sparse_keep_fraction: float = 0.90

    def validate(self) -> None:
        if self.relative_spread_median <= 0.0 or self.minimum_spread <= 0.0:
            raise ValueError("spread parameters must be positive")
        if self.relative_spread_log_sigma < 0.0 or self.microprice_noise_fraction < 0.0:
            raise ValueError("spread dispersion and noise fraction must be non-negative")
        for value in (
            self.base_missing_probability,
            self.wing_missing_probability,
            self.sparse_keep_fraction,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("probabilities must lie in [0,1]")


@dataclass(frozen=True)
class ObservedSurface:
    truth: SyntheticTruth
    bid: FloatArray
    ask: FloatArray
    mid: FloatArray
    spread: FloatArray
    observed: NDArray[np.bool_]
    exclusion_code: NDArray[np.str_]
    seed: int
    config: ObservationConfig

    def __len__(self) -> int:
        return len(self.truth)


def observe_truth(
    truth: SyntheticTruth, config: ObservationConfig | None = None, *, seed: int = 20260928
) -> ObservedSurface:
    """Create realistic-ish quotes while preserving the immutable latent truth.

    Spread and missingness parameters are explicit design assumptions, not
    claims that they were estimated from real SPX data.
    """

    cfg = config or ObservationConfig()
    cfg.validate()
    rng = np.random.default_rng(seed)
    design = truth.design
    log_moneyness = np.abs(np.log(design.strike / design.spot))
    relative_spread = rng.lognormal(
        mean=np.log(cfg.relative_spread_median),
        sigma=cfg.relative_spread_log_sigma,
        size=len(truth),
    )
    spread = np.maximum(cfg.minimum_spread, relative_spread * np.maximum(truth.true_price, 0.25))
    noise = rng.normal(0.0, cfg.microprice_noise_fraction, len(truth)) * spread
    noisy_mid = np.maximum(truth.true_price + noise, 0.0)
    bid = np.maximum(noisy_mid - 0.5 * spread, 0.0)
    ask = np.maximum(noisy_mid + 0.5 * spread, bid)
    mid = 0.5 * (bid + ask)

    wing_scale = np.clip(log_moneyness / 0.35, 0.0, 1.0)
    missing_probability = cfg.base_missing_probability + cfg.wing_missing_probability * wing_scale
    missing = rng.random(len(truth)) < np.clip(missing_probability, 0.0, 1.0)
    sparse_removed = rng.random(len(truth)) > cfg.sparse_keep_fraction
    observed = ~(missing | sparse_removed)
    exclusion = np.full(len(truth), "INCLUDED", dtype="U24")
    exclusion[missing] = "SYNTHETIC_MISSING"
    exclusion[sparse_removed & ~missing] = "SYNTHETIC_SPARSITY"
    bid = np.where(observed, bid, np.nan)
    ask = np.where(observed, ask, np.nan)
    mid = np.where(observed, mid, np.nan)
    return ObservedSurface(truth, bid, ask, mid, spread, observed, exclusion, seed, cfg)
