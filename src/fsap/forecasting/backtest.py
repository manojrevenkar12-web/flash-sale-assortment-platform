"""Rolling-origin backtest and the release gate every forecaster must pass.

Folds: for each cutoff, train on offers from campaigns that ended before the cutoff and test on
campaigns starting in the following four weeks. Features are point-in-time already, so a test
row only sees history that existed when its campaign was planned.

Metrics that work on real data (observable): WAPE and bias against the uncensored demand
estimate, and the coverage of the 80% interval on offers that did not sell out. Metrics only the
simulator can provide (truth): the same against true demand, including the sold-out offers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fsap.forecasting.model import Forecaster, nb_quantile


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.abs(y - yhat).sum() / max(y.sum(), 1e-9))


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(yhat.sum() / max(y.sum(), 1e-9) - 1)


@dataclass
class FoldResult:
    model: str
    cutoff: pd.Timestamp
    predictions: pd.DataFrame  # campaign_id, article_id, y_est, y_obs, censored, mu, lo, hi, [y_true]
    dispersion: float


def folds(
    frame: pd.DataFrame, n_folds: int = 4, horizon_days: int = 28
) -> list[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    start = pd.to_datetime(frame["start_date"])
    end = pd.to_datetime(frame["end_date"])
    last = start.max()
    cutoffs = [last - pd.Timedelta(days=horizon_days * (n_folds - i)) for i in range(n_folds)]
    out = []
    for c in cutoffs:
        train = frame[end < c]
        test = frame[(start >= c) & (start < c + pd.Timedelta(days=horizon_days))]
        out.append((c, train, test))
    return out


def run_backtest(
    frame: pd.DataFrame,
    factories: dict[str, Callable[[], Forecaster]],
    truth: pd.DataFrame | None = None,
    n_folds: int = 4,
) -> list[FoldResult]:
    results = []
    for cutoff, train, test in folds(frame, n_folds):
        # the last two weeks of training data calibrate the dispersion, out of sample
        tend = pd.to_datetime(train["end_date"])
        fit_part = train[tend < cutoff - pd.Timedelta(days=14)]
        cal_part = train[tend >= cutoff - pd.Timedelta(days=14)]
        for name, make in factories.items():
            m = make().fit(fit_part)
            m.calibrate(cal_part)
            mu = m.predict_mean(test)
            p = (
                test[["campaign_id", "article_id", "demand_est", "units_sold", "censored"]]
                .rename(columns={"demand_est": "y_est", "units_sold": "y_obs"})
                .copy()
            )
            p["mu"] = mu
            p["lo"] = nb_quantile(0.1, mu, m.dispersion)
            p["hi"] = nb_quantile(0.9, mu, m.dispersion)
            if truth is not None:
                p = p.merge(
                    truth[["campaign_id", "article_id", "true_demand"]], on=["campaign_id", "article_id"], how="left"
                )
            results.append(FoldResult(name, cutoff, p, m.dispersion))
    return results


def summarise(results: list[FoldResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        p = r.predictions
        clean = p[~p["censored"].astype(bool)]
        row = {
            "model": r.model,
            "cutoff": r.cutoff.date(),
            "rows": len(p),
            "wape": wape(p["y_est"].to_numpy(), p["mu"].to_numpy()),
            "bias": bias(p["y_est"].to_numpy(), p["mu"].to_numpy()),
            "coverage80_uncensored": float(((clean["y_obs"] >= clean["lo"]) & (clean["y_obs"] <= clean["hi"])).mean()),
            "dispersion": r.dispersion,
        }
        if "true_demand" in p:
            t = p["true_demand"].to_numpy(float)
            row.update(
                {
                    "wape_true": wape(t, p["mu"].to_numpy()),
                    "bias_true": bias(t, p["mu"].to_numpy()),
                    "coverage80_true": float(((t >= p["lo"]) & (t <= p["hi"])).mean()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n: int = 2000, seed: int = 0) -> dict:
    """WAPE(b) - WAPE(a) with a 95% interval, resampling whole campaigns (offers in a campaign
    share traffic, so they are not independent)."""
    key = ["campaign_id", "article_id"]
    m = a[key + ["y_est", "mu"]].merge(b[key + ["mu"]], on=key, suffixes=("_a", "_b"))
    g = (
        m.assign(ea=(m["y_est"] - m["mu_a"]).abs(), eb=(m["y_est"] - m["mu_b"]).abs())
        .groupby("campaign_id")[["ea", "eb", "y_est"]]
        .sum()
    )
    ea, eb, y = g["ea"].to_numpy(), g["eb"].to_numpy(), g["y_est"].to_numpy()
    point = (eb.sum() - ea.sum()) / y.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    diffs = (eb[idx].sum(1) - ea[idx].sum(1)) / y[idx].sum(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(point), "ci95": [float(lo), float(hi)], "campaigns": int(len(g))}


@dataclass(frozen=True)
class GatePolicy:
    max_abs_bias: float = 0.08  # demand planning tolerates a few percent; seasonal shifts show up here first
    coverage_low: float = 0.72
    coverage_high: float = 0.90
    min_folds_better: int = 3  # out of the backtest folds


def release_gate(results: list[FoldResult], challenger: str, champion: str, policy: GatePolicy = GatePolicy()) -> dict:
    """Promote the challenger only if it is significantly better than the champion overall,
    better in most folds, unbiased and calibrated. Returns the decision and every check."""
    s = summarise(results)
    ch, cp = s[s["model"] == challenger], s[s["model"] == champion]
    pool = lambda name: pd.concat([r.predictions for r in results if r.model == name], ignore_index=True)  # noqa: E731
    boot = paired_bootstrap(pool(champion), pool(challenger))
    folds_better = int((ch["wape"].to_numpy() < cp["wape"].to_numpy()).sum())
    pooled = pool(challenger)
    clean = pooled[~pooled["censored"].astype(bool)]
    coverage = float(((clean["y_obs"] >= clean["lo"]) & (clean["y_obs"] <= clean["hi"])).mean())
    b = bias(pooled["y_est"].to_numpy(), pooled["mu"].to_numpy())
    checks = {
        "significantly_better": boot["ci95"][1] < 0,
        "better_in_most_folds": folds_better >= policy.min_folds_better,
        "unbiased": abs(b) <= policy.max_abs_bias,
        "calibrated": policy.coverage_low <= coverage <= policy.coverage_high,
    }
    return {
        "challenger": challenger,
        "champion": champion,
        "promote": all(checks.values()),
        "checks": checks,
        "wape_difference": boot,
        "folds_better": folds_better,
        "bias": b,
        "coverage80_uncensored": coverage,
    }
