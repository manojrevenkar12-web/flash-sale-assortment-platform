import numpy as np
import pandas as pd

from fsap.forecasting.backtest import FoldResult, paired_bootstrap, release_gate
from fsap.forecasting.model import FEATURES, GbmForecaster, fit_dispersion


def test_dispersion_mle_recovers_k():
    rng = np.random.default_rng(0)
    mu = rng.uniform(2, 80, 20000)
    k = 3.0
    y = rng.negative_binomial(k, k / (k + mu))
    assert abs(fit_dispersion(y, mu) - k) < 0.15


def toy_frame(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES})
    for c in ("category", "season", "gender", "tier", "kind"):
        f[c] = rng.choice(["a", "b"], n)
    f["demand_est"] = np.exp(1.5 + 0.8 * f["discount"]) * rng.gamma(3, 1 / 3, n)
    f["units_sold"] = np.round(f["demand_est"]).astype(int)
    f["censored"] = False
    return f


def test_gbm_artifact_round_trip_predicts_identically():
    f = toy_frame()
    m = GbmForecaster(rounds=40).fit(f)
    m.calibrate(f.iloc[:500])
    m2 = GbmForecaster.from_artifact(m.to_artifact())
    assert np.allclose(m.predict_mean(f), m2.predict_mean(f))
    assert m2.dispersion == m.dispersion and m2.level == m.level


def preds(mu_scale, seed, n_campaigns=60, per=30):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_campaigns):
        mu = rng.uniform(5, 50, per)
        y = rng.poisson(mu)
        rows.append(
            pd.DataFrame(
                {
                    "campaign_id": f"C{c}",
                    "article_id": [f"A{i}" for i in range(per)],
                    "y_est": y,
                    "y_obs": y,
                    "censored": False,
                    "mu": mu * mu_scale,
                    "lo": mu * mu_scale * 0.5,
                    "hi": mu * mu_scale * 1.5,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_paired_bootstrap_detects_a_worse_model():
    good, worse = preds(1.0, 0), preds(1.0, 0)
    worse["mu"] = worse["mu"] * 1.3
    b = paired_bootstrap(good, worse)
    assert b["difference"] > 0 and b["ci95"][0] > 0


def test_gate_rejects_a_biased_challenger_even_if_it_is_more_accurate():
    cutoff = pd.Timestamp("2025-01-01")
    champ = preds(1.0, 0)
    champ["mu"] = champ["mu"] * 1.5  # a bad champion
    biased = preds(1.0, 0)
    biased["mu"] = biased["mu"] * 1.1  # better, but 10% biased
    results = [FoldResult("champion", cutoff, champ, 3.0), FoldResult("challenger", cutoff, biased, 3.0)] * 3
    g = release_gate(results, "challenger", "champion")
    assert g["checks"]["significantly_better"] and not g["checks"]["unbiased"] and not g["promote"]
