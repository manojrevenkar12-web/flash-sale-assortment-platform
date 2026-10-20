"""Sell-through forecasting: how many units of an article a campaign will demand.

Every forecaster returns a full distribution (negative binomial: mean and dispersion), because the
assortment optimiser needs the probability of selling each additional unit, not a point estimate.

- `HistoryBaseline`: what a buyer would look at — the article's own past demand, else its brand in
  the category, else the brand, else the category.
- `GbmForecaster`: LightGBM with a Tweedie objective over point-in-time features.

Targets are uncensored demand estimates (see pipelines.spark_jobs.uncensor): sold-out offers
would otherwise teach the model that demand equals the units bought.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

CATEGORICAL = ["category", "season", "gender", "tier", "kind"]
NUMERIC = [
    "discount",
    "log_price",
    "log_rrp",
    "log_lot",
    "log_n_live",
    "duration_days",
    "days_since_brand_campaign",
    "brand_campaigns_30d",
    "brand_log_demand",
    "brand_sell_through",
    "brand_n",
    "brand_cat_log_demand",
    "brand_cat_sell_through",
    "brand_cat_n",
    "cat_log_demand",
    "cat_sell_through",
    "cat_n",
    "article_log_demand",
    "article_sell_through",
    "article_n",
    "traffic_log_vpd",
    "traffic_n",
    "start_dow",
    "start_month",
    "week_sin",
    "week_cos",
]
FEATURES = NUMERIC + CATEGORICAL


def nb_logpmf(y: np.ndarray, mu: np.ndarray, k: float) -> np.ndarray:
    p = k / (k + mu)
    return special.gammaln(y + k) - special.gammaln(k) - special.gammaln(y + 1) + k * np.log(p) + y * np.log1p(-p)


def fit_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Maximum-likelihood negative-binomial size k for counts y with predicted means mu."""
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 0.05, None)
    res = optimize.minimize_scalar(
        lambda lk: -nb_logpmf(y, mu, np.exp(lk)).sum(), bounds=(np.log(0.2), np.log(200)), method="bounded"
    )
    return float(np.exp(res.x))


def nb_quantile(q: float, mu: np.ndarray, k: float) -> np.ndarray:
    mu = np.clip(mu, 1e-6, None)
    return stats.nbinom.ppf(q, k, k / (k + mu))


@dataclass
class Forecaster:
    name: str
    dispersion: float = 2.0
    level: float = 1.0  # multiplicative correction fitted on the most recent weeks
    meta: dict = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> Forecaster:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_mean(self, frame: pd.DataFrame) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def calibrate(self, frame: pd.DataFrame) -> Forecaster:
        """On recent data the model has not been trained on: correct the overall level (demand
        drifts with the season and the campaign mix), then fit the dispersion on offers whose
        demand was observed without censoring."""
        self.level = 1.0
        raw = self.predict_mean(frame)
        self.level = float(np.clip(frame["demand_est"].sum() / raw.sum(), 0.8, 1.25))
        clean = frame[~frame["censored"].astype(bool)]
        self.dispersion = fit_dispersion(clean["units_sold"].to_numpy(), self.predict_mean(clean))
        return self

    def interval(self, frame: pd.DataFrame, level: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        mu = self.predict_mean(frame)
        a = (1 - level) / 2
        return nb_quantile(a, mu, self.dispersion), nb_quantile(1 - a, mu, self.dispersion)


class HistoryBaseline(Forecaster):
    def __init__(self):
        super().__init__("history_baseline")

    def fit(self, train: pd.DataFrame) -> HistoryBaseline:
        self.global_log = float(np.log1p(train["demand_est"]).mean())
        return self

    def predict_mean(self, frame: pd.DataFrame) -> np.ndarray:
        log_d = (
            frame["article_log_demand"]
            .fillna(frame["brand_cat_log_demand"])
            .fillna(frame["brand_log_demand"])
            .fillna(frame["cat_log_demand"])
            .fillna(self.global_log)
        )
        return self.level * np.expm1(log_d.to_numpy(float)).clip(0.1)


class GbmForecaster(Forecaster):
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.3,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 7,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 4,
    }
    rounds = 600

    def __init__(self, rounds: int | None = None):
        super().__init__("lgbm_tweedie")
        self.booster: lgb.Booster | None = None
        self.levels: dict[str, list[str]] = {}
        if rounds:
            self.rounds = rounds

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[FEATURES].copy()
        for c in CATEGORICAL:
            x[c] = pd.Categorical(x[c].astype("string").fillna("NA"), categories=self.levels[c])
        for c in NUMERIC:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)
        return x

    def fit(self, train: pd.DataFrame) -> GbmForecaster:
        self.levels = {
            c: sorted(train[c].astype("string").fillna("NA").unique().tolist()) + ["__unseen__"] for c in CATEGORICAL
        }
        ds = lgb.Dataset(
            self._matrix(train),
            label=train["demand_est"].to_numpy(float),
            categorical_feature=CATEGORICAL,
            free_raw_data=True,
        )
        self.booster = lgb.train(self.params, ds, num_boost_round=self.rounds)
        return self

    def predict_mean(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.booster is not None, "model not fitted"
        return self.level * np.clip(self.booster.predict(self._matrix(frame)), 0.05, None)

    def importance(self) -> dict[str, float]:
        gain = self.booster.feature_importance("gain")
        return dict(sorted(zip(FEATURES, (gain / gain.sum()).round(4).tolist(), strict=True), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------ persistence
    def to_artifact(self) -> dict:
        return {
            "name": self.name,
            "dispersion": self.dispersion,
            "level": self.level,
            "levels": self.levels,
            "features": FEATURES,
            "params": self.params,
            "rounds": self.rounds,
            "booster": self.booster.model_to_string(),
            "meta": self.meta,
        }

    @classmethod
    def from_artifact(cls, art: dict) -> GbmForecaster:
        m = cls(rounds=art["rounds"])
        m.dispersion = art["dispersion"]
        m.level = art["level"]
        m.levels = art["levels"]
        m.meta = art.get("meta", {})
        m.booster = lgb.Booster(model_str=art["booster"])
        if art["features"] != FEATURES:
            raise ValueError("artifact was trained with a different feature list")
        return m


def frame_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    """Stable hash of the training data (row order independent), recorded with every model."""
    h = pd.util.hash_pandas_object(frame[columns].sort_values(columns[:2]).reset_index(drop=True), index=False)
    return hashlib.sha256(h.to_numpy().tobytes()).hexdigest()[:16]


def dump(obj) -> str:
    return json.dumps(obj, default=str)
