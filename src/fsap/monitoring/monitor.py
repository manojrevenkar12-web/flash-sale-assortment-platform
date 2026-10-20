"""Monitoring job: is the data healthy, has the input drifted, and were the last plans right?

Three groups of checks, each with a threshold and a runbook link, written to
reports/monitoring/latest.json (read by the API's /metrics and by Grafana alerts):

- data: quarantine ratio per dataset, age of the last successful ingest
- drift: population stability index (PSI) of key features, scoring vs recent training data
- outcomes: for planned campaigns whose results have arrived, forecast WAPE and bias against
  realised demand, and realised vs expected contribution
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fsap.optimisation.evaluate import default_truth, realise
from fsap.optimisation.rules import PlanningRules
from fsap.pipelines.layout import Layout

DRIFT_FEATURES = [
    "discount",
    "log_price",
    "log_lot",
    "article_log_demand",
    "brand_log_demand",
    "days_since_brand_campaign",
    "traffic_log_vpd",
]
THRESHOLDS = {
    "quarantine_ratio": 0.01,
    "ingest_age_hours": 36,
    "psi": 0.2,
    "forecast_wape": 0.75,
    "forecast_abs_bias": 0.12,
    "profit_shortfall": 0.25,
}
RUNBOOK = "docs/runbooks.md"


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    e = expected[np.isfinite(expected)]
    a = actual[np.isfinite(actual)]
    if len(e) < 50 or len(a) < 50:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(((pa - pe) * np.log(pa / pe)).sum())


def realised_outcomes(layout: Layout, lines: pd.DataFrame, rules: PlanningRules) -> tuple[pd.DataFrame, str]:
    """Outcomes of planned campaigns. In production they arrive with the next weekly offers
    delivery; in the simulation they are drawn from the true demand model."""
    delivered = Path(layout.landing) / "realised_offers.csv"
    if delivered.exists():
        return pd.read_csv(delivered), "delivered"
    truth = default_truth(layout.truth)
    rng = np.random.default_rng(99)
    out = []
    for _, g in lines.groupby("campaign_id"):
        g = g.reset_index(drop=True)
        out.append(
            realise(g, truth, rng.random(len(g)), rng.random(len(g)), rules).assign(
                forecast_mean=g["forecast_mean"].to_numpy(), expected_profit=g["expected_profit"].to_numpy()
            )
        )
    return pd.concat(out, ignore_index=True), "simulated"


def monitor(layout: Layout, rules_path: str = "conf/planning_rules.yaml") -> dict:
    now = datetime.now(UTC)
    checks = []

    def check(group: str, name: str, value: float, limit: float, higher_is_bad: bool = True, detail: str = ""):
        breached = bool(np.isfinite(value) and (value > limit if higher_is_bad else value < limit))
        checks.append(
            {
                "group": group,
                "name": name,
                "value": None if not np.isfinite(value) else round(float(value), 4),
                "threshold": limit,
                "breached": breached,
                "detail": detail,
                "runbook": f"{RUNBOOK}#{group}",
            }
        )

    ingest = Path(layout.reports) / "ingest" / "latest.json"
    if ingest.exists():
        rep = json.loads(ingest.read_text())
        for ds, r in rep.items():
            check(
                "data",
                f"quarantine_ratio:{ds}",
                r["quarantine_ratio"],
                THRESHOLDS["quarantine_ratio"],
                detail=", ".join(r["top_reasons"]),
            )
        age = (now.timestamp() - ingest.stat().st_mtime) / 3600
        check("data", "ingest_age_hours", age, THRESHOLDS["ingest_age_hours"])

    train = pd.read_parquet(layout.gold("training_features"), columns=DRIFT_FEATURES + ["start_date"])
    recent = train[
        pd.to_datetime(train["start_date"]) >= pd.to_datetime(train["start_date"]).max() - pd.Timedelta(days=56)
    ]
    score = pd.read_parquet(layout.gold("scoring_features"), columns=DRIFT_FEATURES)
    for f in DRIFT_FEATURES:
        check("drift", f"psi:{f}", psi(recent[f].to_numpy(float), score[f].to_numpy(float)), THRESHOLDS["psi"])

    plans = Path(layout.gold("plans")) / "lines_latest.parquet"
    outcome = {}
    if plans.exists():
        rules = PlanningRules.load(rules_path)
        lines = pd.read_parquet(plans)
        real, source = realised_outcomes(layout, lines, rules)
        live = real[real["units"] > 0]
        # forecast vs realised demand for listed articles (sold-out articles are censored in
        # production; here the simulator reveals full demand)
        wape = float(np.abs(live["demand"] - live["forecast_mean"]).sum() / max(live["demand"].sum(), 1))
        bias = float(live["forecast_mean"].sum() / max(live["demand"].sum(), 1) - 1)
        shortfall = float(1 - live["profit"].sum() / max(live["expected_profit"].sum(), 1e-9))
        check("outcomes", "forecast_wape", wape, THRESHOLDS["forecast_wape"], detail=source)
        check("outcomes", "forecast_abs_bias", abs(bias), THRESHOLDS["forecast_abs_bias"], detail=f"bias {bias:+.3f}")
        check(
            "outcomes",
            "profit_shortfall",
            shortfall,
            THRESHOLDS["profit_shortfall"],
            detail="realised vs expected contribution",
        )
        outcome = {
            "source": source,
            "campaigns": int(live["campaign_id"].nunique()),
            "listed_articles": int(len(live)),
            "realised_profit": float(live["profit"].sum()),
            "expected_profit": float(live["expected_profit"].sum()),
            "sell_through": float(live["sold"].sum() / live["units"].sum()),
            "forecast_wape": wape,
            "forecast_bias": bias,
        }
    report = {
        "at": now.isoformat(),
        "checks": checks,
        "alerts": [c for c in checks if c["breached"]],
        "outcomes": outcome,
    }
    out = Path(layout.local(f"{layout.reports}/monitoring"))
    (out / "latest.json").write_text(json.dumps(report, indent=2))
    return report
