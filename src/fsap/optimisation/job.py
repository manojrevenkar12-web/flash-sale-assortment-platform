"""Planning job: forecast every candidate offer of the upcoming campaigns with the champion model
and write traceable plans.

Each plan line records the model version, rules version and data it was made with, so a buyer
asking "why did we take 40 units of this?" gets the forecast, the expected profit and the reason.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fsap.contracts.model import all_contracts
from fsap.forecasting.model import Forecaster, HistoryBaseline
from fsap.optimisation.evaluate import default_truth, evaluate, summarise
from fsap.optimisation.planner import solve_campaign
from fsap.optimisation.rules import PlanningRules
from fsap.pipelines.layout import Layout
from fsap.registry.registry import Registry


def return_rates(layout: Layout) -> pd.DataFrame:
    """Observed return rate by category and price band, from history."""
    o = pd.read_parquet(
        layout.silver("campaign_offers"), columns=["article_id", "price", "units_sold", "units_returned"]
    )
    a = pd.read_parquet(layout.silver("articles"), columns=["article_id", "category"])
    o = o.merge(a, on="article_id")
    o["price_band"] = np.where(o["price"] > 100, "high", "normal")
    g = o.groupby(["category", "price_band"])[["units_returned", "units_sold"]].sum()
    return (g["units_returned"] / g["units_sold"]).rename("return_rate").reset_index()


def planning_frame(layout: Layout) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = pd.read_parquet(layout.gold("scoring_features"))
    f = f.rename(columns={"units_bought": "lot_units"})
    f["brand_id"] = f["brand_id_article"]
    f["price_band"] = np.where(f["price"] > 100, "high", "normal")
    f = f.merge(return_rates(layout), on=["category", "price_band"], how="left")
    f["return_rate"] = f["return_rate"].fillna(f["return_rate"].mean())
    camps = (
        f[["campaign_id", "kind", "slots", "start_date"]]
        .drop_duplicates("campaign_id")
        .sort_values(["start_date", "campaign_id"])
    )
    return f.sort_values(["campaign_id", "article_id"]).reset_index(drop=True), camps.reset_index(drop=True)


def _mean_fn(model: Forecaster):
    def fn(c: pd.DataFrame, n: int) -> np.ndarray:
        x = c.copy()
        x["n_live"] = n
        x["log_n_live"] = np.log(n)
        return model.predict_mean(x)

    return fn


def plan(layout: Layout, rules_path: str = "conf/planning_rules.yaml", campaign_ids: list[str] | None = None) -> dict:
    rules = PlanningRules.load(rules_path)
    champ = Registry(layout.root).champion()
    if champ is None:
        raise RuntimeError("no champion model registered; run `fsap train` first")
    frame, camps = planning_frame(layout)
    if campaign_ids:
        camps = camps[camps["campaign_id"].isin(campaign_ids)]
    run_at = datetime.now(UTC)
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(frame[["campaign_id", "article_id", "price", "unit_cost", "lot_units"]], index=False)
        .to_numpy()
        .tobytes()
    ).hexdigest()[:16]
    contracts = {n: c.version for n, c in all_contracts().items()}
    lines, decisions = [], []
    for camp in camps.to_dict("records"):
        c = frame[frame["campaign_id"] == camp["campaign_id"]]
        p = solve_campaign(c, camp, _mean_fn(champ.model), champ.model.dispersion, rules)
        decision_id = str(uuid.uuid4())
        lines.append(p.lines.assign(decision_id=decision_id))
        decisions.append(
            {
                "decision_id": decision_id,
                "campaign_id": p.campaign_id,
                "created_at": run_at.isoformat(),
                "status": p.status,
                "articles": p.n_live,
                "slots": int(camp["slots"]),
                "spend": round(p.spend, 2),
                "budget": round(p.budget, 2),
                "expected_profit": round(p.expected_profit, 2),
                "iterations": p.iterations,
                "model_version": champ.version,
                "model_run_id": champ.run_id,
                "rules_version": rules.version,
                "rules_digest": rules.digest,
                "input_fingerprint": fingerprint,
                "contracts": json.dumps(contracts),
            }
        )
    lines_df = pd.concat(lines, ignore_index=True)
    dec = pd.DataFrame(decisions)
    out = Path(layout.gold("plans"))
    out.mkdir(parents=True, exist_ok=True)
    stamp = run_at.strftime("%Y%m%dT%H%M%S")
    lines_df.to_parquet(out / f"lines_{stamp}.parquet")
    dec.to_parquet(out / f"decisions_{stamp}.parquet")
    lines_df.to_parquet(out / "lines_latest.parquet")
    dec.to_parquet(out / "decisions_latest.parquet")
    return {
        "campaigns": len(dec),
        "statuses": dec["status"].value_counts().to_dict(),
        "expected_profit": round(float(dec["expected_profit"].sum()), 2),
        "spend": round(float(dec["spend"].sum()), 2),
        "model_version": champ.version,
        "rules_digest": rules.digest,
    }


def evaluate_policies(layout: Layout, rules_path: str = "conf/planning_rules.yaml", limit: int | None = None) -> dict:
    rules = PlanningRules.load(rules_path)
    champ = Registry(layout.root).champion()
    if champ is None:
        raise RuntimeError("no champion model registered; run `fsap train` first")
    frame, camps = planning_frame(layout)
    hist = pd.read_parquet(layout.gold("training_features"))
    end = pd.to_datetime(hist["end_date"])
    cut = end.max() - pd.Timedelta(days=14)
    baseline = HistoryBaseline().fit(hist[end < cut])
    baseline.calibrate(hist[end >= cut])
    upcoming = pd.read_json(f"{layout.landing}/upcoming_campaigns.jsonl", lines=True)
    results, lines = evaluate(frame, upcoming, champ.model, baseline, default_truth(layout.truth), rules, limit=limit)
    summary = summarise(results)
    report = {"campaigns": int(results["campaign_id"].nunique()), "model_version": champ.version, "policies": summary}
    out = Path(layout.local(f"{layout.reports}/planning"))
    results.to_csv(out / "policy_results.csv", index=False)
    lines.to_parquet(out / "model_plan_lines.parquet")
    (out / "latest.json").write_text(json.dumps(report, indent=2))
    return report
