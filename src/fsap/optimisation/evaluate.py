"""Evaluate planning policies on the upcoming campaigns against the simulator's true demand.

Every policy is played on the same random demand draws (common random numbers), so differences
between policies are not luck. Policies:

- buyer_heuristic   deepest discounts and strongest brands first, full lots, until the budget is spent
                    (how the simulated history was made)
- sell_through_rule rank by unit margin x historical sell-through, buy the share of the lot history
                    says will sell; the strongest rule without a model, and the reference
- baseline_milp     the optimiser fed with the history baseline forecast
- model_milp        the optimiser fed with the champion forecaster
- oracle_milp       the optimiser fed with the true demand distribution (an upper bound)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from fsap.forecasting.model import Forecaster, HistoryBaseline
from fsap.optimisation.planner import buyer_heuristic, sell_through_rule, solve_campaign
from fsap.optimisation.rules import PlanningRules
from fsap.simulate.world import WorldConfig


@dataclass
class Truth:
    base_rate: pd.Series  # indexed by (campaign_id, article_id)
    return_rate: pd.Series
    gamma: float
    k: float


def realise(
    plan: pd.DataFrame, truth: Truth, u_demand: np.ndarray, u_return: np.ndarray, rules: PlanningRules
) -> pd.DataFrame:
    """Outcome of a plan under true demand. `plan` has one row per candidate with `units`."""
    key = pd.MultiIndex.from_frame(plan[["campaign_id", "article_id"]])
    live = plan["units"].to_numpy() > 0
    n_live = max(int(live.sum()), 1)
    lam = truth.base_rate.reindex(key).to_numpy() / n_live**truth.gamma
    d = stats.nbinom.ppf(u_demand, truth.k, truth.k / (truth.k + lam)).astype(int)
    q = plan["units"].to_numpy()
    sold = np.minimum(d, q)
    r = truth.return_rate.reindex(key).to_numpy()
    ret = stats.binom.ppf(u_return, sold, np.clip(r, 0, 0.95)).astype(int)
    ret = np.where(sold > 0, ret, 0)
    price, cost = plan["price"].to_numpy(), plan["unit_cost"].to_numpy()
    net = (sold - ret) * price
    costs = q * cost + sold * rules.fulfilment_cost + ret * rules.return_handling_cost
    leftover = q - sold + ret
    profit = net - costs + leftover * cost * rules.salvage_fraction
    return pd.DataFrame(
        {
            "campaign_id": plan["campaign_id"].to_numpy(),
            "article_id": plan["article_id"].to_numpy(),
            "units": q,
            "sold": sold,
            "returned": ret,
            "demand": np.where(live, d, 0),
            "profit": np.where(live, profit, 0.0),
            "spend": q * cost,
            "revenue": net,
        }
    )


def evaluate(
    candidates: pd.DataFrame,
    campaigns: pd.DataFrame,
    model: Forecaster,
    baseline: HistoryBaseline,
    truth: Truth,
    rules: PlanningRules,
    seed: int = 123,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-campaign results per policy, per-offer lines of the model plan)."""
    rng = np.random.default_rng(seed)
    cand = candidates.copy()
    cand["u_demand"] = rng.random(len(cand))
    cand["u_return"] = rng.random(len(cand))
    camps = campaigns.set_index("campaign_id")
    ids = list(cand["campaign_id"].drop_duplicates())[: limit or None]

    def model_mean(m: Forecaster):
        def fn(c: pd.DataFrame, n: int) -> np.ndarray:
            f = c.copy()
            f["n_live"] = n
            f["log_n_live"] = np.log(n)
            return m.predict_mean(f)

        return fn

    def oracle_mean(c: pd.DataFrame, n: int) -> np.ndarray:
        key = pd.MultiIndex.from_frame(c[["campaign_id", "article_id"]])
        return truth.base_rate.reindex(key).to_numpy() / n**truth.gamma

    results, lines = [], []
    for cid in ids:
        c = cand[cand["campaign_id"] == cid].reset_index(drop=True)
        meta = {"campaign_id": cid, **camps.loc[cid, ["kind", "slots"]].to_dict()}
        plans = {}
        h = buyer_heuristic(c, meta, rules, c["brand_log_demand"].to_numpy(float))
        plans["buyer_heuristic"] = h
        plans["sell_through_rule"] = sell_through_rule(c, meta, rules)
        for name, fn, k in (
            ("baseline_milp", model_mean(baseline), baseline.dispersion),
            ("model_milp", model_mean(model), model.dispersion),
            ("oracle_milp", oracle_mean, truth.k),
        ):
            p = solve_campaign(c, meta, fn, k, rules)
            plans[name] = c.assign(units=p.lines["units"].to_numpy())
            if name == "model_milp":
                lines.append(p.lines.assign(status=p.status, iterations=p.iterations))
        for name, plan in plans.items():
            out = realise(plan, truth, c["u_demand"].to_numpy(), c["u_return"].to_numpy(), rules)
            results.append(
                {
                    "campaign_id": cid,
                    "kind": meta["kind"],
                    "policy": name,
                    "articles": int((out["units"] > 0).sum()),
                    "units": int(out["units"].sum()),
                    "sold": int(out["sold"].sum()),
                    "spend": float(out["spend"].sum()),
                    "profit": float(out["profit"].sum()),
                    "revenue": float(out["revenue"].sum()),
                }
            )
    return pd.DataFrame(results), pd.concat(lines, ignore_index=True)


def summarise(results: pd.DataFrame, reference: str = "sell_through_rule", n_boot: int = 2000, seed: int = 0) -> dict:
    piv = results.pivot(index="campaign_id", columns="policy", values="profit")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(piv), size=(n_boot, len(piv)))
    out = {}
    for policy, g in results.groupby("policy"):
        units, sold = g["units"].sum(), g["sold"].sum()
        row = {
            "campaigns": int(len(g)),
            "profit": float(g["profit"].sum()),
            "spend": float(g["spend"].sum()),
            "sell_through": float(sold / units) if units else 0.0,
            "return_on_spend": float(g["profit"].sum() / g["spend"].sum()) if g["spend"].sum() else 0.0,
            "articles_per_campaign": float(g["articles"].mean()),
        }
        if policy != reference:
            d = piv[policy].to_numpy() - piv[reference].to_numpy()
            base = piv[reference].to_numpy()
            rel = d[idx].sum(1) / np.abs(base[idx].sum(1))
            row["uplift_vs_" + reference] = float(d.sum() / abs(base.sum()))
            row["uplift_ci95"] = [float(x) for x in np.percentile(rel, [2.5, 97.5])]
            row["campaigns_better"] = float((d > 0).mean())
        out[policy] = row
    if "oracle_milp" in out and "model_milp" in out:
        out["model_milp"]["share_of_oracle_gain"] = (out["model_milp"]["profit"] - out[reference]["profit"]) / max(
            out["oracle_milp"]["profit"] - out[reference]["profit"], 1e-9
        )
    return out


def default_truth(truth_dir: str, cfg: WorldConfig = WorldConfig()) -> Truth:
    t = pd.read_parquet(f"{truth_dir}/candidate_truth.parquet").set_index(["campaign_id", "article_id"])
    return Truth(base_rate=t["base_rate"], return_rate=t["return_rate"], gamma=cfg.cannibalisation, k=cfg.dispersion)
