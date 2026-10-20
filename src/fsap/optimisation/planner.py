"""Campaign assortment planning: which offered articles to list and how many units to buy.

For every candidate and every allowed depth (a share of the brand's lot) the expected
contribution is computed from the forecast distribution:

    E[sold] = sum_{j < q} P(D > j)
    profit(q) = E[sold] * (price * (1 - r) - fulfilment - r * handling + r * salvage * cost)
                - q * cost + (q - E[sold]) * salvage * cost

A mixed-integer program (HiGHS via scipy) then picks at most one depth per article to maximise
total expected contribution subject to the campaign's slots, buy budget, minimum fill and brand
mix. Because demand per article falls as more articles share a campaign, the plan is re-forecast
at the chosen number of articles and solved again until the count settles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, sparse, stats

from fsap.optimisation.rules import PlanningRules

MeanFn = Callable[[pd.DataFrame, int], np.ndarray]  # (candidates, n_live) -> forecast mean per row


def expected_sold(mu: np.ndarray, k: float, q: np.ndarray) -> np.ndarray:
    """E[min(D, q)] for negative-binomial demand, vectorised over rows."""
    q = np.asarray(q, int)
    out = np.zeros(len(mu))
    qmax = int(q.max()) if len(q) else 0
    if qmax == 0:
        return out
    p = k / (k + np.clip(mu, 1e-9, None))
    j = np.arange(qmax)
    sf = stats.nbinom.sf(j[None, :], k, p[:, None])  # P(D > j)
    cum = np.cumsum(sf, axis=1)
    has = q > 0
    out[has] = cum[has, q[has] - 1]
    return out


def unit_margin(c: pd.DataFrame, rules: PlanningRules) -> np.ndarray:
    r = c["return_rate"].to_numpy(float)
    cost = c["unit_cost"].to_numpy(float)
    return (
        c["price"].to_numpy(float) * (1 - r)
        - rules.fulfilment_cost
        - r * rules.return_handling_cost
        + r * rules.salvage_fraction * cost
    )


def options(c: pd.DataFrame, mu: np.ndarray, k: float, rules: PlanningRules) -> pd.DataFrame:
    rows = []
    lot = c["lot_units"].to_numpy(int)
    cost = c["unit_cost"].to_numpy(float)
    m = unit_margin(c, rules)
    for lvl in rules.levels:
        q = np.maximum(np.round(lot * lvl).astype(int), np.minimum(lot, rules.min_units))
        es = expected_sold(mu, k, q)
        profit = es * m - q * cost + (q - es) * rules.salvage_fraction * cost
        rows.append(
            pd.DataFrame(
                {
                    "row": np.arange(len(c)),
                    "level": lvl,
                    "units": q,
                    "expected_sold": es,
                    "expected_profit": profit,
                    "spend": q * cost,
                }
            )
        )
    o = pd.concat(rows, ignore_index=True)
    return o.drop_duplicates(["row", "units"]).reset_index(drop=True)


@dataclass
class Plan:
    campaign_id: str
    status: str
    lines: pd.DataFrame  # one row per candidate with decision and explanation
    n_live: int
    expected_profit: float
    spend: float
    budget: float
    iterations: int


def solve_campaign(
    c: pd.DataFrame, campaign: dict, mean_fn: MeanFn, k: float, rules: PlanningRules, max_iter: int = 4
) -> Plan:
    c = c.reset_index(drop=True)
    slots = int(campaign["slots"])
    budget = rules.budget_share * float((c["unit_cost"] * c["lot_units"]).sum())
    min_fill = rules.brand_min_fill if campaign["kind"] == "brand" else rules.theme_min_fill
    min_n = min(len(c), int(np.ceil(min_fill * slots)))
    n_live = slots
    seen: set[int] = set()
    for it in range(1, max_iter + 1):  # noqa: B007 - `it` is reported after the loop
        mu = mean_fn(c, n_live)
        opt = options(c, mu, k, rules)
        x, status = _milp(c, opt, slots, budget, min_n, campaign, rules)
        chosen = opt[x > 0.5]
        n_new = max(len(chosen), 1)
        if n_new == n_live or n_new in seen:
            break
        seen.add(n_live)
        n_live = n_new
    lines = _explain(c, opt, chosen, mu, k, budget, slots)
    return Plan(
        campaign_id=str(campaign["campaign_id"]),
        status=status,
        lines=lines,
        n_live=len(chosen),
        expected_profit=float(chosen["expected_profit"].sum()),
        spend=float(chosen["spend"].sum()),
        budget=budget,
        iterations=it,
    )


def _milp(
    c: pd.DataFrame, opt: pd.DataFrame, slots: int, budget: float, min_n: int, campaign: dict, rules: PlanningRules
):
    n = len(opt)
    rows = opt["row"].to_numpy()
    cons = []
    # at most one depth per article
    a_one = sparse.csr_matrix((np.ones(n), (rows, np.arange(n))), shape=(len(c), n))
    cons.append(optimize.LinearConstraint(a_one, 0, 1))
    ones = np.ones((1, n))
    cons.append(optimize.LinearConstraint(ones, min_n, slots))
    cons.append(optimize.LinearConstraint(opt["spend"].to_numpy()[None, :], 0, budget))
    if campaign["kind"] == "theme":
        brands = c["brand_id"].to_numpy()[rows]
        cap = max(1, int(np.floor(rules.theme_max_brand_share * slots)))
        for b in np.unique(brands):
            cons.append(optimize.LinearConstraint((brands == b).astype(float)[None, :], 0, cap))
    res = optimize.milp(
        c=-opt["expected_profit"].to_numpy(),
        constraints=cons,
        integrality=np.ones(n),
        bounds=optimize.Bounds(0, 1),
        options={"time_limit": 20, "mip_rel_gap": 1e-4},
    )
    if res.x is None:
        # infeasible (e.g. minimum fill cannot be met within the budget): fall back to the best
        # articles that fit the budget at minimum depth, and say so in the plan status
        x = np.zeros(n)
        smallest = opt.loc[opt.groupby("row")["units"].idxmin()].sort_values("expected_profit", ascending=False)
        spent = 0.0
        for idx, r in smallest.iterrows():
            if spent + r["spend"] <= budget and x.sum() < slots:
                x[idx] = 1
                spent += r["spend"]
        return x, "infeasible_fallback"
    return res.x, "optimal" if res.status == 0 else f"limit:{res.message}"


def _explain(
    c: pd.DataFrame, opt: pd.DataFrame, chosen: pd.DataFrame, mu: np.ndarray, k: float, budget: float, slots: int
) -> pd.DataFrame:
    best = opt.loc[opt.groupby("row")["expected_profit"].idxmax()].set_index("row")
    lines = c[["campaign_id", "article_id", "brand_id", "category", "price", "unit_cost", "lot_units"]].copy()
    lines["forecast_mean"] = mu
    lines["p_sell_out_full_lot"] = stats.nbinom.sf(
        lines["lot_units"].to_numpy() - 1, k, k / (k + np.clip(mu, 1e-9, None))
    )
    ch = chosen.set_index("row")
    lines["units"] = ch["units"].reindex(lines.index).fillna(0).astype(int)
    lines["expected_sold"] = ch["expected_sold"].reindex(lines.index).fillna(0.0)
    lines["expected_profit"] = ch["expected_profit"].reindex(lines.index).fillna(0.0)
    lines["best_possible_profit"] = best["expected_profit"].reindex(lines.index).to_numpy()
    spend = float(chosen["spend"].sum())
    budget_tight = spend > 0.97 * budget
    slots_full = len(chosen) >= slots
    reason = np.where(
        lines["units"] > 0,
        np.where(
            lines["units"] < lines["lot_units"],
            "listed: partial lot (expected profit peaks below the full lot)",
            "listed: full lot",
        ),
        "",
    )
    excluded = lines["units"] == 0
    reason = np.where(
        excluded & (lines["best_possible_profit"] <= 0), "not listed: no depth has positive expected profit", reason
    )
    reason = np.where(
        excluded & (lines["best_possible_profit"] > 0),
        "not listed: "
        + (
            "budget used up by better articles"
            if budget_tight
            else "slots taken by better articles"
            if slots_full
            else "brand share cap or minimum-fill trade-off"
        ),
        reason,
    )
    lines["reason"] = reason
    return lines


# ------------------------------------------------------------------------------ comparison policies
def buyer_heuristic(c: pd.DataFrame, campaign: dict, rules: PlanningRules, signal: np.ndarray) -> pd.DataFrame:
    """How the history was made: rank by discount depth plus the brand's track record, take the
    top `slots` offers at full lot until the budget is spent."""
    c = c.reset_index(drop=True)
    budget = rules.budget_share * float((c["unit_cost"] * c["lot_units"]).sum())
    order = np.argsort(
        -(
            c["discount"].to_numpy()
            + 0.15 * np.nan_to_num(signal, nan=np.nanmean(signal) if np.isfinite(signal).any() else 0)
        )
    )
    units = np.zeros(len(c), int)
    spent, n = 0.0, 0
    for i in order:
        cost = c.at[i, "unit_cost"] * c.at[i, "lot_units"]
        if n < int(campaign["slots"]) and spent + cost <= budget:
            units[i] = c.at[i, "lot_units"]
            spent += cost
            n += 1
    return c.assign(units=units)


def sell_through_rule(c: pd.DataFrame, campaign: dict, rules: PlanningRules) -> pd.DataFrame:
    """A stronger rule of thumb an analyst would build without a model: rank offers by unit
    margin times the brand-category's historical sell-through, and buy the share of the lot that
    history says will sell (plus a 15-point cushion), until slots or budget run out."""
    c = c.reset_index(drop=True)
    budget = rules.budget_share * float((c["unit_cost"] * c["lot_units"]).sum())
    st = (
        c["brand_cat_sell_through"]
        .fillna(c["brand_sell_through"])
        .fillna(c["cat_sell_through"])
        .fillna(0.5)
        .to_numpy(float)
    )
    margin = unit_margin(c, rules) - c["unit_cost"].to_numpy(float) * (1 - rules.salvage_fraction)
    order = np.argsort(-(st * margin))
    levels = np.array(rules.levels)
    units = np.zeros(len(c), int)
    spent, n = 0.0, 0
    for i in order:
        if st[i] * margin[i] <= 0:
            break
        share = levels[np.argmin(np.abs(levels - min(1.0, st[i] + 0.15)))]
        q = max(int(round(c.at[i, "lot_units"] * share)), min(int(c.at[i, "lot_units"]), rules.min_units))
        cost = c.at[i, "unit_cost"] * q
        if n < int(campaign["slots"]) and spent + cost <= budget:
            units[i] = q
            spent += cost
            n += 1
    return c.assign(units=units)
