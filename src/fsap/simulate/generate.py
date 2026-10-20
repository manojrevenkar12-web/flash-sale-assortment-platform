"""Generate campaigns, brand offers and outcomes for the synthetic outlet."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from fsap.simulate.titles import supplier_title
from fsap.simulate.world import (
    CATEGORIES,
    World,
    WorldConfig,
    build_articles,
    build_brands,
    discount_response,
    price_response,
    season_fit,
    traffic_level,
)

BASE_VISITS = 26_000
BASE_CONVERSION = 0.0063
DISCOUNT_RANGE = {"premium": (0.30, 0.60), "mid": (0.40, 0.70), "value": (0.50, 0.78)}
LOT_MEAN = {"premium": 45, "mid": 80, "value": 130}
DAY_DECAY = 0.45  # share of a campaign's demand that arrives on each later day falls by exp(-0.45)


def day_weights(duration: int) -> np.ndarray:
    w = np.exp(-DAY_DECAY * np.arange(duration))
    return w / w.sum()


def generate(cfg: WorldConfig = WorldConfig(), plan_weeks: int = 12) -> World:
    rng = np.random.default_rng(cfg.seed)
    brands = build_brands(rng, cfg)
    articles = build_articles(rng, brands)
    titled = [supplier_title(rng, a) for a in articles.itertuples()]
    articles["supplier_title"] = [t for t, _ in titled]
    for f in ("gender", "colour", "material"):
        articles[f"title_states_{f}"] = [s[f] for _, s in titled]
    cat = CATEGORIES.set_index("category")
    by_brand = {b: g for b, g in articles.groupby("brand_id")}
    by_cat = {c: g for c, g in articles.groupby("category")}
    strength = brands.set_index("brand_id")["strength"]
    tier = brands.set_index("brand_id")["tier"]

    plan_start = cfg.start + timedelta(days=cfg.days - plan_weeks * 7)
    last_brand_campaign: dict[str, pd.Timestamp] = {}
    campaigns, candidates = [], []
    cid = 0
    for day in range(cfg.days):
        today = pd.Timestamp(cfg.start + timedelta(days=day))
        for _ in range(int(rng.integers(cfg.campaigns_per_day[0], cfg.campaigns_per_day[1] + 1))):
            duration = int(rng.integers(3, 7))
            if rng.random() < 0.65:
                kind = "brand"
                p = strength.to_numpy() ** 0.5
                brand_id = str(rng.choice(strength.index, p=p / p.sum()))
                pool = by_brand[brand_id]
                theme = brand_id
                since = (today - last_brand_campaign[brand_id]).days if brand_id in last_brand_campaign else 365
                last_brand_campaign[brand_id] = today
                fatigue = 1.0 - 0.5 * np.exp(-since / cfg.brand_fatigue_days)
                visit_mult = strength[brand_id] ** 0.7 * fatigue
            else:
                kind = "theme"
                fit = season_fit(cat["season"], pd.Series([today] * len(cat)))
                p = cat["popularity"].to_numpy() * fit
                theme = str(rng.choice(cat.index, p=p / p.sum()))
                pool = by_cat[theme]
                brand_id = None
                visit_mult = 1.15
            n_cand = min(len(pool), int(rng.integers(50, 200)))
            cand = pool.sample(n=n_cand, random_state=int(rng.integers(1 << 31)))
            slots = int(np.clip(round(n_cand * rng.uniform(0.45, 0.75)), 20, 140))
            shock = float(np.exp(rng.normal(0.0, 0.35)))  # campaign-level luck, unobservable in advance
            visits = BASE_VISITS * traffic_level(pd.Series([today]))[0] * visit_mult * shock * (duration / 4) ** 0.6
            cid += 1
            campaign_id = f"C{cid:06d}"
            campaigns.append(
                {
                    "campaign_id": campaign_id,
                    "kind": kind,
                    "theme": theme,
                    "brand_id": brand_id,
                    "start_date": today.date(),
                    "end_date": (today + timedelta(days=duration - 1)).date(),
                    "duration_days": duration,
                    "slots": slots,
                    "visits": int(rng.poisson(visits)),
                    "planning": today.date() >= plan_start,
                }
            )
            t = tier.loc[cand["brand_id"]].to_numpy()
            lo = np.array([DISCOUNT_RANGE[x][0] for x in t])
            hi = np.array([DISCOUNT_RANGE[x][1] for x in t])
            discount = np.round(rng.uniform(lo, hi) * 20) / 20  # 5% steps, as brands quote them
            price = np.round(cand["rrp"].to_numpy() * (1 - discount), 0) - 0.05
            cost = np.round(price * rng.uniform(0.25, 0.48, len(cand)), 2)
            lot = np.maximum(10, np.round(rng.lognormal(np.log([LOT_MEAN[x] for x in t]), 0.45))).astype(int)
            score = (
                cand["appeal"].to_numpy()
                * cat.loc[cand["category"], "popularity"].to_numpy()
                * discount_response(discount)
                * price_response(price)
                * season_fit(cand["season"].reset_index(drop=True), pd.Series([today] * len(cand)))
                * strength.loc[cand["brand_id"]].to_numpy() ** (0.3 if kind == "brand" else 0.8)
            )
            candidates.append(
                pd.DataFrame(
                    {
                        "campaign_id": campaign_id,
                        "article_id": cand["article_id"].to_numpy(),
                        "price": price,
                        "rrp": cand["rrp"].to_numpy(),
                        "discount": discount,
                        "unit_cost": cost,
                        "lot_units": lot,
                        "base_rate": visits * BASE_CONVERSION * score,  # true mean demand x n_live ** gamma
                    }
                )
            )

    campaigns = pd.DataFrame(campaigns)
    candidates = pd.concat(candidates, ignore_index=True)
    candidates = candidates.merge(articles[["article_id", "brand_id", "category"]], on="article_id")
    candidates["return_rate"] = cat.loc[candidates["category"], "return_rate"].to_numpy() * np.where(
        candidates["price"] > 100, 1.2, 1.0
    )

    # Historical buyers: rank by discount depth plus a noisy read of brand and article appeal, take the
    # top `slots` offers and buy the whole lot. Planning-period campaigns are left for the policies to plan.
    hist = candidates.merge(campaigns[["campaign_id", "slots", "planning"]], on="campaign_id")
    hist = hist[~hist["planning"]].copy()
    hist["buyer_rank_score"] = hist["discount"] + 0.15 * np.log(hist["base_rate"]) + rng.normal(0, 0.25, len(hist))
    hist["rank"] = hist.groupby("campaign_id")["buyer_rank_score"].rank(ascending=False, method="first")
    live = hist[hist["rank"] <= hist["slots"]].copy()
    live["units_bought"] = live["lot_units"]
    offers, daily = simulate_outcomes(live, campaigns, cfg, rng)

    plan_candidates = candidates[
        candidates["campaign_id"].isin(campaigns.loc[campaigns["planning"], "campaign_id"])
    ].copy()
    return World(
        config=cfg,
        brands=brands,
        articles=articles,
        campaigns=campaigns,
        offers=offers,
        candidates=plan_candidates.reset_index(drop=True),
        daily_sales=daily,
        notes={"plan_start": plan_start.isoformat()},
    )


def simulate_outcomes(
    live: pd.DataFrame,
    campaigns: pd.DataFrame,
    cfg: WorldConfig,
    rng: np.random.Generator,
    demand: np.ndarray | None = None,
    with_daily: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw demand for the live offers, cap it by units bought, and record what the shop observes.

    `demand` may be passed in so that several policies can be compared on the same draws
    (common random numbers).
    """
    live = live.copy()
    n_live = live.groupby("campaign_id")["article_id"].transform("count").to_numpy()
    live["true_mean_demand"] = live["base_rate"].to_numpy() / n_live**cfg.cannibalisation
    if demand is None:
        k = cfg.dispersion
        mu = live["true_mean_demand"].to_numpy()
        demand = rng.negative_binomial(k, k / (k + mu))
    live["true_demand"] = demand
    q = live["units_bought"].to_numpy()
    sold = np.minimum(demand, q)
    live["units_sold"] = sold
    live["units_returned"] = rng.binomial(sold, np.clip(live["return_rate"].to_numpy(), 0, 0.9))

    dur = live["campaign_id"].map(campaigns.set_index("campaign_id")["duration_days"]).to_numpy()
    sold_out_day = np.full(len(live), -1)
    daily_rows = []
    for d in np.unique(dur):
        idx = np.where(dur == d)[0]
        w = day_weights(int(d))
        per_day = np.array([rng.multinomial(int(x), w) for x in demand[idx]]).reshape(len(idx), int(d))
        cum = np.cumsum(per_day, axis=1)
        capped = np.minimum(cum, q[idx, None])
        sales_by_day = np.diff(np.concatenate([np.zeros((len(idx), 1), dtype=int), capped], axis=1), axis=1)
        out = (cum >= q[idx, None]) & (q[idx, None] > 0)
        first = np.where(out.any(axis=1), out.argmax(axis=1), -1)
        sold_out_day[idx] = first
        if with_daily:
            for day in range(int(d)):
                daily_rows.append(
                    pd.DataFrame(
                        {
                            "campaign_id": live["campaign_id"].to_numpy()[idx],
                            "article_id": live["article_id"].to_numpy()[idx],
                            "day": day,
                            "units_sold": sales_by_day[:, day],
                        }
                    )
                )
    live["sold_out_day"] = sold_out_day  # 0-based campaign day on which the last unit sold, -1 if never
    daily = (
        pd.concat(daily_rows, ignore_index=True)
        if daily_rows
        else pd.DataFrame(columns=["campaign_id", "article_id", "day", "units_sold"])
    )
    daily = daily[daily["units_sold"] > 0].reset_index(drop=True)
    return live, daily


def profit(frame: pd.DataFrame, cfg: WorldConfig) -> pd.Series:
    """Contribution per offer: net revenue after returns, fulfilment and return handling,
    minus the cost of units bought, plus salvage for unsold units."""
    net_units = frame["units_sold"] - frame["units_returned"]
    revenue = net_units * frame["price"]
    costs = (
        frame["units_bought"] * frame["unit_cost"]
        + frame["units_sold"] * cfg.fulfilment_cost
        + frame["units_returned"] * cfg.return_handling_cost
    )
    leftover = frame["units_bought"] - frame["units_sold"] + frame["units_returned"]
    salvage = leftover * frame["unit_cost"] * cfg.salvage_fraction
    return revenue - costs + salvage
