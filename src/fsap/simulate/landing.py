"""Write the simulated world as the files upstream systems would deliver.

landing/        what the platform ingests: weekly drops of campaign results and offers,
                the brand and article master data, the candidate offers for upcoming campaigns
truth/          what only the simulator knows (true demand, true attributes); used for
                evaluation, never read by the pipelines or the models

A small share of landing rows is corrupted on purpose (negative units, sales above units bought,
missing prices, duplicates, unknown articles) so that contract enforcement has real work to do.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fsap.simulate.world import World

OFFER_COLUMNS = [
    "campaign_id",
    "article_id",
    "price",
    "rrp",
    "discount",
    "unit_cost",
    "units_bought",
    "units_sold",
    "units_returned",
    "sold_out_day",
]


def corrupt(offers: pd.DataFrame, rng: np.random.Generator, rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject realistic defects; returns the corrupted frame and a log of what was injected."""
    df = offers.copy()
    df["price"] = df["price"].astype(object)
    n = len(df)
    log = []
    kinds = ["negative_units", "sold_above_bought", "missing_price", "unknown_article", "price_above_rrp"]
    picks = rng.choice(n, size=int(n * rate), replace=False)
    for i, idx in enumerate(picks):
        kind = kinds[i % len(kinds)]
        if kind == "negative_units":
            df.iat[idx, df.columns.get_loc("units_sold")] = -int(df.iat[idx, df.columns.get_loc("units_sold")]) - 1
        elif kind == "sold_above_bought":
            df.iat[idx, df.columns.get_loc("units_sold")] = int(df.iat[idx, df.columns.get_loc("units_bought")]) + 5
        elif kind == "missing_price":
            df.iat[idx, df.columns.get_loc("price")] = None
        elif kind == "unknown_article":
            df.iat[idx, df.columns.get_loc("article_id")] = "B999-9999"
        elif kind == "price_above_rrp":
            df.iat[idx, df.columns.get_loc("price")] = float(df.iat[idx, df.columns.get_loc("rrp")]) * 1.5
        log.append({"row": int(idx), "campaign_id": df.iat[idx, 0], "defect": kind})
    dups = df.iloc[rng.choice(n, size=max(1, int(n * rate / 5)), replace=False)]
    log += [{"row": -1, "campaign_id": c, "defect": "duplicate"} for c in dups["campaign_id"]]
    return pd.concat([df, dups], ignore_index=True), pd.DataFrame(log)


def write_landing(world: World, root: Path, defect_rate: float = 0.003, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    landing, truth = root / "landing", root / "truth"
    for d in (landing, truth):
        d.mkdir(parents=True, exist_ok=True)

    brands = world.brands.drop(columns=["strength"])  # brand strength is latent
    brands.to_json(landing / "brands.jsonl", orient="records", lines=True)
    catalog = world.articles.drop(
        columns=["appeal", "colour", "material", "title_states_gender", "title_states_colour", "title_states_material"]
    )  # colour/material only in the messy title
    catalog.to_csv(landing / "articles.csv", index=False)
    world.articles[
        [
            "article_id",
            "category",
            "gender",
            "colour",
            "material",
            "supplier_title",
            "title_states_gender",
            "title_states_colour",
            "title_states_material",
        ]
    ].to_parquet(truth / "article_attributes.parquet")

    camps = world.campaigns.copy()
    hist = camps[~camps["planning"]]
    offers = world.offers.merge(hist[["campaign_id", "start_date"]], on="campaign_id")
    week = pd.to_datetime(offers["start_date"]).dt.to_period("W-SUN").dt.start_time.dt.date
    dirty, defects = corrupt(offers[OFFER_COLUMNS].assign(_week=week.to_numpy()), rng, defect_rate)
    for wk, g in dirty.groupby("_week"):
        part = landing / "offers" / f"week={wk.isoformat()}"
        part.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["_week"]).to_csv(part / "offers.csv", index=False)
    defects.to_csv(truth / "injected_defects.csv", index=False)

    daily = world.daily_sales.merge(hist[["campaign_id", "start_date"]], on="campaign_id")
    dweek = pd.to_datetime(daily["start_date"]).dt.to_period("W-SUN").dt.start_time.dt.date
    for wk, g in daily.assign(_week=dweek.to_numpy()).groupby("_week"):
        part = landing / "daily_sales" / f"week={wk.isoformat()}"
        part.mkdir(parents=True, exist_ok=True)
        g[["campaign_id", "article_id", "day", "units_sold"]].to_csv(part / "daily_sales.csv", index=False)

    hweek = pd.to_datetime(hist["start_date"]).dt.to_period("W-SUN").dt.start_time.dt.date
    for wk, g in hist.assign(_week=hweek.to_numpy()).groupby("_week"):
        part = landing / "campaigns" / f"week={wk.isoformat()}"
        part.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["_week", "planning"]).to_json(
            part / "campaigns.jsonl", orient="records", lines=True, date_format="iso"
        )

    upcoming = camps[camps["planning"]].drop(
        columns=["planning", "visits"]
    )  # traffic is unknown before the campaign runs
    upcoming.to_json(landing / "upcoming_campaigns.jsonl", orient="records", lines=True, date_format="iso")
    cand = world.candidates
    cand.drop(columns=["base_rate", "return_rate", "brand_id", "category"]).to_csv(
        landing / "candidate_offers.csv", index=False
    )
    cand[["campaign_id", "article_id", "base_rate", "return_rate"]].to_parquet(truth / "candidate_truth.parquet")
    world.offers[["campaign_id", "article_id", "true_mean_demand", "true_demand"]].to_parquet(
        truth / "offer_truth.parquet"
    )
    camps[camps["planning"]][["campaign_id", "visits"]].to_parquet(truth / "upcoming_visits.parquet")

    manifest = {
        "seed": world.config.seed,
        "plan_start": world.notes["plan_start"],
        "campaigns_history": int((~camps["planning"]).sum()),
        "campaigns_upcoming": int(camps["planning"].sum()),
        "offers_history": int(len(world.offers)),
        "candidate_offers": int(len(cand)),
        "injected_defects": int(len(defects)),
        "config": {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in world.config.__dict__.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # written last: the delivery is complete
    (landing / "_READY").write_text(
        json.dumps(
            {"campaigns_history": manifest["campaigns_history"], "candidate_offers": manifest["candidate_offers"]}
        )
    )
    return manifest
