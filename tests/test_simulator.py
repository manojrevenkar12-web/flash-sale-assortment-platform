import numpy as np

from fsap.simulate.generate import generate, profit
from fsap.simulate.world import WorldConfig


def small(seed=3):
    return generate(WorldConfig(seed=seed, days=60, n_brands=20), plan_weeks=1)


def test_same_seed_same_world():
    a, b = small(), small()
    assert a.offers[["campaign_id", "article_id", "units_sold"]].equals(
        b.offers[["campaign_id", "article_id", "units_sold"]]
    )
    assert not a.offers["units_sold"].equals(small(seed=4).offers["units_sold"].iloc[: len(a.offers)])


def test_outcomes_respect_physics():
    w = small()
    o = w.offers
    assert (o["units_sold"] <= o["units_bought"]).all()
    assert (o["units_returned"] <= o["units_sold"]).all()
    assert (o["units_sold"] == np.minimum(o["true_demand"], o["units_bought"])).all()
    sold_out = o["sold_out_day"] >= 0
    assert (o.loc[sold_out, "units_sold"] == o.loc[sold_out, "units_bought"]).all()
    assert (o.loc[~sold_out, "true_demand"] < o.loc[~sold_out, "units_bought"]).all()
    daily = w.daily_sales.groupby(["campaign_id", "article_id"])["units_sold"].sum()
    merged = o.set_index(["campaign_id", "article_id"])["units_sold"]
    assert (daily.reindex(merged.index).fillna(0).astype(int) == merged).all()


def test_planning_period_has_candidates_but_no_outcomes():
    w = small()
    planning = set(w.campaigns.loc[w.campaigns["planning"], "campaign_id"])
    assert planning and planning.isdisjoint(set(w.offers["campaign_id"]))
    assert set(w.candidates["campaign_id"]) == planning


def test_economics_are_plausible():
    o = small().offers
    st = o["units_sold"].sum() / o["units_bought"].sum()
    assert 0.35 < st < 0.8
    p = profit(o, WorldConfig())
    assert 0.2 < (p < 0).mean() < 0.7  # plenty of loss-making offers for a planner to avoid
