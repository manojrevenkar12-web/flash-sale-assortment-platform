import numpy as np
import pandas as pd
import pytest

from fsap.optimisation.planner import buyer_heuristic, expected_sold, sell_through_rule, solve_campaign
from fsap.optimisation.rules import PlanningRules

RULES = PlanningRules.load("conf/planning_rules.yaml")


def test_expected_sold_matches_monte_carlo():
    rng = np.random.default_rng(0)
    mu, k = np.array([5.0, 40.0, 120.0]), 3.0
    q = np.array([10, 30, 100])
    exact = expected_sold(mu, k, q)
    sims = np.minimum(rng.negative_binomial(k, k / (k + mu), size=(200000, 3)), q).mean(0)
    assert np.allclose(exact, sims, rtol=0.01)


def candidates(n=40, seed=0, brands=4):
    rng = np.random.default_rng(seed)
    price = rng.uniform(15, 120, n)
    return pd.DataFrame(
        {
            "campaign_id": "C000001",
            "article_id": [f"B00{i % brands}-{i:04d}" for i in range(n)],
            "brand_id": [f"B00{i % brands}" for i in range(n)],
            "category": "sneakers",
            "price": price,
            "rrp": price * 2.5,
            "discount": 0.6,
            "unit_cost": price * rng.uniform(0.3, 0.5, n),
            "lot_units": rng.integers(30, 150, n),
            "return_rate": 0.25,
            "brand_log_demand": rng.normal(3, 0.5, n),
            "brand_cat_sell_through": rng.uniform(0.3, 0.9, n),
            "brand_sell_through": 0.6,
            "cat_sell_through": 0.6,
        }
    )


def mean_fn(mu):
    return lambda c, n: mu[: len(c)] * (30 / n) ** 0.35


@pytest.mark.parametrize("kind", ["brand", "theme"])
def test_plan_respects_every_rule(kind):
    c = candidates()
    mu = np.random.default_rng(1).uniform(2, 120, len(c))
    camp = {"campaign_id": "C000001", "kind": kind, "slots": 20}
    p = solve_campaign(c, camp, mean_fn(mu), 3.0, RULES)
    listed = p.lines[p.lines["units"] > 0]
    assert p.status == "optimal"
    assert len(listed) <= 20
    assert len(listed) >= np.ceil((RULES.brand_min_fill if kind == "brand" else RULES.theme_min_fill) * 20)
    assert (listed["units"] * listed["unit_cost"]).sum() <= p.budget + 1e-6
    assert (listed["units"] <= listed["lot_units"]).all() and (listed["units"] >= RULES.min_units).all()
    if kind == "theme":
        assert listed["brand_id"].value_counts().max() <= np.floor(RULES.theme_max_brand_share * 20)
    assert set(p.lines["reason"].str.split(":").str[0]) <= {"listed", "not listed"}


def test_unprofitable_offers_are_left_out_and_weak_demand_gets_partial_lots():
    c = candidates(n=10)
    c["unit_cost"] = c["price"] * 0.4
    mu = np.full(len(c), 25.0)
    mu[0] = 0.3  # nobody wants it
    c.loc[1, "lot_units"] = 400  # far more than demand
    mu[1] = 150.0
    p = solve_campaign(
        c,
        {"campaign_id": "C000001", "kind": "theme", "slots": 10},
        lambda x, n: mu[: len(x)],
        3.0,
        PlanningRules.from_dict({**_rules_doc(), "fill": {"brand_campaign_min_fill": 0, "theme_campaign_min_fill": 0}}),
    )
    lines = p.lines.set_index("article_id")
    assert lines.iloc[0]["units"] == 0 and "no depth has positive expected profit" in lines.iloc[0]["reason"]
    assert 0 < lines.iloc[1]["units"] < 400


def test_rule_baselines_stay_within_budget_and_slots():
    c = candidates()
    camp = {"campaign_id": "C000001", "kind": "brand", "slots": 15}
    budget = RULES.budget_share * (c["unit_cost"] * c["lot_units"]).sum()
    for plan in (buyer_heuristic(c, camp, RULES, c["brand_log_demand"].to_numpy()), sell_through_rule(c, camp, RULES)):
        listed = plan[plan["units"] > 0]
        assert len(listed) <= 15 and (listed["units"] * listed["unit_cost"]).sum() <= budget + 1e-6


def _rules_doc():
    import yaml

    return yaml.safe_load(open("conf/planning_rules.yaml"))
