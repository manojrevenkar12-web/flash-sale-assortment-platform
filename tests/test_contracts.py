import pandas as pd
import pytest

from fsap.contracts import model as cm
from fsap.contracts.model import (
    ContractBreach,
    ContractError,
    all_contracts,
    breaking_changes,
    check_version_bump,
    parse,
)
from fsap.contracts.pandas_engine import enforce

GOOD = {
    "campaign_id": "C000001",
    "article_id": "B001-0001",
    "price": "20.0",
    "rrp": "50.0",
    "discount": "0.6",
    "unit_cost": "8.0",
    "units_bought": "40",
    "units_sold": "30",
    "units_returned": "5",
    "sold_out_day": "-1",
}


def offers(rows):
    return pd.DataFrame([{**GOOD, **r} for r in rows])


def test_all_shipped_contracts_parse():
    cs = all_contracts()
    assert {"campaign_offers", "campaigns", "articles", "daily_sales", "candidate_offers"} <= set(cs)
    for c in cs.values():
        assert c.owner and c.primary_key


def test_each_defect_is_quarantined_with_its_reason():
    c = cm.load("campaign_offers")
    df = offers(
        [
            {"campaign_id": "C000001"},
            {"campaign_id": "C000002", "units_sold": "-3"},
            {"campaign_id": "C000003", "units_sold": "50"},
            {"campaign_id": "C000004", "price": ""},
            {"campaign_id": "C000005", "article_id": "B999-9999"},
            {"campaign_id": "C000006", "price": "80.0"},
            {"campaign_id": "C000007", "units_bought": "12.5"},
            {"campaign_id": "X1"},
            {"campaign_id": "C000001"},  # duplicate key of the first row
        ]
    )
    valid, bad, rep = enforce(df, c, {"articles": {"B001-0001"}}, raise_on_breach=False)
    reasons = dict(zip(bad["campaign_id"], bad["violations"], strict=True))
    assert len(valid) == 1 and valid.iloc[0]["units_sold"] == 30
    assert "min:units_sold" in reasons["C000002"]
    assert "rule:sold_within_bought" in reasons["C000003"]
    assert "required:price" in reasons["C000004"]
    assert "reference:article_id" in reasons["C000005"]
    assert "rule:price_not_above_rrp" in reasons["C000006"]
    assert "type:units_bought" in reasons["C000007"]
    assert "pattern:campaign_id" in reasons["X1"]
    assert bad.iloc[-1]["violations"] == "duplicate_key"
    assert rep["quarantined"] == 8 and rep["breached"]


def test_breach_stops_the_run():
    c = cm.load("campaign_offers")
    with pytest.raises(ContractBreach) as e:
        enforce(offers([{}, {"units_sold": "-1", "campaign_id": "C000002"}]), c)
    assert e.value.report["quarantine_ratio"] == 0.5


def test_invalid_contracts_are_rejected():
    doc = cm.load("campaign_offers").raw
    with pytest.raises(ContractError):
        parse({**doc, "version": "1.2"})
    with pytest.raises(ContractError):
        parse({**doc, "rules": [{"name": "x", "expr": "units_sold <= nonexistent"}]})
    with pytest.raises(ContractError):
        parse({**doc, "rules": [{"name": "x", "expr": "drop table offers"}]})


def test_breaking_changes_need_a_major_version():
    old = cm.load("campaign_offers")
    fields = [f for f in old.raw["fields"] if f["name"] != "sold_out_day"]
    new = parse({**old.raw, "fields": fields, "version": "1.3.0"})
    assert breaking_changes(old, new) == ["field sold_out_day removed"]
    assert check_version_bump(old, new)
    assert check_version_bump(old, parse({**new.raw, "version": "2.0.0"})) == []
    widened = [{**f, "type": "double"} if f["name"] == "units_bought" else f for f in old.raw["fields"]]
    assert breaking_changes(old, parse({**old.raw, "fields": widened, "version": "1.3.0"})) == []
