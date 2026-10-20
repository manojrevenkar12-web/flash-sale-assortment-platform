"""The whole pipeline on a one-year world: Spark ingest and features, training with the gate,
planning, curation, monitoring and the API on top. Slow (about two minutes)."""

import json

import pandas as pd
import pytest

from tests.conftest import has_java, run

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not has_java(), reason="needs Java for Spark")]


def test_contracts_quarantine_exactly_the_injected_defects(small_world):
    rep = json.loads((small_world / "reports/ingest/latest.json").read_text())
    injected = pd.read_csv(small_world / "truth/injected_defects.csv")
    assert rep["campaign_offers"]["quarantined"] == len(injected)
    assert all(r["quarantined"] == 0 for k, r in rep.items() if k != "campaign_offers")


def test_features_are_point_in_time(small_world):
    f = pd.read_parquet(small_world / "gold/training_features")
    offers = pd.read_parquet(small_world / "silver/campaign_offers").merge(
        pd.read_parquet(small_world / "silver/campaigns"), on="campaign_id"
    )
    # the first campaigns have no history at all
    first = f[pd.to_datetime(f["start_date"]) == pd.to_datetime(f["start_date"]).min()]
    assert first["brand_log_demand"].isna().all()
    # an article's history count never includes its own campaign or later ones
    row = f.dropna(subset=["article_n"]).sort_values("start_date").iloc[-1]
    earlier = offers[
        (offers["article_id"] == row["article_id"])
        & (pd.to_datetime(offers["end_date"]) < pd.to_datetime(row["start_date"]))
    ]
    assert row["article_n"] <= len(earlier)


def test_training_registers_a_gated_model(small_world):
    rep = json.loads((small_world / "reports/training/latest.json").read_text())
    assert rep["version"] == 1 and set(rep["gate"]["checks"]) >= {"significantly_better", "unbiased", "calibrated"}


def test_plans_are_traceable(small_world):
    dec = pd.read_parquet(small_world / "gold/plans/decisions_latest.parquet")
    lines = pd.read_parquet(small_world / "gold/plans/lines_latest.parquet")
    assert (dec["status"] == "optimal").all()
    assert dec[["model_version", "rules_digest", "input_fingerprint", "contracts"]].notna().all().all()
    assert set(lines["decision_id"]) == set(dec["decision_id"])
    assert lines["reason"].str.len().gt(0).all()


def test_curation_and_monitoring_reports(small_world):
    run("curate", data=small_world)
    run("monitor", data=small_world)
    g = json.loads((small_world / "reports/genai/latest.json").read_text())
    assert g["copy_violation_rate"] == 0.0 and g["validator"]["recall"] == 1.0
    assert g["attributes"]["simulated_noisy_llm_with_guardrails"]["fields"]["material"]["hallucination_rate"] == 0.0
    m = json.loads((small_world / "reports/monitoring/latest.json").read_text())
    assert m["outcomes"]["campaigns"] > 0 and {c["group"] for c in m["checks"]} == {"data", "drift", "outcomes"}


def test_api_serves_plans_and_replans(small_world):
    from fastapi.testclient import TestClient

    from fsap.serving.app import create_app

    with TestClient(create_app(str(small_world))) as c:
        assert c.get("/ready").status_code == 200
        cid = c.get("/v1/campaigns?limit=1").json()["campaigns"][0]["campaign_id"]
        plan = c.get(f"/v1/campaigns/{cid}/plan").json()
        assert plan["decision"]["campaign_id"] == cid and plan["lines"]
        excluded = [ln for ln in plan["lines"] if ln["units"] == 0]
        target = (excluded or plan["lines"])[0]["article_id"]
        r = c.post(f"/v1/campaigns/{cid}/replan", json={"changes": [{"article_id": target, "unit_cost": 0.5}]})
        assert r.status_code == 200 and [ln for ln in r.json()["lines"] if ln["article_id"] == target][0]["units"] > 0
        bad = c.post(f"/v1/campaigns/{cid}/replan", json={"changes": [{"article_id": target, "price": 99999}]})
        assert bad.status_code == 422
        assert c.post(f"/v1/campaigns/{cid}/replan", json={"changes": [{"article_id": "B999-9999"}]}).status_code == 422
        assert (
            c.post("/v1/curation/attributes", json={"title": "Herren Jeans blau Baumwolle"}).json()["attributes"][
                "category"
            ]
            == "jeans"
        )
        v = c.post(
            "/v1/curation/copy/check", json={"campaign_id": cid, "title": "Nur heute -99%", "teaser": "t", "push": "p"}
        ).json()
        assert not v["ok"] and any(x.startswith("discount_overclaimed") for x in v["violations"])
        metrics = c.get("/metrics").text
        assert "fsap_plans_total" in metrics and "fsap_model_info" in metrics
