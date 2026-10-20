"""Decision API (FastAPI), deployed as a Kubernetes service next to the batch jobs.

Buyers and the campaign tool use it to read a campaign's plan with explanations, to re-plan a
campaign when a brand changes its offer ("what if they drop the cost by 10%?"), to normalise a
supplier title, and to check campaign copy before it goes live. Every plan carries the model
version, rules digest and input fingerprint it was made with.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from fsap.contracts import model as contracts
from fsap.contracts.pandas_engine import enforce
from fsap.genai.attributes import LexiconExtractor, LLMExtractor
from fsap.genai.copy import CampaignFacts, validate
from fsap.genai.llm import LLMClient
from fsap.optimisation.job import planning_frame
from fsap.optimisation.planner import solve_campaign
from fsap.optimisation.rules import PlanningRules
from fsap.pipelines.layout import Layout
from fsap.registry.registry import Registry

log = logging.getLogger("fsap.api")


class OfferChange(BaseModel):
    article_id: str
    price: float | None = Field(None, gt=0)
    unit_cost: float | None = Field(None, ge=0)
    lot_units: int | None = Field(None, ge=1, le=5000)
    exclude: bool = False


class ReplanRequest(BaseModel):
    changes: list[OfferChange] = []
    budget_share: float | None = Field(None, gt=0, le=1.5)


class TitleRequest(BaseModel):
    title: str = Field(min_length=2, max_length=300)
    brand: str | None = None


class CopyRequest(BaseModel):
    campaign_id: str
    language: str = Field("de", pattern="^(de|en)$")
    title: str
    teaser: str
    push: str


class State:
    def __init__(self, layout: Layout, rules_path: str):
        self.layout = layout
        self.rules_path = rules_path
        self.registry = Registry(layout.root)
        self.champion = self.registry.champion()
        self.rules = PlanningRules.load(rules_path)
        self.frame, self.campaigns = planning_frame(layout)
        self.campaign_index = self.campaigns.set_index("campaign_id")
        brands = pd.read_json(f"{layout.landing}/brands.jsonl", lines=True)
        self.brand_names = brands.set_index("brand_id")["brand_name"].to_dict()
        self.known_brands = set(self.brand_names.values())
        self.articles = set(pd.read_parquet(layout.silver("articles"), columns=["article_id"])["article_id"])
        client = LLMClient.from_env()
        self.extractor = LLMExtractor(client) if client else LexiconExtractor()
        plans = Path(layout.gold("plans"))
        self.lines = (
            pd.read_parquet(plans / "lines_latest.parquet")
            if (plans / "lines_latest.parquet").exists()
            else pd.DataFrame()
        )
        self.decisions = (
            pd.read_parquet(plans / "decisions_latest.parquet")
            if (plans / "decisions_latest.parquet").exists()
            else pd.DataFrame()
        )
        self.candidate_contract = contracts.load("candidate_offers")


def build_metrics() -> dict:
    reg = CollectorRegistry()
    return {
        "registry": reg,
        "requests": Counter("fsap_http_requests_total", "HTTP requests", ["route", "method", "status"], registry=reg),
        "latency": Histogram(
            "fsap_http_request_seconds",
            "HTTP latency",
            ["route"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
            registry=reg,
        ),
        "plans": Counter("fsap_plans_total", "Plans computed", ["status"], registry=reg),
        "solve": Histogram(
            "fsap_plan_solve_seconds",
            "Time to plan one campaign",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
            registry=reg,
        ),
        "guardrails": Counter("fsap_genai_guardrail_events_total", "GenAI guardrail events", ["kind"], registry=reg),
        "copy_violations": Counter("fsap_copy_violations_total", "Copy rule violations found", ["rule"], registry=reg),
        "model": Gauge("fsap_model_info", "Champion model in use", ["version", "recipe"], registry=reg),
        "report": Gauge("fsap_report_value", "Values from the latest batch reports", ["report", "name"], registry=reg),
        "alerts": Gauge("fsap_monitoring_alerts", "Breached monitoring checks", ["group"], registry=reg),
    }


def refresh_report_gauges(m: dict, layout: Layout) -> None:
    rep = Path(layout.reports)
    ingest = rep / "ingest" / "latest.json"
    if ingest.exists():
        for ds, r in json.loads(ingest.read_text()).items():
            m["report"].labels("ingest", f"quarantine_ratio:{ds}").set(r["quarantine_ratio"])
        m["report"].labels("ingest", "age_seconds").set(time.time() - ingest.stat().st_mtime)
    mon = rep / "monitoring" / "latest.json"
    if mon.exists():
        r = json.loads(mon.read_text())
        for g in ("data", "drift", "outcomes"):
            m["alerts"].labels(g).set(sum(1 for c in r["alerts"] if c["group"] == g))
        for k in ("forecast_wape", "forecast_bias", "sell_through", "realised_profit", "expected_profit"):
            if k in r.get("outcomes", {}):
                m["report"].labels("outcomes", k).set(r["outcomes"][k])
    gen = rep / "genai" / "latest.json"
    if gen.exists():
        r = json.loads(gen.read_text())
        if r.get("copy_violation_rate") is not None:
            m["report"].labels("genai", "copy_violation_rate").set(r["copy_violation_rate"])
        m["report"].labels("genai", "validator_recall").set(r["validator"]["recall"])


def create_app(data_root: str | None = None, rules_path: str = "conf/planning_rules.yaml") -> FastAPI:
    layout = Layout(data_root) if data_root else Layout.from_env()
    metrics = build_metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.s = State(layout, rules_path)
        if app.state.s.champion:
            metrics["model"].labels(app.state.s.champion.version, app.state.s.champion.tags.get("recipe", "")).set(1)
        yield

    app = FastAPI(title="Flash-sale assortment decisions", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        route = request.scope.get("route")
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - logged and turned into a 500 with the request id
            log.exception("unhandled error", extra={"request_id": rid})
            response = JSONResponse({"error": "internal", "request_id": rid}, status_code=500)
        path = getattr(request.scope.get("route"), "path", None) or (route.path if route else "unmatched")
        metrics["requests"].labels(path, request.method, str(response.status_code)).inc()
        metrics["latency"].labels(path).observe(time.perf_counter() - t0)
        response.headers["x-request-id"] = rid
        log.info(
            json.dumps(
                {
                    "request_id": rid,
                    "method": request.method,
                    "route": path,
                    "status": response.status_code,
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                }
            )
        )
        return response

    def st() -> State:
        return app.state.s

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        s = st()
        ok = s.champion is not None and len(s.frame) > 0
        if not ok:
            raise HTTPException(503, "no champion model or no planning data")
        return {"status": "ready", "model_version": s.champion.version, "campaigns": len(s.campaigns)}

    @app.get("/metrics")
    def prom():
        refresh_report_gauges(metrics, layout)
        return PlainTextResponse(generate_latest(metrics["registry"]).decode(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model")
    def model_info():
        s = st()
        if not s.champion:
            raise HTTPException(404, "no champion model")
        return {
            "version": s.champion.version,
            "run_id": s.champion.run_id,
            "tags": s.champion.tags,
            "dispersion": s.champion.model.dispersion,
            "level": s.champion.model.level,
        }

    @app.get("/v1/campaigns")
    def campaigns(limit: int = 50, offset: int = 0):
        s = st()
        dec = s.decisions.set_index("campaign_id") if len(s.decisions) else None
        rows = []
        for c in s.campaigns.iloc[offset : offset + limit].to_dict("records"):
            d = dec.loc[c["campaign_id"]].to_dict() if dec is not None and c["campaign_id"] in dec.index else {}
            rows.append(
                {
                    "campaign_id": c["campaign_id"],
                    "kind": c["kind"],
                    "slots": int(c["slots"]),
                    "start_date": str(c["start_date"]),
                    "planned_articles": d.get("articles"),
                    "expected_profit": d.get("expected_profit"),
                    "decision_id": d.get("decision_id"),
                }
            )
        return {"total": len(s.campaigns), "campaigns": rows}

    @app.get("/v1/campaigns/{campaign_id}/plan")
    def get_plan(campaign_id: str):
        s = st()
        if not len(s.lines) or campaign_id not in set(s.lines["campaign_id"]):
            raise HTTPException(404, f"no plan for {campaign_id}")
        lines = s.lines[s.lines["campaign_id"] == campaign_id]
        dec = s.decisions[s.decisions["campaign_id"] == campaign_id].iloc[0].to_dict()
        return {"decision": _jsonable(dec), "lines": _records(lines.sort_values("expected_profit", ascending=False))}

    @app.post("/v1/campaigns/{campaign_id}/replan")
    def replan(campaign_id: str, req: ReplanRequest):
        s = st()
        if campaign_id not in s.campaign_index.index:
            raise HTTPException(404, f"unknown campaign {campaign_id}")
        if s.champion is None:
            raise HTTPException(503, "no champion model")
        c = s.frame[s.frame["campaign_id"] == campaign_id].copy().set_index("article_id")
        unknown = [ch.article_id for ch in req.changes if ch.article_id not in c.index]
        if unknown:
            raise HTTPException(422, {"error": "articles not offered in this campaign", "articles": unknown})
        for ch in req.changes:
            if ch.exclude:
                c = c.drop(ch.article_id)
                continue
            for k in ("price", "unit_cost", "lot_units"):
                v = getattr(ch, k)
                if v is not None:
                    c.loc[ch.article_id, k] = v
        c = c.reset_index()
        c["discount"] = 1 - c["price"] / c["rrp"]
        c["log_price"], c["log_lot"] = np.log(c["price"]), np.log(c["lot_units"])
        # the changed offers must still meet the candidate_offers contract
        _, bad, report = enforce(
            c[s.candidate_contract.columns], s.candidate_contract, {"articles": s.articles}, raise_on_breach=False
        )
        if len(bad):
            raise HTTPException(
                422,
                {
                    "error": "changed offers break the candidate_offers contract",
                    "violations": bad[["article_id", "violations"]].to_dict("records"),
                },
            )
        rules = s.rules
        if req.budget_share is not None:
            from dataclasses import replace

            rules = replace(rules, budget_share=req.budget_share)
        camp = {"campaign_id": campaign_id, **s.campaign_index.loc[campaign_id, ["kind", "slots"]].to_dict()}
        model = s.champion.model

        def mean_fn(frame: pd.DataFrame, n: int) -> np.ndarray:
            x = frame.copy()
            x["n_live"], x["log_n_live"] = n, np.log(n)
            return model.predict_mean(x)

        t0 = time.perf_counter()
        plan = solve_campaign(c, camp, mean_fn, model.dispersion, rules)
        metrics["solve"].observe(time.perf_counter() - t0)
        metrics["plans"].labels(plan.status).inc()
        return {
            "campaign_id": campaign_id,
            "status": plan.status,
            "articles": plan.n_live,
            "expected_profit": round(plan.expected_profit, 2),
            "spend": round(plan.spend, 2),
            "budget": round(plan.budget, 2),
            "model_version": s.champion.version,
            "rules_digest": rules.digest,
            "lines": _records(plan.lines.sort_values("expected_profit", ascending=False)),
        }

    @app.post("/v1/curation/attributes")
    def attributes(req: TitleRequest):
        e = st().extractor.extract(req.title, req.brand)
        for ev in e.events:
            metrics["guardrails"].labels(ev.split(":")[0]).inc()
        return {"attributes": e.attributes, "evidence": e.evidence, "source": e.source, "events": e.events}

    @app.post("/v1/curation/copy/check")
    def check_copy(req: CopyRequest):
        s = st()
        if not len(s.lines) or req.campaign_id not in set(s.lines["campaign_id"]):
            raise HTTPException(404, f"no plan for {req.campaign_id}")
        g = s.lines[(s.lines["campaign_id"] == req.campaign_id) & (s.lines["units"] > 0)].merge(
            s.frame[["campaign_id", "article_id", "discount"]], on=["campaign_id", "article_id"]
        )
        camp = s.campaign_index.loc[req.campaign_id]
        upcoming = pd.read_json(f"{s.layout.landing}/upcoming_campaigns.jsonl", lines=True).set_index("campaign_id")
        facts = CampaignFacts(
            campaign_id=req.campaign_id,
            language=req.language,
            brands=tuple(s.brand_names[b] for b in g["brand_id"].unique()),
            categories=tuple(g["category"].value_counts().index),
            max_discount=float(g["discount"].max()),
            min_discount=float(g["discount"].min()),
            articles=int(len(g)),
            duration_days=int(upcoming.loc[req.campaign_id, "duration_days"]),
            recycled_or_organic_share=0.0,
        )
        violations = validate(req.model_dump(), facts, s.known_brands)
        for v in violations:
            metrics["copy_violations"].labels(v.split(":")[0]).inc()
        return {
            "ok": not violations,
            "violations": violations,
            "facts": {"max_discount": facts.max_discount, "brands": facts.brands, "kind": camp["kind"]},
        }

    return app


def _jsonable(d: dict) -> dict:
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in d.items()}


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", double_precision=4))


app = None  # created by uvicorn's factory: `uvicorn fsap.serving.app:create_app --factory`
