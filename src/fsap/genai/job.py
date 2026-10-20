"""Curation job: attributes for the articles offered in upcoming campaigns, copy for every planned
campaign, and the evaluation report that monitoring and the release checklist read."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fsap.genai.attributes import LexiconExtractor, LLMExtractor
from fsap.genai.copy import CampaignFacts, CopyWriter
from fsap.genai.evaluate import evaluate_extractor, evaluate_validator, golden_set, noisy_model
from fsap.genai.llm import LLMClient
from fsap.pipelines.layout import Layout

GREEN_MATERIALS = {"recycled polyester", "organic cotton", "recycled polyamide"}


def curate(layout: Layout, golden_size: int = 1000) -> dict:
    client = LLMClient.from_env()
    extractor = LLMExtractor(client) if client else LexiconExtractor()
    articles = pd.read_parquet(layout.silver("articles"))
    brands = pd.read_json(f"{layout.landing}/brands.jsonl", lines=True)
    brand_name = brands.set_index("brand_id")["brand_name"]

    # 1. attributes for every article offered in an upcoming campaign
    cands = pd.read_parquet(layout.silver("candidate_offers"), columns=["campaign_id", "article_id", "discount"])
    todo = articles[articles["article_id"].isin(cands["article_id"])]
    rows = []
    for a in todo.itertuples():
        e = extractor.extract(a.supplier_title, brand_name.get(a.brand_id))
        rows.append(
            {
                "article_id": a.article_id,
                **{f"attr_{k}": v for k, v in e.attributes.items()},
                "source": e.source,
                "events": ";".join(e.events),
            }
        )
    attrs = pd.DataFrame(rows)
    out = Path(layout.gold("curation"))
    out.mkdir(parents=True, exist_ok=True)
    attrs.to_parquet(out / "attributes.parquet")

    # 2. copy for every planned campaign, from the articles the plan actually lists
    plan_path = Path(layout.gold("plans")) / "lines_latest.parquet"
    copies = []
    writer = CopyWriter(client, set(brand_name))
    if plan_path.exists():
        lines = pd.read_parquet(plan_path)
        lines = (
            lines[lines["units"] > 0]
            .merge(cands, on=["campaign_id", "article_id"])
            .merge(attrs, on="article_id", how="left")
        )
        upcoming = pd.read_json(f"{layout.landing}/upcoming_campaigns.jsonl", lines=True).set_index("campaign_id")
        for cid, g in lines.groupby("campaign_id"):
            for lang in ("de", "en"):
                facts = CampaignFacts(
                    campaign_id=cid,
                    language=lang,
                    brands=tuple(brand_name[b] for b in g["brand_id"].value_counts().index[:3])
                    if upcoming.loc[cid, "kind"] == "brand"
                    else tuple(brand_name[b] for b in g["brand_id"].unique()),
                    categories=tuple(g["category"].value_counts().index),
                    max_discount=float(g["discount"].max()),
                    min_discount=float(g["discount"].min()),
                    articles=int(len(g)),
                    duration_days=int(upcoming.loc[cid, "duration_days"]),
                    recycled_or_organic_share=float(g["attr_material"].isin(GREEN_MATERIALS).mean()),
                )
                c = writer.write(facts)
                copies.append(
                    {
                        "campaign_id": cid,
                        "language": lang,
                        "title": c.title,
                        "teaser": c.teaser,
                        "push": c.push,
                        "source": c.source,
                        "violations": ";".join(c.violations),
                    }
                )
    copy_df = pd.DataFrame(copies)
    if len(copy_df):
        copy_df.to_parquet(out / "campaign_copy.parquet")

    # 3. evaluation
    truth_path = Path(layout.truth) / "article_attributes.parquet"
    report: dict = {"extractor": extractor.name, "articles_curated": len(attrs), "copies": len(copy_df)}
    report["copy_violation_rate"] = float((copy_df["violations"] != "").mean()) if len(copy_df) else None
    report["copy_sources"] = copy_df["source"].value_counts().to_dict() if len(copy_df) else {}
    report["validator"] = evaluate_validator(set(brand_name))
    if truth_path.exists():
        golden = golden_set(pd.read_parquet(truth_path), golden_size)
        report["attributes"] = {"lexicon": evaluate_extractor(LexiconExtractor(), golden)}
        report["attributes"]["simulated_noisy_llm_with_guardrails"] = evaluate_extractor(
            LLMExtractor(noisy_model(golden)), golden
        )
        if client:
            report["attributes"]["configured_llm"] = evaluate_extractor(extractor, golden)
    rep = Path(layout.local(f"{layout.reports}/genai"))
    (rep / "latest.json").write_text(
        json.dumps(report, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    )
    return report
