"""Evaluation for the GenAI curation step, run in CI and after every curation job.

Attributes: a golden set built from the simulator's true attributes. The expected answer for a
field is the true value only if the generator put it into the title (an attribute the supplier left
out cannot be known, so the right answer is null). The category is always stated, sometimes with a
typo. Reported per field: accuracy, precision of stated values,
recall, and hallucination rate (a value asserted when the title says nothing).

Copy: every generated text is validated; an adversarial suite of hand-written bad drafts checks
that the validator itself catches what it should.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fsap.genai.attributes import FIELDS, SCHEMA
from fsap.genai.copy import CampaignFacts, validate
from fsap.genai.llm import ScriptedLLM


def golden_set(truth: pd.DataFrame, n: int = 1000, seed: int = 3) -> pd.DataFrame:
    g = truth.sample(n=min(n, len(truth)), random_state=seed).reset_index(drop=True)
    g["expected_category"] = g["category"]
    for f in ("gender", "colour", "material"):
        g[f"expected_{f}"] = g[f].where(g[f"title_states_{f}"].astype(bool), None)
    return g


def evaluate_extractor(extractor, golden: pd.DataFrame) -> dict:
    preds, lat, events, sources = [], [], [], []
    for row in golden.itertuples():
        e = extractor.extract(row.supplier_title)
        preds.append(e.attributes)
        lat.append(e.latency_ms)
        events += e.events
        sources.append(e.source)
    p = pd.DataFrame(preds)
    out = {"extractor": extractor.name, "titles": len(golden), "fields": {}}
    for f in FIELDS:
        exp = golden[f"expected_{f}"]
        pred = p[f]
        stated = pred.notna()
        expected = exp.notna()
        out["fields"][f] = {
            "accuracy": float(((pred == exp) | (~stated & ~expected)).mean()),
            "precision": float((pred[stated] == exp[stated]).mean()) if stated.any() else None,
            "recall": float((pred[expected] == exp[expected]).mean()) if expected.any() else None,
            "hallucination_rate": float((stated & ~expected).mean()),
            "schema_valid": float(pred[stated].isin(SCHEMA[f]).mean()) if stated.any() else 1.0,
        }
    all_right = np.ones(len(golden), bool)
    for f in FIELDS:
        exp, pred = golden[f"expected_{f}"], p[f]
        all_right &= ((pred == exp) | (pred.isna() & exp.isna())).to_numpy()
    out["all_fields_correct"] = float(all_right.mean())
    out["latency_ms_p95"] = float(np.percentile(lat, 95))
    out["fallback_rate"] = float(np.mean([s == "llm+fallback" for s in sources]))
    out["guardrail_events"] = (
        pd.Series(events, dtype=object).str.split(":").str[0].value_counts().to_dict() if events else {}
    )
    return out


def noisy_model(golden: pd.DataFrame, hallucinate: float = 0.12, malformed: float = 0.03, seed: int = 5) -> ScriptedLLM:
    """A stand-in for a real LLM with its typical failure modes: it reads typos well (it answers
    from the true attributes when the title supports them), but on some titles it invents a
    material or colour the supplier never mentioned, and occasionally returns prose instead of JSON.
    Used to prove the guardrails, not to claim real-model accuracy."""
    rng = np.random.default_rng(seed)
    by_title = {r.supplier_title: r for r in golden.itertuples()}

    def respond(prompt: str) -> str:
        title = prompt.rsplit("Supplier title: ", 1)[1].strip()
        r = by_title[title]
        if rng.random() < malformed:
            return "Sure! The product looks like a nice item."
        ans = {f: getattr(r, f"expected_{f}") for f in FIELDS}
        if rng.random() < hallucinate:
            f = "material" if rng.random() < 0.6 else "colour"
            ans[f] = ans[f] or rng.choice(SCHEMA[f])  # asserts a value the title does not contain
        import json

        return "```json\n" + json.dumps(ans) + "\n```"

    return ScriptedLLM(respond, model="simulated-noisy-llm")


ADVERSARIAL = [
    # (draft, facts overrides, expected violation prefixes)
    (
        {"title": "Nordhaven: bis zu -80%", "teaser": "Starke Preise.", "push": "Nordhaven -80%"},
        {},
        ["discount_overclaimed"],
    ),
    (
        {"title": "Nordhaven: -70%", "teaser": "Alles reduziert.", "push": "Nordhaven -70%"},
        {},
        ["discount_without_up_to"],
    ),
    (
        {"title": "Nordhaven & Velora: bis zu -70%", "teaser": "Zwei Marken.", "push": "Jetzt shoppen"},
        {},
        ["brand_not_in_campaign"],
    ),
    (
        {"title": "Nachhaltige Mode von Nordhaven", "teaser": "Gut für den Planeten.", "push": "Nordhaven nachhaltig"},
        {},
        ["generic_green_claim"],
    ),
    (
        {
            "title": "Nordhaven aus recycelten Materialien",
            "teaser": "Recycelt und reduziert.",
            "push": "Nordhaven recycelt",
        },
        {},
        ["green_claim_unsupported"],
    ),
    ({"title": "Nur heute: Nordhaven", "teaser": "Letzte Chance!", "push": "Nur heute!"}, {}, ["false_urgency"]),
    (
        {"title": "Nordhaven: bis zu -70% auf Sneaker, Boots, Jacken, Mäntel und mehr", "teaser": "x", "push": "y"},
        {},
        ["too_long:title"],
    ),
    ({"title": "Nordhaven: bis zu -70%", "teaser": "", "push": "Nordhaven"}, {}, ["empty:teaser"]),
    # compliant drafts: must produce no violations
    (
        {
            "title": "Nordhaven: bis zu -70%",
            "teaser": "120 Artikel von Nordhaven, 4 Tage lang.",
            "push": "Nordhaven bis zu -70%",
        },
        {},
        [],
    ),
    (
        {"title": "Nordhaven: bis zu -70%", "teaser": "Mit recycelten Materialien.", "push": "Nordhaven bis zu -70%"},
        {"recycled_or_organic_share": 0.6},
        [],
    ),
    (
        {
            "title": "Nordhaven: up to 70% off",
            "teaser": "120 pieces from Nordhaven.",
            "push": "Nordhaven: up to 70% off",
        },
        {"language": "en"},
        [],
    ),
]
BASE_FACTS = CampaignFacts("C000001", "de", ("Nordhaven",), ("sneakers", "boots"), 0.70, 0.40, 120, 4, 0.1)


def evaluate_validator(known_brands: set[str]) -> dict:
    caught, missed, false_alarms = 0, [], []
    for draft, overrides, expected in ADVERSARIAL:
        facts = CampaignFacts(**{**BASE_FACTS.__dict__, **overrides})
        got = validate(draft, facts, known_brands | {"Nordhaven", "Velora"})
        if expected:
            if all(any(g.startswith(e) for g in got) for e in expected):
                caught += 1
            else:
                missed.append((draft["title"], expected, got))
        elif got:
            false_alarms.append((draft["title"], got))
    bad = sum(1 for _, _, e in ADVERSARIAL if e)
    return {
        "bad_drafts": bad,
        "caught": caught,
        "recall": caught / bad,
        "compliant_drafts": len(ADVERSARIAL) - bad,
        "false_alarms": len(false_alarms),
        "missed": missed,
    }
