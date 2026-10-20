"""Campaign copy (title, teaser, push notification) in German and English, with guardrails.

Generated text goes in front of millions of members and has legal weight: a discount claim must be
true for the campaign (EU price-indication rules), brands must actually be in the campaign,
environmental claims need substance (EU green-claims rules), and there must be no false urgency.
Every candidate text is checked by `validate`; a failing LLM draft gets one retry with the
violations as feedback, then falls back to the template, and the event is counted.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field

from fsap.genai.llm import LLMClient, LLMError

LIMITS = {"title": 60, "teaser": 160, "push": 40}
CATEGORY_NAMES = {
    "de": {
        "sneakers": "Sneaker",
        "boots": "Boots",
        "sandals": "Sandalen",
        "jeans": "Jeans",
        "dresses": "Kleider",
        "knitwear": "Strick",
        "jackets": "Jacken",
        "coats": "Mäntel",
        "t-shirts": "T-Shirts",
        "shirts": "Hemden & Blusen",
        "swimwear": "Bademode",
        "bags": "Taschen",
        "sportswear": "Sportbekleidung",
        "accessories": "Accessoires",
    },
    "en": {
        "sneakers": "sneakers",
        "boots": "boots",
        "sandals": "sandals",
        "jeans": "jeans",
        "dresses": "dresses",
        "knitwear": "knitwear",
        "jackets": "jackets",
        "coats": "coats",
        "t-shirts": "T-shirts",
        "shirts": "shirts & blouses",
        "swimwear": "swimwear",
        "bags": "bags",
        "sportswear": "sportswear",
        "accessories": "accessories",
    },
}
GREEN_WORDS = re.compile(
    r"\b(nachhaltig\w*|umweltfreundlich\w*|klimaneutral\w*|öko\w*|eco[- ]?friendly|sustainab\w*|green|climate[- ]neutral|planet)\b",
    re.I,
)
SPECIFIC_GREEN = re.compile(r"\b(recycelt\w*|recycled|bio-baumwolle|organic cotton)\b", re.I)
URGENCY = re.compile(
    r"\b(nur heute|heute nur|today only|only today|letzte chance|last chance|nur noch \d+ stunden|ends in \d+ hours)\b",
    re.I,
)
PERCENT = re.compile(r"(-?\d{1,2})\s?%")
UP_TO = re.compile(r"\b(bis zu|up to)\b", re.I)


@dataclass(frozen=True)
class CampaignFacts:
    campaign_id: str
    language: str  # de | en
    brands: tuple[str, ...]
    categories: tuple[str, ...]  # most frequent first
    max_discount: float  # deepest discount of any listed article (0.7 = 70% off RRP)
    min_discount: float
    articles: int
    duration_days: int
    recycled_or_organic_share: float  # share of articles with recycled or organic material


@dataclass
class Copy:
    title: str
    teaser: str
    push: str
    source: str  # template | llm | llm+fallback
    violations: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


def validate(copy: dict, facts: CampaignFacts, known_brands: set[str]) -> list[str]:
    v = []
    for k, limit in LIMITS.items():
        text = copy.get(k) or ""
        if not text.strip():
            v.append(f"empty:{k}")
        elif len(text) > limit:
            v.append(f"too_long:{k}")
    text = " ".join(str(copy.get(k) or "") for k in LIMITS)
    claimed = [abs(int(m)) for m in PERCENT.findall(text)]
    true_max = int(math.floor(facts.max_discount * 100 + 1e-9))
    true_min = int(math.floor(facts.min_discount * 100 + 1e-9))
    for pct in claimed:
        if pct > true_max:
            v.append(f"discount_overclaimed:{pct}>{true_max}")
        elif pct > true_min and not UP_TO.search(text):
            v.append("discount_without_up_to")  # not every article has this discount
    mentioned = {b for b in known_brands if re.search(rf"\b{re.escape(b)}\b", text, re.I)}
    for b in sorted(mentioned - set(facts.brands)):
        v.append(f"brand_not_in_campaign:{b}")
    if GREEN_WORDS.search(text):
        v.append("generic_green_claim")  # vague environmental claims are not allowed at all
    if SPECIFIC_GREEN.search(text) and facts.recycled_or_organic_share < 0.5:
        v.append("green_claim_unsupported")
    if URGENCY.search(text) and facts.duration_days > 1:
        v.append("false_urgency")
    return v


def template_copy(f: CampaignFacts) -> Copy:
    t0 = time.perf_counter()
    pct = int(math.floor(f.max_discount * 20 + 1e-9) * 5)  # round down to 5%
    names = CATEGORY_NAMES[f.language]
    cats = [names[c] for c in f.categories[:2]]
    if f.language == "de":
        subject = f.brands[0] if len(f.brands) == 1 else " & ".join(cats)
        subject = subject[0].upper() + subject[1:]
        title = f"{subject}: bis zu -{pct}%"
        teaser = f"{f.articles} Artikel{' von ' + f.brands[0] if len(f.brands) == 1 else ''} – {', '.join(cats)} – {f.duration_days} Tage lang, solange der Vorrat reicht."
        push = f"{subject[:22]} bis zu -{pct}%"
    else:
        subject = f.brands[0] if len(f.brands) == 1 else " & ".join(cats)
        subject = subject[0].upper() + subject[1:]
        title = f"{subject}: up to {pct}% off"
        teaser = f"{f.articles} pieces{' from ' + f.brands[0] if len(f.brands) == 1 else ''}: {', '.join(cats)}. For {f.duration_days} days, while stocks last."
        push = f"{subject[:22]}: up to {pct}% off"
    return Copy(title[:60], teaser[:160], push[:40], "template", latency_ms=(time.perf_counter() - t0) * 1000)


PROMPT = """Write campaign copy for a fashion outlet's {language_name} app.
Facts (use nothing else): {facts}
Rules: title at most 60 characters, teaser at most 160, push at most 40. A discount may only be
stated as "up to" the maximum ({max_pct}%) or lower. Mention only these brands: {brands}. No
environmental claims. No urgency beyond "while stocks last".
{feedback}Return JSON with keys title, teaser, push."""


class CopyWriter:
    def __init__(self, client: LLMClient | None, known_brands: set[str]):
        self.client = client
        self.known_brands = known_brands

    def write(self, f: CampaignFacts) -> Copy:
        if self.client is None:
            c = template_copy(f)
            c.violations = validate(c.__dict__, f, self.known_brands)
            return c
        t0 = time.perf_counter()
        feedback = ""
        for _attempt in range(2):
            prompt = PROMPT.format(
                language_name="German" if f.language == "de" else "English",
                facts=json.dumps({k: v for k, v in f.__dict__.items() if k != "campaign_id"}, ensure_ascii=False),
                max_pct=int(f.max_discount * 100),
                brands=", ".join(f.brands),
                feedback=feedback,
            )
            try:
                from fsap.genai.attributes import parse_json_object

                draft = parse_json_object(self.client.complete(prompt, max_tokens=300, temperature=0.4))
            except (LLMError, ValueError):
                break
            problems = validate(draft, f, self.known_brands)
            if not problems:
                return Copy(
                    draft["title"], draft["teaser"], draft["push"], "llm", [], (time.perf_counter() - t0) * 1000
                )
            feedback = f"Your previous draft broke these rules: {problems}. Fix them.\n"
        c = template_copy(f)
        c.source = "llm+fallback"
        c.violations = validate(c.__dict__, f, self.known_brands)
        c.latency_ms = (time.perf_counter() - t0) * 1000
        return c
