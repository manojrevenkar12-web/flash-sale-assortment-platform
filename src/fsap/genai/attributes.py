"""Product curation: turn supplier titles into clean attributes.

Brands send titles like "W anthrazit Sommerkleid BW". The catalogue needs category, gender,
colour and material. Two extractors share one interface:

- `LexiconExtractor`: deterministic, fuzzy-matched vocabulary. Runs offline, in CI, and as the
  fallback whenever the LLM fails a guardrail.
- `LLMExtractor`: any chat model (Anthropic Messages API or an OpenAI-compatible endpoint) asked
  for JSON within a fixed schema. Its output is never trusted as-is: values outside the schema are
  rejected and every attribute must be grounded in a word of the title, otherwise the lexicon
  value (or nothing) is used and the event is counted.
"""

from __future__ import annotations

import difflib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field

from fsap.genai.llm import LLMClient, LLMError
from fsap.simulate.titles import CATEGORY_WORDS, COLOUR_WORDS, GENDER_WORDS, MATERIAL_WORDS

FIELDS = ("category", "gender", "colour", "material")
SCHEMA = {
    "category": sorted(CATEGORY_WORDS),
    "gender": sorted(GENDER_WORDS),
    "colour": sorted(COLOUR_WORDS),
    "material": sorted(MATERIAL_WORDS),
}
VOCAB = {"category": CATEGORY_WORDS, "gender": GENDER_WORDS, "colour": COLOUR_WORDS, "material": MATERIAL_WORDS}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower()).replace("ß", "ss")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def clean_title(title: str) -> str:
    """Normalised title without sizes ("Gr. M") and article numbers, which look like attributes."""
    t = norm(title)
    t = re.sub(r"\bgr\.\s*[a-z0-9]{1,3}\b", " ", t)
    return re.sub(r"\bart\.?\s*\d+\b", " ", t)


def _phrases(field_: str) -> list[tuple[str, str]]:
    out = []
    for value, words in VOCAB[field_].items():
        for w in words:
            if w and len(w) > 1:
                out.append((norm(w), value))
    return sorted(out, key=lambda kv: -len(kv[0]))  # longest phrases first ("recycled poly" before "poly")


PHRASES = {f: _phrases(f) for f in FIELDS}
SINGLE_LETTER_GENDER = {"w": "women", "m": "men", "u": "unisex"}


@dataclass
class Extraction:
    attributes: dict[str, str | None]
    evidence: dict[str, str | None]
    source: str  # lexicon | llm | llm+fallback
    latency_ms: float = 0.0
    events: list[str] = field(default_factory=list)  # guardrail events, e.g. "ungrounded:material"


def evidence_for(field_: str, value: str, title: str) -> str | None:
    """The word(s) in the title that support `value`, if any (exact or one-typo fuzzy match)."""
    t = clean_title(title)
    tokens = re.findall(r"[a-z0-9\-]+", t)
    for phrase, v in PHRASES[field_]:
        if v != value:
            continue
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", t):
            return phrase
        if " " not in phrase and len(phrase) >= 5:
            close = difflib.get_close_matches(phrase, tokens, n=1, cutoff=0.86)
            if close:
                return close[0]
    if field_ == "gender":
        for tok in tokens:
            if SINGLE_LETTER_GENDER.get(tok) == value:
                return tok
    return None


class LexiconExtractor:
    name = "lexicon"

    def extract(self, title: str, brand: str | None = None) -> Extraction:
        t0 = time.perf_counter()
        t = clean_title(title)
        tokens = re.findall(r"[a-z0-9\-]+", t)
        attrs, ev = {}, {}
        for f in FIELDS:
            hit = None
            for phrase, value in PHRASES[f]:
                if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", t):
                    hit = (value, phrase)
                    break
            if hit is None:
                for tok in tokens:
                    if len(tok) >= 5:
                        close = difflib.get_close_matches(
                            tok, [p for p, _ in PHRASES[f] if " " not in p and len(p) >= 5], n=1, cutoff=0.86
                        )
                        if close:
                            hit = (dict(PHRASES[f])[close[0]], tok)
                            break
            if hit is None and f == "gender":
                for tok in tokens:
                    if tok in SINGLE_LETTER_GENDER:
                        hit = (SINGLE_LETTER_GENDER[tok], tok)
                        break
            attrs[f], ev[f] = (hit[0], hit[1]) if hit else (None, None)
        return Extraction(attrs, ev, "lexicon", (time.perf_counter() - t0) * 1000)


PROMPT = """You normalise fashion product data for an online outlet.
Return only a JSON object with the keys category, gender, colour, material.
Allowed values:
{schema}
Use null when the title does not state a value. Never guess a material or colour that the title
does not mention. The title may be German, English or abbreviated.

Brand: {brand}
Supplier title: {title}"""


class LLMExtractor:
    def __init__(self, client: LLMClient, fallback: LexiconExtractor | None = None):
        self.client = client
        self.fallback = fallback or LexiconExtractor()
        self.name = f"llm:{client.model}"

    def extract(self, title: str, brand: str | None = None) -> Extraction:
        t0 = time.perf_counter()
        events: list[str] = []
        prompt = PROMPT.format(schema=json.dumps(SCHEMA), brand=brand or "unknown", title=title)
        try:
            raw = self.client.complete(prompt, max_tokens=120)
            data = parse_json_object(raw)
        except (LLMError, ValueError) as e:
            fb = self.fallback.extract(title, brand)
            fb.source = "llm+fallback"
            fb.events.append(f"llm_failed:{type(e).__name__}")
            fb.latency_ms = (time.perf_counter() - t0) * 1000
            return fb
        lex = None
        attrs, ev = {}, {}
        for f in FIELDS:
            v = data.get(f)
            if v is not None and v not in SCHEMA[f]:
                events.append(f"out_of_schema:{f}")
                v = None
            proof = evidence_for(f, v, title) if v else None
            if v and proof is None:
                events.append(f"ungrounded:{f}")  # the model asserted something the title does not say
                lex = lex or self.fallback.extract(title, brand)
                v, proof = lex.attributes[f], lex.evidence[f]
            attrs[f], ev[f] = v, proof
        source = "llm+fallback" if events else "llm"
        return Extraction(attrs, ev, source, (time.perf_counter() - t0) * 1000, events)


def parse_json_object(text: str) -> dict:
    """The first JSON object in a model response (models sometimes wrap it in prose or fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in response")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("response is not an object")
    return obj
