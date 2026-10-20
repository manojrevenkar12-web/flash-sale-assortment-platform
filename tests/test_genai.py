import json

import httpx
import pytest

from fsap.genai.attributes import LexiconExtractor, LLMExtractor
from fsap.genai.copy import CampaignFacts, CopyWriter, template_copy, validate
from fsap.genai.evaluate import evaluate_validator
from fsap.genai.llm import LLMClient, LLMError, ScriptedLLM

FACTS = CampaignFacts("C000001", "de", ("Nordhaven",), ("sneakers", "boots"), 0.72, 0.4, 85, 4, 0.0)


@pytest.mark.parametrize(
    "title, expected",
    [
        (
            "Damen Stiefelette schwarz Leder Gr. 38",
            {"category": "boots", "gender": "women", "colour": "black", "material": "leather"},
        ),
        (
            "MEN DUNKELBLAU ART.38387 RUCKSACK LEATHER",
            {"category": "bags", "gender": "men", "colour": "navy", "material": "leather"},
        ),
        (
            "W anthrazit Sommerkleid BW",
            {"category": "dresses", "gender": "women", "colour": "grey", "material": "cotton"},
        ),
        ("Strickjace gray wool", {"category": "knitwear", "gender": None, "colour": "grey", "material": "wool"}),
        ("Sneaker grau Gr. M", {"category": "sneakers", "gender": None, "colour": "grey", "material": None}),
    ],
)
def test_lexicon_reads_messy_titles(title, expected):
    assert LexiconExtractor().extract(title).attributes == expected


def test_llm_answers_that_the_title_does_not_support_are_replaced():
    llm = ScriptedLLM(
        lambda p: (
            '```json\n{"category": "sneakers", "gender": "women", "colour": "black", "material": "cashmere blend"}\n```'
        )
    )
    e = LLMExtractor(llm).extract("Damen Sneaker schwarz")
    assert e.attributes == {"category": "sneakers", "gender": "women", "colour": "black", "material": None}
    assert e.events == ["ungrounded:material"] and e.source == "llm+fallback"


def test_out_of_schema_and_malformed_answers_fall_back():
    e = LLMExtractor(ScriptedLLM(lambda p: '{"category": "footwear", "gender": "women"}')).extract(
        "Damen Sneaker schwarz"
    )
    assert "out_of_schema:category" in e.events and e.attributes["category"] is None
    e2 = LLMExtractor(ScriptedLLM(lambda p: "I think these are sneakers.")).extract("Damen Sneaker schwarz")
    assert e2.source == "llm+fallback" and e2.attributes["category"] == "sneakers"
    e3 = LLMExtractor(ScriptedLLM(lambda p: LLMError("timeout"))).extract("Damen Sneaker schwarz")
    assert e3.events == ["llm_failed:LLMError"]


def test_prompt_contains_schema_and_title():
    llm = ScriptedLLM(lambda p: '{"category": "sneakers"}')
    LLMExtractor(llm).extract("Herren Sneaker", "Nordhaven")
    assert "Supplier title: Herren Sneaker" in llm.prompts[0] and '"recycled polyester"' in llm.prompts[0]


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_http_client_formats_requests_and_retries_overload(provider):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(529 if provider == "anthropic" else 503, json={"error": "overloaded"})
        if provider == "anthropic":
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "hello"}],
                    "usage": {"input_tokens": 12, "output_tokens": 3},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    c = LLMClient(provider, "some-model", "secret", transport=httpx.MockTransport(handler))
    assert c.complete("hi") == "hello"
    assert len(calls) == 2 and c.usage.input_tokens == 12
    body = json.loads(calls[-1].content)
    assert body["model"] == "some-model" and body["messages"][0]["content"] == "hi"
    if provider == "anthropic":
        assert calls[-1].headers["x-api-key"] == "secret" and calls[-1].url.path == "/v1/messages"
    else:
        assert calls[-1].headers["authorization"] == "Bearer secret" and calls[-1].url.path == "/v1/chat/completions"


def test_client_errors_are_not_retried():
    calls = []
    c = LLMClient(
        "openai",
        "m",
        "k",
        transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(400, json={"error": "bad"})),
    )
    with pytest.raises(LLMError):
        c.complete("hi")
    assert len(calls) == 1


def test_template_copy_is_compliant_in_both_languages():
    for lang in ("de", "en"):
        f = CampaignFacts(**{**FACTS.__dict__, "language": lang})
        c = template_copy(f)
        assert validate(c.__dict__, f, {"Nordhaven", "Velora"}) == []
        assert "70" in c.title  # rounded down to 5%, never up


def test_validator_catches_every_adversarial_draft():
    r = evaluate_validator({"Nordhaven", "Velora"})
    assert r["recall"] == 1.0 and r["false_alarms"] == 0, r["missed"]


def test_llm_copy_is_retried_with_feedback_then_falls_back():
    drafts = iter(
        [
            '{"title": "Nordhaven: -80%", "teaser": "Nur heute!", "push": "Jetzt"}',
            '{"title": "Nordhaven: bis zu -70%", "teaser": "85 Artikel, 4 Tage.", "push": "Nordhaven bis zu -70%"}',
        ]
    )
    llm = ScriptedLLM(lambda p: next(drafts))
    c = CopyWriter(llm, {"Nordhaven"}).write(FACTS)
    assert c.source == "llm" and "broke these rules" in llm.prompts[1]
    always_bad = ScriptedLLM(lambda p: '{"title": "Velora: -90%", "teaser": "x", "push": "y"}')
    c2 = CopyWriter(always_bad, {"Nordhaven", "Velora"}).write(FACTS)
    assert c2.source == "llm+fallback" and c2.violations == []
