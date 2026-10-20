from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PlanningRules:
    version: str
    fulfilment_cost: float
    return_handling_cost: float
    salvage_fraction: float
    budget_share: float
    levels: tuple[float, ...]
    min_units: int
    brand_min_fill: float
    theme_min_fill: float
    theme_max_brand_share: float
    digest: str

    @classmethod
    def load(cls, path: str | Path = "conf/planning_rules.yaml") -> PlanningRules:
        doc = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(doc)

    @classmethod
    def from_dict(cls, doc: dict) -> PlanningRules:
        levels = tuple(float(x) for x in doc["depth"]["levels"])
        if not levels or any(not 0 < x <= 1 for x in levels):
            raise ValueError("depth levels must be in (0, 1]")
        r = cls(
            version=str(doc["version"]),
            fulfilment_cost=float(doc["economics"]["fulfilment_cost_per_unit"]),
            return_handling_cost=float(doc["economics"]["return_handling_cost_per_unit"]),
            salvage_fraction=float(doc["economics"]["salvage_fraction"]),
            budget_share=float(doc["budget"]["share_of_offered_cost"]),
            levels=levels,
            min_units=int(doc["depth"]["min_units"]),
            brand_min_fill=float(doc["fill"]["brand_campaign_min_fill"]),
            theme_min_fill=float(doc["fill"]["theme_campaign_min_fill"]),
            theme_max_brand_share=float(doc["mix"]["theme_max_brand_share"]),
            digest=hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:12],
        )
        if not 0 <= r.salvage_fraction < 1:
            raise ValueError("salvage_fraction must be in [0, 1)")
        return r
