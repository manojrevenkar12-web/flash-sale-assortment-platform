"""A synthetic outlet marketplace with known ground truth.

Real flash-sale data is not public, so the platform is developed against a simulator whose
demand process is known. That makes it possible to measure what real data never shows: the true
(uncensored) demand behind sold-out articles, the forecast's calibration against it, and the
profit an assortment plan leaves on the table compared with an oracle.

The world:
- fictional brands (tier, strength, category focus, fatigue when they run campaigns too often);
- articles with category, gender, season, colour, material, recommended retail price (RRP) and a
  latent appeal nobody observes;
- daily time-limited campaigns (single-brand or themed multi-brand), 3-6 days long;
- brand offers: lots of an article at a cost price, sold at a discount to RRP;
- demand: campaign traffic x conversion x article appeal x discount response x season fit,
  shared between the articles in the campaign (cannibalisation), negative-binomial noise;
- sales are capped by the units bought (sell-outs censor demand), returns follow a
  category-specific rate, unsold units are liquidated at a fraction of cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

CATEGORIES = pd.DataFrame(
    [
        # name, season, return rate, popularity, rrp low, rrp high (mid tier, EUR)
        ("sneakers", "ALL", 0.24, 1.25, 70, 160),
        ("boots", "AW", 0.22, 0.95, 110, 260),
        ("sandals", "SS", 0.20, 0.70, 45, 120),
        ("jeans", "ALL", 0.38, 1.10, 60, 140),
        ("dresses", "SS", 0.46, 1.00, 50, 180),
        ("knitwear", "AW", 0.30, 0.90, 50, 160),
        ("jackets", "AW", 0.28, 1.05, 90, 280),
        ("coats", "AW", 0.26, 0.80, 140, 380),
        ("t-shirts", "SS", 0.20, 1.15, 20, 60),
        ("shirts", "ALL", 0.32, 0.85, 40, 110),
        ("swimwear", "SS", 0.18, 0.60, 30, 90),
        ("bags", "ALL", 0.12, 0.85, 60, 300),
        ("sportswear", "ALL", 0.22, 1.00, 30, 110),
        ("accessories", "ALL", 0.10, 0.65, 15, 80),
    ],
    columns=["category", "season", "return_rate", "popularity", "rrp_low", "rrp_high"],
)

TIERS = {"value": 0.65, "mid": 1.0, "premium": 1.9}  # RRP multiplier
COLOURS = ["black", "white", "navy", "grey", "beige", "brown", "green", "red", "blue", "pink"]
MATERIALS = {
    "sneakers": ["leather", "suede", "textile", "recycled polyester"],
    "boots": ["leather", "suede"],
    "sandals": ["leather", "textile"],
    "jeans": ["cotton", "organic cotton", "cotton blend"],
    "dresses": ["viscose", "cotton", "linen", "polyester"],
    "knitwear": ["wool", "merino wool", "cotton", "cashmere blend"],
    "jackets": ["polyester", "recycled polyester", "leather", "cotton"],
    "coats": ["wool blend", "polyester", "down"],
    "t-shirts": ["cotton", "organic cotton"],
    "shirts": ["cotton", "linen"],
    "swimwear": ["polyamide", "recycled polyamide"],
    "bags": ["leather", "textile", "faux leather"],
    "sportswear": ["polyester", "recycled polyester", "elastane blend"],
    "accessories": ["leather", "cotton", "metal"],
}
GENDERS = ["women", "men", "unisex"]

_SYL_A = [
    "Nor",
    "Vel",
    "Ar",
    "Lum",
    "Kes",
    "Mar",
    "Sol",
    "Bren",
    "Tal",
    "Os",
    "Fen",
    "Ilo",
    "Dra",
    "Cor",
    "Hal",
    "Zen",
    "Rov",
    "Ely",
]
_SYL_B = [
    "haven",
    "ora",
    "dale",
    "ik",
    "wick",
    "sen",
    "ano",
    "berg",
    "line",
    "voss",
    "mira",
    "ston",
    "ett",
    "ard",
    "ulo",
    "ence",
]


@dataclass(frozen=True)
class WorldConfig:
    seed: int = 7
    start: date = date(2024, 1, 1)
    days: int = 730 + 84  # two years of history plus 12 weeks to plan and evaluate
    n_brands: int = 80
    campaigns_per_day: tuple[int, int] = (3, 6)
    cannibalisation: float = 0.35  # share of traffic per article falls as n_articles ** -gamma
    dispersion: float = 3.0  # negative-binomial size parameter of article demand
    salvage_fraction: float = (
        0.55  # unsold units go to later campaigns or off-price channels, recovering this share of cost
    )
    fulfilment_cost: float = 4.5  # per shipped unit (pick, pack, ship), EUR
    return_handling_cost: float = 3.0  # per returned unit, EUR
    brand_fatigue_days: float = 21.0  # traffic recovers with this time constant after a brand campaign


@dataclass
class World:
    config: WorldConfig
    brands: pd.DataFrame
    articles: pd.DataFrame
    campaigns: pd.DataFrame
    offers: pd.DataFrame  # one row per (campaign, article) that went live, with outcomes
    candidates: pd.DataFrame  # planning period only: all offers buyers could choose from, with true demand params
    daily_sales: pd.DataFrame
    notes: dict = field(default_factory=dict)


def _brand_names(rng: np.random.Generator, n: int) -> list[str]:
    names: set[str] = set()
    while len(names) < n:
        names.add(rng.choice(_SYL_A) + rng.choice(_SYL_B))
    return sorted(names)


def build_brands(rng: np.random.Generator, cfg: WorldConfig) -> pd.DataFrame:
    names = _brand_names(rng, cfg.n_brands)
    tiers = rng.choice(list(TIERS), size=cfg.n_brands, p=[0.35, 0.45, 0.20])
    strength = np.exp(rng.normal(0.0, 0.55, cfg.n_brands))
    rows = []
    for i, name in enumerate(names):
        k = int(rng.integers(2, 6))
        focus = rng.choice(
            CATEGORIES["category"], size=k, replace=False, p=CATEGORIES["popularity"] / CATEGORIES["popularity"].sum()
        )
        rows.append(
            {
                "brand_id": f"B{i:03d}",
                "brand_name": name,
                "tier": tiers[i],
                "strength": float(strength[i]),
                "categories": list(focus),
                "gender_focus": rng.choice(GENDERS, p=[0.5, 0.3, 0.2]),
            }
        )
    return pd.DataFrame(rows)


def build_articles(rng: np.random.Generator, brands: pd.DataFrame) -> pd.DataFrame:
    cat = CATEGORIES.set_index("category")
    rows = []
    for b in brands.itertuples():
        n = int(rng.integers(60, 220))
        for j in range(n):
            c = rng.choice(b.categories)
            low, high = cat.loc[c, "rrp_low"], cat.loc[c, "rrp_high"]
            rrp = float(np.round(rng.uniform(low, high) * TIERS[b.tier] / 5) * 5 - 0.05)
            gender = b.gender_focus if rng.random() < 0.7 else rng.choice(GENDERS)
            rows.append(
                {
                    "article_id": f"{b.brand_id}-{j:04d}",
                    "brand_id": b.brand_id,
                    "category": c,
                    "season": cat.loc[c, "season"] if rng.random() < 0.85 else "ALL",
                    "gender": gender,
                    "colour": rng.choice(COLOURS, p=[0.22, 0.14, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.07, 0.05]),
                    "material": rng.choice(MATERIALS[c]),
                    "rrp": max(rrp, 9.95),
                    "appeal": float(np.exp(rng.normal(0.0, 0.6))),  # latent, never published
                }
            )
    return pd.DataFrame(rows)


def season_fit(season: pd.Series, when: pd.Series) -> np.ndarray:
    """1.0 in season, 0.55 out of season, smooth around the transitions."""
    doy = pd.to_datetime(when).dt.dayofyear.to_numpy()
    # warmth: +1 mid-July, -1 mid-January
    warmth = -np.cos(2 * np.pi * (doy - 15) / 365.25)
    s = season.to_numpy()
    fit = np.ones(len(s))
    fit = np.where(s == "SS", 0.775 + 0.225 * warmth, fit)
    fit = np.where(s == "AW", 0.775 - 0.225 * warmth, fit)
    return fit


def discount_response(discount: np.ndarray) -> np.ndarray:
    """Demand multiplier for the discount off RRP (convex: deep discounts sell disproportionately)."""
    return np.exp(3.2 * (discount - 0.5))


def price_response(price: np.ndarray) -> np.ndarray:
    return (price / 50.0) ** -0.45


def traffic_level(start: pd.Series) -> np.ndarray:
    d = pd.to_datetime(start)
    dow = d.dt.dayofweek.to_numpy()
    dow_mult = np.array([1.0, 0.95, 0.97, 1.02, 1.08, 1.12, 1.18])[dow]
    month = d.dt.month.to_numpy()
    # outlet traffic peaks at end-of-season sales (January, July) and Black Week (November)
    month_mult = np.array([1.25, 1.0, 0.9, 0.95, 1.0, 1.05, 1.25, 1.05, 0.95, 1.0, 1.35, 1.1])[month - 1]
    return dow_mult * month_mult
