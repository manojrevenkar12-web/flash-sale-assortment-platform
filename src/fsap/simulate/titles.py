"""Messy supplier product titles, generated from the true attributes.

Brands deliver product data in their own words: German or English, abbreviated, with sizes and
internal codes mixed in, sometimes missing the colour or material. The GenAI curation step has to
turn these into clean attributes; because the titles are generated from known attributes, its
accuracy can be measured exactly.
"""

from __future__ import annotations

import numpy as np

CATEGORY_WORDS = {
    "sneakers": ["Sneaker", "Sneakers", "Low Top Sneaker", "Turnschuh", "Trainers"],
    "boots": ["Stiefelette", "Boots", "Chelsea Boot", "Stiefel", "Ankle Boot"],
    "sandals": ["Sandale", "Sandals", "Sandalette", "Slides"],
    "jeans": ["Jeans", "Denim", "Slim Fit Jeans", "Straight Leg Jeans", "5-Pocket"],
    "dresses": ["Kleid", "Dress", "Midikleid", "Maxi Dress", "Sommerkleid"],
    "knitwear": ["Strickpullover", "Pullover", "Knit Jumper", "Cardigan", "Strickjacke"],
    "jackets": ["Jacke", "Jacket", "Übergangsjacke", "Bomberjacke", "Steppjacke"],
    "coats": ["Mantel", "Coat", "Wollmantel", "Parka", "Daunenmantel"],
    "t-shirts": ["T-Shirt", "Tee", "Basic Shirt", "Print Tee"],
    "shirts": ["Hemd", "Shirt", "Bluse", "Oxford Shirt", "Leinenhemd"],
    "swimwear": ["Bikini", "Badeanzug", "Swimsuit", "Badeshorts"],
    "bags": ["Tasche", "Handtasche", "Shopper", "Bag", "Rucksack"],
    "sportswear": ["Trainingshose", "Sports Top", "Leggings", "Funktionsshirt", "Track Jacket"],
    "accessories": ["Gürtel", "Belt", "Mütze", "Schal", "Cap"],
}
COLOUR_WORDS = {
    "black": ["schwarz", "black", "blk", "SCHWARZ"],
    "white": ["weiß", "weiss", "white", "wht", "offwhite"],
    "navy": ["navy", "dunkelblau", "marine"],
    "grey": ["grau", "grey", "gray", "anthrazit"],
    "beige": ["beige", "sand", "camel"],
    "brown": ["braun", "brown", "cognac"],
    "green": ["grün", "green", "oliv", "khaki"],
    "red": ["rot", "red", "bordeaux"],
    "blue": ["blau", "blue", "hellblau"],
    "pink": ["rosa", "pink", "altrosa"],
}
MATERIAL_WORDS = {
    "leather": ["Leder", "leather", "Echtleder"],
    "suede": ["Veloursleder", "suede", "Wildleder"],
    "textile": ["Textil", "textile", "Canvas"],
    "recycled polyester": ["recyceltes Polyester", "recycled poly", "rPET"],
    "cotton": ["Baumwolle", "cotton", "BW"],
    "organic cotton": ["Bio-Baumwolle", "organic cotton", "GOTS cotton"],
    "cotton blend": ["Baumwollmix", "cotton blend"],
    "viscose": ["Viskose", "viscose"],
    "linen": ["Leinen", "linen"],
    "polyester": ["Polyester", "poly"],
    "wool": ["Wolle", "wool"],
    "merino wool": ["Merinowolle", "merino"],
    "cashmere blend": ["mit Kaschmir", "cashmere blend"],
    "wool blend": ["Wollmix", "wool blend"],
    "down": ["Daune", "down filled"],
    "polyamide": ["Polyamid", "nylon"],
    "recycled polyamide": ["recyceltes Polyamid", "econyl"],
    "faux leather": ["Lederimitat", "vegan leather", "faux leather"],
    "elastane blend": ["mit Elasthan", "stretch"],
    "metal": ["Metall", "metal"],
}
GENDER_WORDS = {
    "women": ["Damen", "Women", "W", "Ladies"],
    "men": ["Herren", "Men", "M", "Mens"],
    "unisex": ["Unisex", "", "U"],
}


def _typo(rng: np.random.Generator, word: str) -> str:
    if len(word) < 5 or rng.random() > 0.08:
        return word
    i = int(rng.integers(1, len(word) - 1))
    return word[:i] + word[i + 1 :]


def supplier_title(rng: np.random.Generator, a) -> tuple[str, dict[str, bool]]:
    """Returns the title and which attributes it states (the ground truth for evaluation)."""
    parts = []
    stated = {"category": True, "gender": False, "colour": False, "material": False}
    g = rng.choice(GENDER_WORDS[a.gender])
    if g and rng.random() < 0.8:
        parts.append(g)
        stated["gender"] = True
    parts.append(_typo(rng, rng.choice(CATEGORY_WORDS[a.category])))
    if rng.random() < 0.82:
        parts.append(rng.choice(COLOUR_WORDS[a.colour]))
        stated["colour"] = True
    if rng.random() < 0.65:
        parts.append(rng.choice(MATERIAL_WORDS[a.material]))
        stated["material"] = True
    if rng.random() < 0.4:
        parts.append(f"Gr. {rng.choice(['36', '38', '40', 'S', 'M', 'L', '42', '44'])}")
    if rng.random() < 0.35:
        parts.append(f"Art.{int(rng.integers(10000, 99999))}")
    order = rng.permutation(len(parts)) if rng.random() < 0.3 else np.arange(len(parts))
    title = " ".join(parts[i] for i in order)
    return (title.upper() if rng.random() < 0.1 else title), stated
