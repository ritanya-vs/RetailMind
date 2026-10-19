"""Rule-based query parser: pulls structured constraints out of a free-text query.

Handles brand and color (from the catalog lexicon), size and material, price limits, delivery
limits, stock, and "without X" exclusions. It is deterministic and needs no API key. Words that
describe the product (brand, color, size, material) stay in product_terms because they also help
retrieval; price, delivery, stock and exclusion phrases are removed from it.
"""

from __future__ import annotations

import re

from retailmind.query.lexicon import Lexicon, tokens
from retailmind.schemas import Constraints, ParsedQuery

_MONEY = r"(?:\$\s?(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s?(?:dollars?|usd|bucks))"


def _amount(match: re.Match[str], offset: int = 1) -> float:
    raw = match.group(offset) or match.group(offset + 1)
    return float(raw.replace(",", ""))


_RANGE = re.compile(rf"(?<![\w$])(?:between\s+)?{_MONEY}\s*(?:-|to|and)\s*{_MONEY}")
_MAX_WORDS = (
    r"under|below|less than|up to|upto|max(?:imum)?|at most|no more than|within|not more than"
)
_MAX = re.compile(rf"\b(?:{_MAX_WORDS})\s+{_MONEY}")
_MIN = re.compile(rf"\b(?:over|above|more than|at least|min(?:imum)?|starting at)\s+{_MONEY}")

_DAYS = r"(\d+)\s*(?:business\s+)?days?"
_DELIVERY_DAYS = re.compile(
    r"\b(?:(?:deliver(?:ed|y)?|arrive[sd]?|ship(?:ped|s|ping)?|get it|received?)"
    rf"\s+(?:in|within|by)?\s*{_DAYS}"
    rf"|(?:within|in)\s+{_DAYS}\s+(?:delivery|shipping)"
    r"|(\d+)[\s-]*days?\s+(?:delivery|shipping))"
)
_WEEK = r"(?:delivered|arrive[sd]?|get it|ship(?:ped|s)?)\s+(?:within|in)\s+a\s+week"
_DELIVERY_WORDS = (
    (re.compile(r"\bsame[\s-]day\s+(?:delivery|shipping)"), 0),
    (re.compile(r"\b(?:next[\s-]day|overnight)\s+(?:delivery|shipping)"), 1),
    (re.compile(r"\b(?:delivered|arrive[sd]?|get it|by)\s+tomorrow"), 1),
    (re.compile(rf"\b{_WEEK}"), 7),
)
_STOCK = re.compile(r"\b(?:in[\s-]stock|available now|ready to ship)\b")
_SOFT = (("cheap", "low price"), ("budget", "low price"), ("affordable", "low price"))

_NEGATORS = {"without", "no", "not", "excluding", "except", "exclude"}
_SKIP_AFTER_NEGATOR = {"a", "an", "the", "any", "more", "much", "as", "with", "having"}

# "in" is not a unit here: "2 in 1" is far more common than "2 in" meaning inches.
_UNIT = (
    r"fl\s?oz|oz|ounces?|lbs?|pounds?|kg|grams?|g|ml|liters?|litres?|l|gallons?|inch(?:es)?|"
    r"ft|feet|foot|cm|mm|gb|tb|packs?|pk|ct|count|pieces?|pcs"
)
_SIZE_UNIT = re.compile(rf"(?<![\w.])(\d+(?:\.\d+)?)\s?({_UNIT}|\")(?![a-z])")
_NUM = r"(\d+(?:\.\d+)?)"
_SIZE_DIM = re.compile(rf"(?<![\w.]){_NUM}\s?x\s?{_NUM}(?:\s?x\s?{_NUM})?(?![\w.])")
_SIZE_CLOTHES = re.compile(r"\b(xxs|xs|xxl|xxxl|[2-5]xl|xl|x-large|extra large|extra small)\b")
_SIZE_NUMBER = re.compile(r"\bsize\s+(\d+(?:\.\d+)?)\b")


def _sizes(text: str) -> list[str]:
    found: list[str] = []
    for m in _SIZE_DIM.finditer(text):
        found.append("x".join(g for g in m.groups() if g))
    without_dims = _SIZE_DIM.sub(" ", text)
    for m in _SIZE_UNIT.finditer(without_dims):
        unit = "".join(m.group(2).split())
        found.append(f"{m.group(1)} {unit}")
    for m in _SIZE_CLOTHES.finditer(text):
        found.append(m.group(1).replace(" ", "-"))
    for m in _SIZE_NUMBER.finditer(text):
        found.append(f"size {m.group(1)}")
    return list(dict.fromkeys(found))


def _longest_matches(words: list[str], vocabulary: frozenset[str], max_len: int) -> list[str]:
    """Greedy left-to-right longest match of word sequences found in vocabulary."""
    found: list[str] = []
    i = 0
    while i < len(words):
        for length in range(min(max_len, len(words) - i), 0, -1):
            phrase = " ".join(words[i : i + length])
            if phrase in vocabulary:
                found.append(phrase)
                i += length
                break
        else:
            i += 1
    return list(dict.fromkeys(found))


class QueryParser:
    def __init__(self, lexicon: Lexicon) -> None:
        self.lexicon = lexicon

    def parse(self, query: str, query_id: str = "") -> ParsedQuery:
        text = " " + query.lower().strip() + " "
        price_min = price_max = None
        delivery = None
        soft: list[str] = []

        if m := _RANGE.search(text):
            low, high = _amount(m, 1), _amount(m, 3)
            price_min, price_max = min(low, high), max(low, high)
            text = text.replace(m.group(0), " ")
        else:
            if m := _MAX.search(text):
                price_max = _amount(m)
                text = text.replace(m.group(0), " ")
            if m := _MIN.search(text):
                price_min = _amount(m)
                text = text.replace(m.group(0), " ")
        if price_min is not None and price_max is not None and price_min > price_max:
            price_min = price_max = None  # contradictory, ignore rather than guess

        if m := _DELIVERY_DAYS.search(text):
            delivery = int(next(g for g in m.groups() if g))
            text = text.replace(m.group(0), " ")
        else:
            for pattern, days in _DELIVERY_WORDS:
                if m := pattern.search(text):
                    delivery = days
                    text = text.replace(m.group(0), " ")
                    break

        in_stock = False
        if m := _STOCK.search(text):
            in_stock = True
            text = text.replace(m.group(0), " ")

        for word, label in _SOFT:
            if re.search(rf"\b{word}\b", text) and label not in soft:
                soft.append(label)

        words = tokens(text)
        words, excluded = self._split_exclusions(words)
        kept = " ".join(words)

        constraints = Constraints(
            brand=_longest_matches(words, self.lexicon.brands, 4),
            color=_longest_matches(words, self.lexicon.colors, 3),
            size=_sizes(text),
            material=_longest_matches(words, self.lexicon.materials, 2),
            price_min=price_min,
            price_max=price_max,
            delivery_max_days=delivery,
            in_stock_only=in_stock,
            exclude_terms=excluded,
        )
        return ParsedQuery(
            query_id=query_id,
            raw=query,
            product_terms=kept,
            constraints=constraints,
            soft_preferences=soft,
        )

    @staticmethod
    def _split_exclusions(words: list[str]) -> tuple[list[str], list[str]]:
        """Remove 'without X' style phrases. Returns (remaining words, excluded words)."""
        remaining: list[str] = []
        excluded: list[str] = []
        i = 0
        while i < len(words):
            if words[i] in _NEGATORS:
                j = i + 1
                while j < len(words) and words[j] in _SKIP_AFTER_NEGATOR:
                    j += 1
                if j < len(words) and not words[j].isdigit() and words[j] not in _NEGATORS:
                    excluded.append(words[j])
                    i = j + 1
                    continue
            remaining.append(words[i])
            i += 1
        return remaining, list(dict.fromkeys(excluded))
