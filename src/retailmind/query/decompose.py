"""A5: split a multi-part request into simpler sub-queries with an LLM.

A query like "a yoga mat and a water bottle" retrieves poorly as one string because no product is
both. Decomposition asks the LLM to list the separate product needs, each of which is retrieved on
its own and the results merged. A single-product query returns one sub-query unchanged, so the
stage is safe to always run. Constraints are parsed separately; this only concerns product needs.
"""

from __future__ import annotations

import json

from retailmind.generate.llm import LLMClient, LLMError, Message

SYSTEM = (
    "Split a shopping search into the distinct product needs it contains. "
    "Return ONLY a JSON array of short search strings, one per product the shopper wants. "
    "If the search is for a single product, return a one-element array with it cleaned up. "
    "Drop prices, delivery and stock wording. Example: "
    '"yoga mat and a water bottle under $20" -> ["yoga mat", "water bottle"].'
)
MAX_SUB_QUERIES = 4


def _messages(query: str) -> list[Message]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]


def _parse(reply: str) -> list[str] | None:
    start, end = reply.find("["), reply.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    cleaned = [str(item).strip() for item in data if str(item).strip()]
    return cleaned or None


class QueryDecomposer:
    def __init__(
        self, client: LLMClient, *, temperature: float = 0.0, max_tokens: int = 128
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def decompose(self, query: str, fallback: str | None = None) -> list[str]:
        """Return sub-queries. On any LLM or parse failure, fall back to the query unchanged."""
        base = (fallback if fallback is not None else query).strip()
        try:
            reply = self.client.complete(
                _messages(query), temperature=self.temperature, max_tokens=self.max_tokens
            )
        except LLMError:
            return [base] if base else []
        parsed = _parse(reply)
        if not parsed:
            return [base] if base else []
        return parsed[:MAX_SUB_QUERIES]
