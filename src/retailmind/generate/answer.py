"""Grounded answer generation: write a short recommendation from the retrieved products only.

The LLM is shown a numbered list of the top products and told to use only those, citing them as
[1], [2]. This keeps the answer grounded in retrieval instead of the model's own knowledge, which
is what the faithfulness metric in evaluation checks. Every product the LLM may cite is returned
alongside the answer, so a citation can be traced back to a real product_id.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from retailmind.generate.llm import LLMClient, Message
from retailmind.schemas import Constraints

SYSTEM = (
    "You are a shopping assistant. Recommend products using ONLY the numbered products provided. "
    "Never invent products, prices, or details that are not listed. Cite every product you mention "
    "as [1], [2], matching its number. If none of the products fit the request, say so plainly. "
    "Answer in two or three sentences."
)


def _constraint_line(constraints: Constraints) -> str:
    parts = []
    if constraints.price_max is not None:
        parts.append(f"max price ${constraints.price_max:g}")
    if constraints.price_min is not None:
        parts.append(f"min price ${constraints.price_min:g}")
    if constraints.delivery_max_days is not None:
        parts.append(f"delivery within {constraints.delivery_max_days} days")
    if constraints.in_stock_only:
        parts.append("in stock")
    for kind in ("brand", "color", "material"):
        if values := getattr(constraints, kind):
            parts.append(f"{kind}: {', '.join(values)}")
    return "; ".join(parts)


def build_context(
    product_ids: Sequence[str],
    products: Mapping[str, Mapping],
    *,
    max_products: int,
    max_chars: int = 320,
) -> tuple[str, list[str]]:
    """Return a numbered product list for the prompt and the ids shown, in order."""
    shown = [pid for pid in product_ids[:max_products] if pid in products]
    lines = []
    for number, pid in enumerate(shown, start=1):
        record = products[pid]
        title = str(record.get("title") or "").strip()
        text = str(record.get("text") or "").strip()[:max_chars]
        detail = f" {text}" if text and text != title else ""
        lines.append(f"[{number}] {title}.{detail}".strip())
    return "\n".join(lines), shown


def build_messages(query: str, context: str, constraints: Constraints) -> list[Message]:
    limits = _constraint_line(constraints)
    user = [f"Shopper's request: {query}"]
    if limits:
        user.append(f"Hard requirements: {limits}")
    user.append(f"\nProducts:\n{context}\n\nRecommendation:")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(user)}]


class AnswerGenerator:
    def __init__(
        self, client: LLMClient, *, temperature: float = 0.0, max_tokens: int = 512
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        query: str,
        product_ids: Sequence[str],
        products: Mapping[str, Mapping],
        *,
        constraints: Constraints | None = None,
        max_products: int = 5,
    ) -> tuple[str, list[str]]:
        """Return (answer text, the product ids it was grounded on)."""
        constraints = constraints or Constraints()
        context, shown = build_context(product_ids, products, max_products=max_products)
        if not shown:
            return "I couldn't find any products matching that request.", []
        messages = build_messages(query, context, constraints)
        answer = self.client.complete(
            messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        return answer, shown
