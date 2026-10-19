"""Apply parsed constraints to a ranked list, strictly or leniently."""

from __future__ import annotations

from collections.abc import Sequence

from retailmind.kg.graph import ProductKG
from retailmind.schemas import Constraints

Ranked = Sequence[tuple[str, float]]  # [(product_id, score), ...] best first

# Explicit numeric limits the user stated. Brand, color and material are weaker: they come from
# free-text queries where the product may legitimately omit or phrase them differently.
HARD_KINDS = frozenset({"price", "delivery", "in_stock"})


class ConstraintFilter:
    """strict drops products that violate an enabled constraint.

    lenient keeps them but lowers their score for each violated constraint: by `hard_penalty` for
    price, delivery and stock limits and by `penalty` for brand, color and material. With a large
    hard_penalty, products that break a stated limit sink below every product that meets it but
    stay in the list, so recall at depth is not lost. hard_penalty=None uses `penalty` for all.
    """

    def __init__(
        self,
        kg: ProductKG,
        *,
        mode: str = "lenient",
        penalty: float = 0.5,
        hard_penalty: float | None = None,
        enabled: Sequence[str] = (),
    ) -> None:
        if mode not in ("strict", "lenient"):
            raise ValueError("mode must be 'strict' or 'lenient'")
        if penalty < 0 or (hard_penalty is not None and hard_penalty < 0):
            raise ValueError("penalties must not be negative")
        self.kg = kg
        self.mode = mode
        self.penalty = penalty
        self.hard_penalty = penalty if hard_penalty is None else hard_penalty
        self.enabled = frozenset(enabled)

    def apply(
        self, constraints: Constraints, ranked: Ranked
    ) -> tuple[list[tuple[str, float]], dict[str, list[str]]]:
        """Return (filtered ranking, {product_id: violated constraints} for violators)."""
        if not self.enabled or constraints.is_empty():
            return list(ranked), {}
        violations: dict[str, list[str]] = {}
        for product_id, _ in ranked:
            if self.kg.has(product_id):
                found = self.kg.violations(product_id, constraints, self.enabled)
                if found:
                    violations[product_id] = found
        if self.mode == "strict":
            return [(p, s) for p, s in ranked if p not in violations], violations
        adjusted = [(p, s - self._cost(violations.get(p, ()))) for p, s in ranked]
        adjusted.sort(key=lambda item: -item[1])  # stable: equal scores keep their order
        return adjusted, violations

    def _cost(self, violated: Sequence[str]) -> float:
        return sum(self.hard_penalty if kind in HARD_KINDS else self.penalty for kind in violated)
