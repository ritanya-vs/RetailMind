"""Product knowledge graph and constraint checking.

Every product is a node carrying its price, delivery time and stock flag. Brand, color and
material are separate nodes that products link to, so "which products are Nike?" is a lookup of
one node's neighbors. Colors and materials are read from the product's own field and from its
title and text, because the field is empty for about a third of products.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from retailmind.query.lexicon import Lexicon, tokens
from retailmind.query.parser import longest_matches
from retailmind.schemas import Constraints

MAX_TEXT_TOKENS = 150  # the title comes first, so the start of the text is the informative part
ENTITY_KINDS = ("brand", "color", "material")


def _node(kind: str, name: str) -> str:
    return f"{kind}:{name}"


class ProductKG:
    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph

    @classmethod
    def from_products(
        cls, products: pd.DataFrame, attributes: pd.DataFrame, lexicon: Lexicon
    ) -> ProductKG:
        """products needs product_id, brand, color, text; attributes needs the synthetic fields."""
        attrs = attributes.set_index("product_id")
        graph = nx.DiGraph()
        for row in products.itertuples():
            product = _node("product", row.product_id)
            record = attrs.loc[row.product_id]
            graph.add_node(
                product,
                kind="product",
                price=float(record["price"]),
                delivery_days=int(record["delivery_days"]),
                in_stock=bool(record["in_stock"]),
            )
            words = tokens(row.text)[:MAX_TEXT_TOKENS]
            found = {
                "brand": set(longest_matches(words, lexicon.brands, 4)),
                "color": set(longest_matches(words, lexicon.colors, 3))
                | set(longest_matches(tokens(row.color), lexicon.colors, 3)),
                "material": set(longest_matches(words, lexicon.materials, 2)),
            }
            own_brand = " ".join(tokens(row.brand))
            if own_brand:
                found["brand"].add(own_brand)
            for kind, names in found.items():
                for name in names:
                    entity = _node(kind, name)
                    graph.add_node(entity, kind=kind)
                    graph.add_edge(product, entity, relation=f"has_{kind}")
        return cls(graph)

    @property
    def n_products(self) -> int:
        return sum(1 for _, kind in self.graph.nodes(data="kind") if kind == "product")

    def has(self, product_id: str) -> bool:
        return _node("product", product_id) in self.graph

    def entities(self, product_id: str, kind: str) -> set[str]:
        """Names of the brands, colors or materials a product is linked to."""
        node = _node("product", product_id)
        return {
            target.split(":", 1)[1]
            for target in self.graph.successors(node)
            if self.graph.nodes[target]["kind"] == kind
        }

    def products_with(self, kind: str, name: str) -> set[str]:
        """Ids of all products linked to the given brand, color or material."""
        node = _node(kind, name)
        if node not in self.graph:
            return set()
        return {source.split(":", 1)[1] for source in self.graph.predecessors(node)}

    def violations(
        self, product_id: str, constraints: Constraints, enabled: frozenset[str] | set[str]
    ) -> list[str]:
        """Names of the enabled constraints this product does not meet."""
        data = self.graph.nodes[_node("product", product_id)]
        found: list[str] = []
        if "price" in enabled:
            too_high = constraints.price_max is not None and data["price"] > constraints.price_max
            too_low = constraints.price_min is not None and data["price"] < constraints.price_min
            if too_high or too_low:
                found.append("price")
        if "delivery" in enabled and constraints.delivery_max_days is not None:
            if data["delivery_days"] > constraints.delivery_max_days:
                found.append("delivery")
        if "in_stock" in enabled and constraints.in_stock_only and not data["in_stock"]:
            found.append("in_stock")
        for kind in ENTITY_KINDS:
            wanted = {" ".join(tokens(w)) for w in getattr(constraints, kind)}
            if kind in enabled and wanted and not (wanted & self.entities(product_id, kind)):
                found.append(kind)
        return found
