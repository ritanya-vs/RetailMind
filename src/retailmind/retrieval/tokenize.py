"""Tokenizer shared by every keyword-based component."""

import re

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into letters and digits. No stemming and no stop-word removal."""
    return _TOKEN.findall(text.lower())
