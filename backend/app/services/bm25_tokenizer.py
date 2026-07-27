from __future__ import annotations

import re
from collections.abc import Callable

# A tokenizer turns free text into the comparable term list BM25 scores over.
# Both the index (corpus side) and the query builder (query side) MUST use the
# same tokenizer, otherwise query terms will not line up with indexed terms.
Tokenizer = Callable[[str], list[str]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def default_tokenizer(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric terms.

    Deliberately simple and dependency-free: it lowercases the input and keeps
    maximal ``[a-z0-9]`` runs, dropping punctuation and whitespace. This is
    sufficient for a baseline offline BM25 index and keeps corpus/query
    tokenization identical by construction.
    """
    return _TOKEN_RE.findall(text.lower())
