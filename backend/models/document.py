from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any]
