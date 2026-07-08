from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    content: str
    metadata: dict[str, Any]
