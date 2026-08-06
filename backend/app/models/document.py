from dataclasses import dataclass
from typing import Any

# Metadata key under which a connector stores SEMANTIC-role property content
# ({original_name: text}) so the chunker can merge it into the page body before
# chunking. It is stripped from chunk metadata during chunking so it never
# leaks into the vector store payload as a filterable field.
SEMANTIC_PROPERTIES_KEY = "semantic_properties"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    content: str
    metadata: dict[str, Any]
