from abc import ABC, abstractmethod

from app.models.document import Document
from app.models.metadata_schema import MetadataSchema
from app.processors.schema_discovery import schema_from_documents


class DocumentConnector(ABC):
    @abstractmethod
    def fetch_documents(self) -> list[Document]:
        """Return documents from a source."""
        ...

    def describe_schema(
        self, documents: list[Document] | None = None
    ) -> MetadataSchema:
        """Describe the filterable metadata schema this connector produces.

        The default implementation infers the schema from the connector's own
        typed document properties, so every connector contributes a schema for
        the LLM intent analyzer without source-specific code. Subclasses may
        override to supply richer type information from the source itself.

        Passing already-fetched ``documents`` avoids a redundant fetch during
        ingestion; when omitted, documents are fetched on demand.
        """
        docs = documents if documents is not None else self.fetch_documents()
        return schema_from_documents(docs)
