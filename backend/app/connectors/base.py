from abc import ABC, abstractmethod

from app.models.document import Document


class DocumentConnector(ABC):
    @abstractmethod
    def fetch_documents(self) -> list[Document]:
        """Return documents from a source."""
        pass
