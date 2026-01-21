import abc

from src.domain.prompt.protocols import PromptStorageAdapter
from src.domain.prompt.types import PromptTemplate


class BaseStorageAdapter(PromptStorageAdapter, abc.ABC):
    """Base abstract class for all storage adapters."""

    @abc.abstractmethod
    def load_template(self, path: str) -> PromptTemplate:
        """
        Abstract method to load a template.
        
        Subclasses must implement this method to satisfy the 
        PromptStorageAdapter protocol.
        """
        ...