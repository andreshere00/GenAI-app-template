from __future__ import annotations

from typing import Any, Optional, Union

from langchain_ollama import OllamaEmbeddings
from pydantic import SecretStr

from ....domain.embedding.protocols import EmbeddingConfig
from ....domain.embedding.types import EmbeddingProvider
from ...utils import resolve_parameters
from ..base import BaseEmbedding
from ..factory import EmbeddingFactory


@EmbeddingFactory.register(EmbeddingProvider.HUGGINGFACE)
class OllamaEmbeddingModel(BaseEmbedding):
    """Wrapper for LangChain's OllamaEmbeddings with flexible config.

    Attributes:
        client (OllamaEmbeddings): The Ollama embeddings client.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama embedding wrapper.

        Args:
            config: An object adhering to the EmbeddingConfig protocol.
            model: The model name (e.g., "nomic-embed-text").
            base_url: URL of the Ollama server.
            api_key: Unused; accepted for protocol compatibility.
            **kwargs: Additional arguments for OllamaEmbeddings.

        Raises:
            ValidationError: If required parameters are missing.
        """
        params: dict[str, Any] = resolve_parameters(
            config,
            model=model,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )
        params.pop("api_key", None)
        self.client = OllamaEmbeddings(**params)
