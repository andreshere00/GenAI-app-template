from __future__ import annotations

from typing import Any, Optional, Union

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from ....domain.embedding.constants import OPENAI_EMBEDDING_PARAM_MAP
from ....domain.embedding.protocols import EmbeddingConfig
from ....domain.embedding.types import EmbeddingProvider
from ...utils import resolve_parameters
from ..base import BaseEmbedding
from ..factory import EmbeddingFactory


@EmbeddingFactory.register(EmbeddingProvider.OPENAI)
class OpenAIEmbeddingModel(BaseEmbedding):
    """Wrapper for LangChain's OpenAIEmbeddings with flexible configuration.

    Attributes:
        client (OpenAIEmbeddings): The instantiated OpenAI embeddings client.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI embedding wrapper.

        Args:
            config: An object adhering to the EmbeddingConfig protocol.
            model: The model name (e.g., "text-embedding-3-small").
            api_key: The OpenAI API key.
            **kwargs: Additional arguments passed to OpenAIEmbeddings.

        Raises:
            ValidationError: If required parameters are missing.
        """
        params: dict[str, Any] = resolve_parameters(
            config, model=model, api_key=api_key, **kwargs
        )

        for cfg_key, lc_key in OPENAI_EMBEDDING_PARAM_MAP.items():
            if cfg_key in params:
                params[lc_key] = params.pop(cfg_key)

        self.client = OpenAIEmbeddings(**params)
