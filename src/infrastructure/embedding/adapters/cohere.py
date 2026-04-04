from __future__ import annotations

from typing import Any, Optional, Union

from langchain_cohere import CohereEmbeddings
from pydantic import SecretStr

from ....domain.embedding.constants import COHERE_EMBEDDING_PARAM_MAP
from ....domain.embedding.protocols import EmbeddingConfig
from ....domain.embedding.types import EmbeddingProvider
from ...utils import resolve_parameters
from ..base import BaseEmbedding
from ..factory import EmbeddingFactory


@EmbeddingFactory.register(EmbeddingProvider.COHERE)
class CohereEmbeddingModel(BaseEmbedding):
    """Wrapper for LangChain's CohereEmbeddings with flexible config.

    Attributes:
        client (CohereEmbeddings): The Cohere embeddings client.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Cohere embedding wrapper.

        Args:
            config: An object adhering to the EmbeddingConfig protocol.
            model: The model name (e.g., "embed-english-v3.0").
            api_key: The Cohere API key.
            **kwargs: Additional arguments for CohereEmbeddings.

        Raises:
            ValidationError: If required parameters are missing.
        """
        params: dict[str, Any] = resolve_parameters(
            config, model=model, api_key=api_key, **kwargs
        )

        for cfg_key, lc_key in COHERE_EMBEDDING_PARAM_MAP.items():
            if cfg_key in params:
                params[lc_key] = params.pop(cfg_key)

        self.client = CohereEmbeddings(**params)
