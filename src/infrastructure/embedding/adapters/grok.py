from __future__ import annotations

from typing import Any, Optional, Union

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from ....domain.embedding.constants import XAI_EMBEDDING_PARAM_MAP
from ....domain.embedding.protocols import EmbeddingConfig
from ....domain.embedding.types import EmbeddingProvider
from ...utils import resolve_parameters
from ..base import BaseEmbedding
from ..factory import EmbeddingFactory

XAI_BASE_URL: str = "https://api.x.ai/v1"


@EmbeddingFactory.register(EmbeddingProvider.XAI)
class GrokEmbeddingModel(BaseEmbedding):
    """Wrapper for xAI Grok embeddings via the OpenAI-compatible API.

    Uses LangChain's ``OpenAIEmbeddings`` pointed at the xAI endpoint.

    Attributes:
        client (OpenAIEmbeddings): OpenAI client configured for xAI.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Grok embedding wrapper.

        Args:
            config: An object adhering to the EmbeddingConfig protocol.
            model: The model name.
            api_key: The xAI API key.
            **kwargs: Additional arguments for OpenAIEmbeddings.

        Raises:
            ValidationError: If required parameters are missing.
        """
        params: dict[str, Any] = resolve_parameters(
            config, model=model, api_key=api_key, **kwargs
        )
        params.setdefault("base_url", XAI_BASE_URL)

        for cfg_key, lc_key in XAI_EMBEDDING_PARAM_MAP.items():
            if cfg_key in params:
                params[lc_key] = params.pop(cfg_key)

        self.client = OpenAIEmbeddings(**params)
