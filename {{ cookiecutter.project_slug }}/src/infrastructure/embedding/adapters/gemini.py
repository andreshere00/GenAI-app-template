{%- if "4" in cookiecutter.embedding_providers -%}
from __future__ import annotations

from typing import Any, Optional, Union

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from ....domain.embedding.constants import GEMINI_EMBEDDING_PARAM_MAP
from ....domain.embedding.protocols import EmbeddingConfig
from ....domain.embedding.types import EmbeddingProvider
from ...utils import resolve_parameters
from ..base import BaseEmbedding
from ..factory import EmbeddingFactory


@EmbeddingFactory.register(EmbeddingProvider.GOOGLE)
class GeminiEmbeddingModel(BaseEmbedding):
    """Wrapper for GoogleGenerativeAIEmbeddings with flexible config.

    Attributes:
        client (GoogleGenerativeAIEmbeddings): The Gemini embeddings
            client.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Gemini embedding wrapper.

        Args:
            config: An object adhering to the EmbeddingConfig protocol.
            model: The model name (e.g., "models/embedding-001").
            api_key: The Google API key.
            **kwargs: Additional arguments for the embeddings client.

        Raises:
            ValidationError: If required parameters are missing.
        """
        params: dict[str, Any] = resolve_parameters(
            config, model=model, api_key=api_key, **kwargs
        )

        for cfg_key, lc_key in GEMINI_EMBEDDING_PARAM_MAP.items():
            if cfg_key in params:
                params[lc_key] = params.pop(cfg_key)

        self.client = GoogleGenerativeAIEmbeddings(**params)
{%- endif -%}
