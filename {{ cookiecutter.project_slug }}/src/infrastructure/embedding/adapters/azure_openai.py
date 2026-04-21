{%- if "1" in cookiecutter.embedding_providers -%}
from __future__ import annotations

from typing import Any, Optional, Union

from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr

from ....domain.embedding.constants import (
    AZURE_OPENAI_EMBEDDING_PARAM_MAP,
)
from ....domain.embedding.protocols import EmbeddingConfig
from ....domain.embedding.types import EmbeddingProvider
from ...utils import resolve_parameters
from ..base import BaseEmbedding
from ..factory import EmbeddingFactory


@EmbeddingFactory.register(EmbeddingProvider.AZURE)
class AzureOpenAIEmbeddingModel(BaseEmbedding):
    """Wrapper for LangChain's AzureOpenAIEmbeddings with flexible config.

    Attributes:
        client (AzureOpenAIEmbeddings): The Azure embeddings client.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Azure OpenAI embedding wrapper.

        Args:
            config: An object adhering to the EmbeddingConfig protocol.
            model: The deployment name.
            api_key: The Azure API key.
            azure_endpoint: The Azure endpoint URL.
            api_version: The Azure API version.
            **kwargs: Additional arguments for AzureOpenAIEmbeddings.
        """
        params: dict[str, Any] = resolve_parameters(
            config,
            model=model,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            openai_api_version=api_version,
            **kwargs,
        )

        for cfg_key, lc_key in AZURE_OPENAI_EMBEDDING_PARAM_MAP.items():
            if cfg_key in params:
                if lc_key not in params:
                    params[lc_key] = params.pop(cfg_key)

        self.client = AzureOpenAIEmbeddings(**params)
{%- endif -%}
