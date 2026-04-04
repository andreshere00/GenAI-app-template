{%- if "2" in cookiecutter.embedding_providers -%}
from __future__ import annotations

from typing import Any, Optional, Union

from langchain_aws import BedrockEmbeddings
from pydantic import SecretStr

from ....domain.embedding.constants import BEDROCK_EMBEDDING_PARAM_MAP
from ....domain.embedding.protocols import EmbeddingConfig
from ....domain.embedding.types import EmbeddingProvider
from ...utils import resolve_parameters
from ..base import BaseEmbedding
from ..factory import EmbeddingFactory


@EmbeddingFactory.register(EmbeddingProvider.AWS)
class BedrockEmbeddingModel(BaseEmbedding):
    """Wrapper for LangChain's BedrockEmbeddings with flexible config.

    Attributes:
        client (BedrockEmbeddings): The Bedrock embeddings client.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Bedrock embedding wrapper.

        Args:
            config: An object adhering to the EmbeddingConfig protocol.
            model: The model ID (e.g., "amazon.titan-embed-text-v1").
            api_key: Unused; Bedrock relies on AWS credentials.
            **kwargs: Additional arguments for BedrockEmbeddings.

        Raises:
            ValidationError: If required parameters are missing.
        """
        params: dict[str, Any] = resolve_parameters(
            config, model=model, api_key=api_key, **kwargs
        )

        for cfg_key, lc_key in BEDROCK_EMBEDDING_PARAM_MAP.items():
            if cfg_key in params:
                params[lc_key] = params.pop(cfg_key)

        params.pop("api_key", None)
        self.client = BedrockEmbeddings(**params)
{%- endif -%}
