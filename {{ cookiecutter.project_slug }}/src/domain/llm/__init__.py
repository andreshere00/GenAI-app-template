from .types import LLMProvider
from .protocols import ModelConfig
{% if cookiecutter.llm_providers %}
from .constants import (
    {% if "7" in cookiecutter.llm_providers %}
    OPENAI_PARAM_MAP,
    {%- endif -%}
    {% if "4" in cookiecutter.llm_providers %}
    GEMINI_PARAM_MAP,
    {%- endif -%}
    {% if "2" in cookiecutter.llm_providers %}
    AZURE_OPENAI_PARAM_MAP,
    {%- endif -%}
    {% if "3" in cookiecutter.llm_providers %}
    BEDROCK_PARAM_MAP,
    {%- endif -%}
    {% if "5" in cookiecutter.llm_providers %}
    XAI_PARAM_MAP,
    {%- endif -%}
    {% if "1" in cookiecutter.llm_providers %}
    CLAUDE_PARAM_MAP,
    {% endif %}
)
{% endif %}

__all__: list[str] = [
    "LLMProvider",
    "ModelConfig",
    {% if "7" in cookiecutter.llm_providers %}
    "OPENAI_PARAM_MAP",
    {%- endif -%}
    {% if "4" in cookiecutter.llm_providers %}
    "GEMINI_PARAM_MAP",
    {%- endif -%}
    {% if "2" in cookiecutter.llm_providers %}
    "AZURE_OPENAI_PARAM_MAP",
    {%- endif -%}
    {% if "3" in cookiecutter.llm_providers %}
    "BEDROCK_PARAM_MAP",
    {%- endif -%}
    {% if "5" in cookiecutter.llm_providers %}
    "XAI_PARAM_MAP",
    {%- endif -%}
    {% if "1" in cookiecutter.llm_providers %}
    "CLAUDE_PARAM_MAP",
    {%- endif -%}
]
