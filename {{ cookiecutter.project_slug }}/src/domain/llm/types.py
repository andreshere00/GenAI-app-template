from enum import Enum

{% if cookiecutter.llm_providers %}
class LLMProvider(str, Enum):
    {% if "1" in cookiecutter.llm_providers %}
    ANTHROPIC = "claude"
    {%- endif -%}
    {% if "2" in cookiecutter.llm_providers %}
    AZURE = "azure-openai"
    {%- endif -%}
    {% if "3" in cookiecutter.llm_providers %}
    AWS = "bedrock"
    {%- endif -%}
    {% if "4" in cookiecutter.llm_providers %}
    GOOGLE = "gemini"
    {%- endif -%}
    {% if "5" in cookiecutter.llm_providers %}
    XAI = "grok"
    {%- endif -%}
    {% if "6" in cookiecutter.llm_providers %}
    HUGGINGFACE = "ollama"
    {%- endif -%}
    {% if "7" in cookiecutter.llm_providers %}
    OPENAI = "openai"
    {%- endif -%}
{%- endif -%}