{%- if "1" in cookiecutter.llm_providers -%}
from .anthropic import AnthropicModel
{% endif %}
{%- if "2" in cookiecutter.llm_providers -%}
from .azure_openai import AzureOpenAIModel
{% endif %}
{% if "3" in cookiecutter.llm_providers %}
from .bedrock import BedrockModel
{% endif %}
{% if "4" in cookiecutter.llm_providers %}
from .gemini import GeminiModel
{% endif %}
{% if "5" in cookiecutter.llm_providers %}
from .grok import GrokModel
{% endif %}
{% if "6" in cookiecutter.llm_providers %}
from .ollama import OllamaModel
{% endif %}
{% if "7" in cookiecutter.llm_providers %}
from .openai import OpenAIModel
{% endif %}


{%- if cookiecutter.llm_providers -%}__all__ = [
    {%- if "1" in cookiecutter.llm_providers -%}
    "AnthropicModel",
    {%- endif -%}
    {%- if "2" in cookiecutter.llm_providers -%}
    "AzureOpenAIModel",
    {%- endif -%}
    {%- if "3" in cookiecutter.llm_providers -%}
    "BedrockModel",
    {%- endif -%}
    {%- if "4" in cookiecutter.llm_providers -%}
    "GeminiModel",
    {%- endif -%}
    {%- if "5" in cookiecutter.llm_providers -%}
    "GrokModel",
    {%- endif -%}
    {%- if "6" in cookiecutter.llm_providers -%}
    "OllamaModel",
    {%- endif -%}
    {%- if "7" in cookiecutter.llm_providers -%}
    "OpenAIModel",
    {%- endif -%}
]{%- endif -%}