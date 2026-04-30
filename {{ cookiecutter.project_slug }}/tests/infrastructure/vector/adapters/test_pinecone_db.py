{%- if "5" in cookiecutter.vector_db -%}
from typing import Any

import src.infrastructure.vector.adapters.pinecone_db as pinecone_module

from tests.infrastructure.vector.adapters._shared import (
    BaseVectorDatabase,
    FakePydanticConfig,
    VectorDBConfig,
)


class TestPineconeVectorDatabase:
    def test_instantiation_modes_and_isolation(self, monkeypatch):
        captured_calls: list[dict[str, Any]] = []

        class FakePinecone:
            def __init__(self, **kwargs: Any):
                captured_calls.append(kwargs)

            def Index(self, *_args: Any, **_kwargs: Any):
                class _FakeIndex:
                    def describe_index_stats(self):
                        return {}

                return _FakeIndex()

        monkeypatch.setattr(pinecone_module, "Pinecone", FakePinecone)

        cfg_dataclass = VectorDBConfig(api_key="k1", host="h1", collection="idx1")
        cfg_model = FakePydanticConfig(api_key="k2", host="h2", collection="idx2")
        cfg_dict = {"api_key": "k3", "host": "h3", "collection": "idx3"}

        a = pinecone_module.PineconeVectorDatabase(config=cfg_dataclass)
        b = pinecone_module.PineconeVectorDatabase(config=cfg_model)
        c = pinecone_module.PineconeVectorDatabase(config=cfg_dict)
        d = pinecone_module.PineconeVectorDatabase(
            api_key="k4",
            host="h4",
            index_name="idx4",
        )

        assert isinstance(a, BaseVectorDatabase)
        assert a.client is not b.client
        assert b.client is not c.client
        assert c.client is not d.client
        assert captured_calls[0]["api_key"] == "k1"
        assert captured_calls[1]["api_key"] == "k2"
        assert captured_calls[2]["api_key"] == "k3"
        assert captured_calls[3]["api_key"] == "k4"
{%- endif -%}
