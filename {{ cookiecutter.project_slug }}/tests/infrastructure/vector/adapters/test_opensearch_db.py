{%- if "4" in cookiecutter.vector_db -%}
from typing import Any

import src.infrastructure.vector.adapters.opensearch_db as opensearch_module

from tests.infrastructure.vector.adapters._shared import (
    BaseVectorDatabase,
    DummyClient,
    FakePydanticConfig,
    VectorDBConfig,
)


class TestOpenSearchVectorDatabase:
    def test_hosts_mapping_and_instantiation_modes(self, monkeypatch):
        captured_calls: list[dict[str, Any]] = []

        def fake_opensearch_client(**kwargs: Any):
            captured_calls.append(kwargs)
            return DummyClient(**kwargs)

        monkeypatch.setattr(opensearch_module, "OpenSearch", fake_opensearch_client)

        cfg_dataclass = VectorDBConfig(host="os.local", port=9200, https=False)
        cfg_model = FakePydanticConfig(host="os.model", port=9201, https=True)
        cfg_dict = {"host": "os.dict", "port": 9202, "https": False}

        a = opensearch_module.OpenSearchVectorDatabase(config=cfg_dataclass)
        b = opensearch_module.OpenSearchVectorDatabase(config=cfg_model)
        c = opensearch_module.OpenSearchVectorDatabase(config=cfg_dict)
        d = opensearch_module.OpenSearchVectorDatabase(host="os.kw", port=9203, https=True)

        assert isinstance(a, BaseVectorDatabase)
        assert a.client is not b.client
        assert captured_calls[0]["hosts"][0]["host"] == "os.local"
        assert captured_calls[1]["hosts"][0]["scheme"] == "https"
        assert captured_calls[2]["hosts"][0]["host"] == "os.dict"
        assert captured_calls[3]["hosts"][0]["port"] == 9203
{%- endif -%}
