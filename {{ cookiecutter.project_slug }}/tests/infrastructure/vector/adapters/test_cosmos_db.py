{%- if "1" in cookiecutter.vector_db -%}
from typing import Any

import src.infrastructure.vector.adapters.cosmos_db as cosmos_module

from tests.infrastructure.vector.adapters._shared import (
    BaseVectorDatabase,
    DummyClient,
    FakePydanticConfig,
    VectorDBConfig,
)


class TestCosmosDBVectorDatabase:
    def test_credential_mapping_and_instantiation_modes(self, monkeypatch):
        captured_calls: list[dict[str, Any]] = []

        def fake_cosmos_client(**kwargs: Any):
            captured_calls.append(kwargs)
            return DummyClient(**kwargs)

        monkeypatch.setattr(cosmos_module, "CosmosClient", fake_cosmos_client)

        cfg_dataclass = VectorDBConfig(url="https://cosmos.local", api_key="k1", database="db1")
        cfg_model = FakePydanticConfig(url="https://cosmos.model", api_key="k2", database="db2")
        cfg_dict = {"url": "https://cosmos.dict", "api_key": "k3", "database": "db3"}

        a = cosmos_module.CosmosDBVectorDatabase(config=cfg_dataclass)
        b = cosmos_module.CosmosDBVectorDatabase(config=cfg_model)
        c = cosmos_module.CosmosDBVectorDatabase(config=cfg_dict)
        d = cosmos_module.CosmosDBVectorDatabase(
            url="https://cosmos.kw",
            api_key="k4",
            database="db4",
        )

        assert isinstance(a, BaseVectorDatabase)
        assert a.client is not b.client
        assert captured_calls[0]["credential"] == "k1"
        assert "api_key" not in captured_calls[0]
        assert captured_calls[1]["credential"] == "k2"
        assert captured_calls[2]["credential"] == "k3"
        assert captured_calls[3]["credential"] == "k4"
{%- endif -%}
