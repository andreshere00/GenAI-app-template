{%- if "3" in cookiecutter.vector_db -%}
from typing import Any

import src.infrastructure.vector.adapters.mongo_db as mongo_module

from tests.infrastructure.vector.adapters._shared import (
    BaseVectorDatabase,
    DummyClient,
    FakePydanticConfig,
    VectorDBConfig,
)


class TestMongoDBVectorDatabase:
    def test_url_mapping_and_instantiation_modes(self, monkeypatch):
        captured_calls: list[dict[str, Any]] = []

        def fake_mongo_client(**kwargs: Any):
            captured_calls.append(kwargs)
            return DummyClient(**kwargs)

        monkeypatch.setattr(mongo_module, "MongoClient", fake_mongo_client)

        cfg_dataclass = VectorDBConfig(url="mongodb://localhost:27017", database="db1")
        cfg_model = FakePydanticConfig(url="mongodb://mongo-model:27017", database="db2")
        cfg_dict = {"url": "mongodb://mongo-dict:27017", "database": "db3"}

        a = mongo_module.MongoDBVectorDatabase(config=cfg_dataclass)
        b = mongo_module.MongoDBVectorDatabase(config=cfg_model)
        c = mongo_module.MongoDBVectorDatabase(config=cfg_dict)
        d = mongo_module.MongoDBVectorDatabase(url="mongodb://mongo-kw:27017", database="db4")

        assert isinstance(a, BaseVectorDatabase)
        assert a.client is not b.client
        assert captured_calls[0]["host"] == "mongodb://localhost:27017"
        assert "url" not in captured_calls[0]
        assert captured_calls[1]["host"] == "mongodb://mongo-model:27017"
        assert captured_calls[2]["host"] == "mongodb://mongo-dict:27017"
        assert captured_calls[3]["host"] == "mongodb://mongo-kw:27017"
{%- endif -%}
