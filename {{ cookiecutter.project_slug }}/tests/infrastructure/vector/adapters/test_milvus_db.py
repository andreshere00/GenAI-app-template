{%- if "2" in cookiecutter.vector_db -%}
from typing import Any

import src.infrastructure.vector.adapters.milvus_db as milvus_module

from tests.infrastructure.vector.adapters._shared import (
    BaseVectorDatabase,
    DummyClient,
    FakePydanticConfig,
    VectorDBConfig,
)


class TestMilvusVectorDatabase:
    def test_mapping_and_instantiation_modes(self, monkeypatch):
        captured_calls: list[dict[str, Any]] = []

        def fake_milvus_client(**kwargs: Any):
            captured_calls.append(kwargs)
            return DummyClient(**kwargs)

        monkeypatch.setattr(milvus_module, "MilvusClient", fake_milvus_client)

        cfg_dataclass = VectorDBConfig(host="milvus.local", port=19530, database="db1")
        cfg_model = FakePydanticConfig(host="milvus.model", port=19531, database="db2")
        cfg_dict = {"host": "milvus.dict", "port": 19532, "database": "db3"}

        a = milvus_module.MilvusVectorDatabase(config=cfg_dataclass)
        b = milvus_module.MilvusVectorDatabase(config=cfg_model)
        c = milvus_module.MilvusVectorDatabase(config=cfg_dict)
        d = milvus_module.MilvusVectorDatabase(
            host="milvus.kw",
            port=19533,
            database="db4",
        )

        assert isinstance(a, BaseVectorDatabase)
        assert a.client is not b.client
        assert captured_calls[0]["uri"] == "http://milvus.local:19530"
        assert captured_calls[0]["db_name"] == "db1"
        assert captured_calls[1]["uri"] == "http://milvus.model:19531"
        assert captured_calls[2]["uri"] == "http://milvus.dict:19532"
        assert captured_calls[3]["db_name"] == "db4"
{%- endif -%}
