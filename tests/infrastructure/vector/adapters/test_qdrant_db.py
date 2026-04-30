from typing import Any

import src.infrastructure.vector.adapters.qdrant_db as qdrant_module

from tests.infrastructure.vector.adapters._shared import (
    BaseVectorDatabase,
    DummyClient,
    FakePydanticConfig,
    VectorDBConfig,
)


class TestQdrantVectorDatabase:
    def test_instantiation_modes_and_isolation(self, monkeypatch):
        captured_calls: list[dict[str, Any]] = []

        def fake_qdrant_client(**kwargs: Any):
            captured_calls.append(kwargs)
            return DummyClient(**kwargs)

        monkeypatch.setattr(qdrant_module, "QdrantClient", fake_qdrant_client)

        cfg_dataclass = VectorDBConfig(host="qdrant.local", port=6333)
        cfg_model = FakePydanticConfig(host="qdrant.model", port=6333)
        cfg_dict = {"host": "qdrant.dict", "port": 6333}

        a = qdrant_module.QdrantVectorDatabase(config=cfg_dataclass)
        b = qdrant_module.QdrantVectorDatabase(config=cfg_model)
        c = qdrant_module.QdrantVectorDatabase(config=cfg_dict)
        d = qdrant_module.QdrantVectorDatabase(host="qdrant.kwargs", port=6333)

        assert isinstance(a, BaseVectorDatabase)
        assert a.client is not b.client
        assert b.client is not c.client
        assert c.client is not d.client
        assert captured_calls[0]["host"] == "qdrant.local"
        assert captured_calls[1]["host"] == "qdrant.model"
        assert captured_calls[2]["host"] == "qdrant.dict"
        assert captured_calls[3]["host"] == "qdrant.kwargs"
