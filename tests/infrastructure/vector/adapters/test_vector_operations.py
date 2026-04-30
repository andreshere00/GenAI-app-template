from __future__ import annotations

from typing import Any

import src.infrastructure.vector.adapters.cosmos_db as cosmos_module
import src.infrastructure.vector.adapters.milvus_db as milvus_module
import src.infrastructure.vector.adapters.mongo_db as mongo_module
import src.infrastructure.vector.adapters.opensearch_db as opensearch_module
import src.infrastructure.vector.adapters.pinecone_db as pinecone_module
import src.infrastructure.vector.adapters.qdrant_db as qdrant_module
import src.infrastructure.vector.adapters.vertex_db as vertex_module
from src.domain.vector import VectorRecordDTO


# ---- Mocks, fixtures & helpers ---- #
def _records() -> list[VectorRecordDTO]:
    return [VectorRecordDTO(id="r1", vector=[0.1, 0.2], payload={"chunk": "a"})]


# ---- Happy path ---- #
def test_qdrantvectordatabase_upsert_search_delete_valid_payload_calls_client(
    monkeypatch,
) -> None:
    class DummyClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.deleted = False

        def upsert(self, **kwargs: Any) -> None:
            self.upsert_kwargs = kwargs

        def search(self, **kwargs: Any) -> list[Any]:
            return [type("Hit", (), {"id": "r1", "score": 0.9, "payload": {"chunk": "a"}})()]

        def delete(self, **kwargs: Any) -> None:
            self.deleted = True

    monkeypatch.setattr(qdrant_module, "QdrantClient", DummyClient)
    adapter = qdrant_module.QdrantVectorDatabase(host="localhost", port=6333)
    adapter.upsert("docs", _records())
    results = adapter.search("docs", [0.1, 0.2], limit=2)
    adapter.delete("docs", ["r1"])

    assert adapter.client.upsert_kwargs["collection_name"] == "docs"
    assert results[0].id == "r1"
    assert adapter.client.deleted is True


def test_milvusvectordatabase_upsert_search_delete_valid_payload_calls_client(
    monkeypatch,
) -> None:
    class DummyClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def upsert(self, **kwargs: Any) -> None:
            self.upsert_kwargs = kwargs

        def search(self, **kwargs: Any) -> list[Any]:
            return [[{"entity": {"id": "r1", "chunk": "a"}, "distance": 0.8}]]

        def delete(self, **kwargs: Any) -> None:
            self.delete_kwargs = kwargs

    monkeypatch.setattr(milvus_module, "MilvusClient", DummyClient)
    adapter = milvus_module.MilvusVectorDatabase(host="localhost", port=19530)
    adapter.upsert("docs", _records())
    results = adapter.search("docs", [0.1, 0.2], limit=2)
    adapter.delete("docs", ["r1"])

    assert adapter.client.upsert_kwargs["collection_name"] == "docs"
    assert results[0].payload["chunk"] == "a"
    assert adapter.client.delete_kwargs["ids"] == ["r1"]


def test_pineconevectordatabase_upsert_search_delete_valid_payload_calls_index(
    monkeypatch,
) -> None:
    class DummyIndex:
        def upsert(self, **kwargs: Any) -> None:
            self.upsert_kwargs = kwargs

        def query(self, **kwargs: Any) -> dict[str, Any]:
            return {"matches": [{"id": "r1", "score": 0.7, "metadata": {"chunk": "a"}}]}

        def delete(self, **kwargs: Any) -> None:
            self.delete_kwargs = kwargs

    class DummyClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.index = DummyIndex()

        def Index(self, *args: Any, **kwargs: Any) -> DummyIndex:
            return self.index

        def list_indexes(self) -> list[Any]:
            return []

    monkeypatch.setattr(pinecone_module, "Pinecone", DummyClient)
    adapter = pinecone_module.PineconeVectorDatabase(api_key="key")
    adapter.upsert("docs", _records())
    results = adapter.search("docs", [0.1, 0.2], limit=2)
    adapter.delete("docs", ["r1"])

    assert adapter.client.index.upsert_kwargs["vectors"][0][0] == "r1"
    assert results[0].score == 0.7
    assert adapter.client.index.delete_kwargs["ids"] == ["r1"]


def test_mongodbvectordatabase_upsert_search_delete_valid_payload_calls_collection(
    monkeypatch,
) -> None:
    class DummyCollection:
        def replace_one(self, *args: Any, **kwargs: Any) -> None:
            self.replace_called = True

        def aggregate(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            return [{"_id": "r1", "payload": {"chunk": "a"}, "score": 0.6}]

        def delete_many(self, *args: Any, **kwargs: Any) -> None:
            self.delete_called = True

    class DummyDB:
        def __init__(self) -> None:
            self.collection = DummyCollection()

        def __getitem__(self, key: str) -> DummyCollection:
            return self.collection

        def list_collection_names(self) -> list[str]:
            return []

    class DummyClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.db = DummyDB()
            self.admin = type("Admin", (), {"command": lambda *args, **kwargs: {"ok": 1}})()

        def __getitem__(self, key: str) -> DummyDB:
            return self.db

    monkeypatch.setattr(mongo_module, "MongoClient", DummyClient)
    adapter = mongo_module.MongoDBVectorDatabase(host="localhost", port=27017, database="test")
    adapter.upsert("docs", _records())
    results = adapter.search("docs", [0.1, 0.2], limit=2)
    adapter.delete("docs", ["r1"])

    assert adapter.client.db.collection.replace_called is True
    assert results[0].id == "r1"
    assert adapter.client.db.collection.delete_called is True


def test_opensearchvectordatabase_upsert_search_delete_valid_payload_calls_client(
    monkeypatch,
) -> None:
    class DummyIndices:
        def exists(self, **kwargs: Any) -> bool:
            return True

        def get_alias(self, **kwargs: Any) -> dict[str, Any]:
            return {}

    class DummyClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.indices = DummyIndices()

        def index(self, **kwargs: Any) -> None:
            self.index_called = True

        def search(self, **kwargs: Any) -> dict[str, Any]:
            return {"hits": {"hits": [{"_id": "r1", "_score": 0.5, "_source": {"payload": {}}}]}}

        def delete(self, **kwargs: Any) -> None:
            self.delete_called = True

    monkeypatch.setattr(opensearch_module, "OpenSearch", DummyClient)
    adapter = opensearch_module.OpenSearchVectorDatabase(host="localhost", port=9200)
    adapter.upsert("docs", _records())
    results = adapter.search("docs", [0.1, 0.2], limit=2)
    adapter.delete("docs", ["r1"])

    assert adapter.client.index_called is True
    assert results[0].id == "r1"
    assert adapter.client.delete_called is True


def test_cosmosdbvectordatabase_upsert_search_delete_valid_payload_calls_container(
    monkeypatch,
) -> None:
    class DummyContainer:
        def upsert_item(self, item: dict[str, Any], **kwargs: Any) -> None:
            self.item = item

        def query_items(self, **kwargs: Any) -> list[dict[str, Any]]:
            return [{"id": "r1", "vector": [0.1, 0.2], "payload": {"chunk": "a"}}]

        def delete_item(self, **kwargs: Any) -> None:
            self.deleted = True

    class DummyDB:
        def __init__(self) -> None:
            self.container = DummyContainer()

        def get_container_client(self, name: str) -> DummyContainer:
            return self.container

        def list_containers(self) -> list[dict[str, str]]:
            return []

    class DummyClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.db = DummyDB()

        def get_database_client(self, name: str) -> DummyDB:
            return self.db

    monkeypatch.setattr(cosmos_module, "CosmosClient", DummyClient)
    adapter = cosmos_module.CosmosDBVectorDatabase(
        url="https://example.com",
        api_key="key",
        database="db",
    )
    adapter.upsert("docs", _records())
    results = adapter.search("docs", [0.1, 0.2], limit=2)
    adapter.delete("docs", ["r1"])

    assert adapter.client.db.container.item["id"] == "r1"
    assert results[0].id == "r1"
    assert adapter.client.db.container.deleted is True


def test_vertexdbvectordatabase_upsert_search_delete_valid_payload_uses_memory_cache(
    monkeypatch,
) -> None:
    monkeypatch.setattr(vertex_module.aiplatform, "init", lambda **kwargs: None)
    adapter = vertex_module.VertexDBVectorDatabase(project_id="p", region="us-central1")
    adapter.upsert("docs", _records())
    results = adapter.search("docs", [0.1, 0.2], limit=2)
    adapter.delete("docs", ["r1"])

    assert results[0].id == "r1"
    assert adapter.search("docs", [0.1, 0.2]) == []


# ---- Error paths ---- #
# ---- Edge cases ---- #
