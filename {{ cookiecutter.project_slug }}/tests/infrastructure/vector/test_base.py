{%- if cookiecutter.vector_db -%}
from typing import Any

import pytest

from src.domain.vector import CollectionConfig, VectorDBConfigDTO as VectorDBConfig
from src.infrastructure.vector.base import BaseVectorDatabase


class ConcreteVectorDatabase(BaseVectorDatabase):
    """Test double for validating the abstract base class contract."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._collections: dict[str, Any] = {}

    def connect(self) -> None:
        self.client = object()

    def disconnect(self) -> None:
        self.client = None

    def health(self) -> bool:
        return self.client is not None

    def create_collection(self, config: CollectionConfig) -> None:
        self._collections[config.name] = config

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)

    def list_collections(self) -> list[str]:
        return sorted(self._collections)

    def has_collection(self, name: str) -> bool:
        return name in self._collections

    def upsert(self, collection_name: str, records: list[Any], **kwargs: Any) -> None:
        store = self._collections.setdefault(collection_name, {})
        for record in records:
            store[record.id] = record

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        **kwargs: Any,
    ) -> list[Any]:
        records = list(self._collections.get(collection_name, {}).values())
        return records[:limit]

    def delete(self, collection_name: str, ids: list[str], **kwargs: Any) -> None:
        store = self._collections.setdefault(collection_name, {})
        for record_id in ids:
            store.pop(record_id, None)


class TestBaseVectorDatabase:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseVectorDatabase()

    def test_concrete_subclass_initializes_and_manages_client_state(self):
        config = VectorDBConfig(host="localhost", port=6333)
        adapter = ConcreteVectorDatabase(config=config)

        assert adapter.config == config
        assert adapter.client is None

        adapter.connect()
        assert adapter.health() is True

        adapter.disconnect()
        assert adapter.health() is False
{%- endif -%}
