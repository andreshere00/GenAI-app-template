from typing import Any

import src.infrastructure.vector.adapters.vertex_db as vertex_module

from tests.infrastructure.vector.adapters._shared import (
    BaseVectorDatabase,
    FakePydanticConfig,
    VectorDBConfig,
)


class TestVertexDBVectorDatabase:
    def test_instantiation_modes_and_isolation(self, monkeypatch):
        init_calls: list[dict[str, Any]] = []

        class FakeAiplatform:
            @staticmethod
            def init(**kwargs: Any):
                init_calls.append(kwargs)

            class MatchingEngineIndex:
                def __init__(self, **kwargs: Any):
                    self.kwargs = kwargs

        monkeypatch.setattr(vertex_module, "aiplatform", FakeAiplatform)

        cfg_dataclass = VectorDBConfig(
            database="proj1",
            region="us-central1",
            collection="idx1",
        )
        cfg_model = FakePydanticConfig(
            database="proj2",
            region="europe-west1",
            collection="idx2",
        )
        cfg_dict = {"database": "proj3", "region": "us-east1", "collection": "idx3"}

        a = vertex_module.VertexDBVectorDatabase(config=cfg_dataclass)
        b = vertex_module.VertexDBVectorDatabase(config=cfg_model)
        c = vertex_module.VertexDBVectorDatabase(config=cfg_dict)
        d = vertex_module.VertexDBVectorDatabase(project_id="proj4", region="us-west1")

        assert isinstance(a, BaseVectorDatabase)
        assert a is not b
        assert b is not c
        assert c is not d

        a.connect()
        b.connect()
        c.connect()
        d.connect()

        assert init_calls[0]["project"] == "proj1"
        assert init_calls[1]["project"] == "proj2"
        assert init_calls[2]["project"] == "proj3"
        assert init_calls[3]["project"] == "proj4"
