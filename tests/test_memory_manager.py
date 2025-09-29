from eaa.agents.memory import LocalVectorStore, MemoryManager, MemoryManagerConfig


def dummy_embedder(texts):
    return [[float(len(text))] for text in texts]


def test_memory_manager_config_from_dict():
    payload = {
        "enabled": True,
        "write_enabled": False,
        "retrieval_enabled": True,
        "top_k": 3,
        "score_threshold": 0.5,
        "min_content_length": 5,
        "embedding_model": "custom",
        "vector_store_path": "/tmp/path",
        "injection_role": "user",
        "ignored": "value",
    }

    config = MemoryManagerConfig.from_dict(payload)

    assert config.enabled is True
    assert config.write_enabled is False
    assert config.top_k == 3
    assert config.score_threshold == 0.5
    assert config.min_content_length == 5
    assert config.embedding_model == "custom"
    assert config.vector_store_path == "/tmp/path"
    assert config.injection_role == "user"

def test_memory_manager_force_store_short_snippet(tmp_path):
    store_path = tmp_path / "memory.json"
    manager = MemoryManager(
        dummy_embedder,
        config=MemoryManagerConfig(enabled=True, vector_store_path=str(store_path)),
    )

    record = manager.remember("Call me Bo", metadata={"note": "Preferred name", "force_store": True})
    assert record is not None
    assert record.metadata["note"] == "Preferred name"

    manager.flush()
    assert store_path.exists()


def test_memory_manager_recall_returns_ranked_results(tmp_path):
    manager = MemoryManager(
        dummy_embedder,
        config=MemoryManagerConfig(enabled=True, top_k=2),
        vector_store=LocalVectorStore(),
    )
    manager.remember("Remember that experiment A is sensitive", metadata={"note": "Experiment tip"})
    manager.remember("Keep in mind the threshold is 0.7", metadata={"note": "Threshold"})

    results = manager.recall("threshold guidance")
    assert len(results) == 2
    assert results[0].score >= results[1].score
    formatted = manager.format_results(results)
    assert "Relevant stored context" in formatted
