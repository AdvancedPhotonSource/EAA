import os
import uuid

import pytest

pytestmark = pytest.mark.local

try:
    from eaa.agents.memory import MemoryRecord, PostgresVectorStore
except ImportError:  # pragma: no cover - psycopg not installed
    PostgresVectorStore = None
    MemoryRecord = None


@pytest.fixture
def pg_store():
    if PostgresVectorStore is None:
        pytest.skip("psycopg not installed")
    dsn = os.getenv("EAA_TEST_PG_DSN")
    if not dsn:
        pytest.skip("EAA_TEST_PG_DSN not set")
    store = PostgresVectorStore(
        dsn,
        ensure_schema=True,
        vector_dimension=3,
    )
    yield store
    store.delete([str(uuid.UUID(int=0))])  # no-op cleanup call for interface sanity


def _record(content: str, embedding):
    return MemoryRecord(content=content, embedding=list(embedding), metadata={}, record_id=str(uuid.uuid4()))


@pytest.mark.local
def test_postgres_vector_store_roundtrip(pg_store):
    record = _record("Remember the threshold is 0.7", [0.1, 0.2, 0.3])
    pg_store.add([record])

    results = pg_store.search([0.1, 0.2, 0.3], top_k=1)
    assert results
    assert results[0].record.record_id == record.record_id

    pg_store.delete([record.record_id])
    assert not pg_store.search([0.1, 0.2, 0.3], top_k=1)

