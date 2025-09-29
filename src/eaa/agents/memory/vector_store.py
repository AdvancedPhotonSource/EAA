from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Protocol, Sequence
import json
import logging
import time

import numpy as np

try:  # pragma: no cover - import guard for optional dependency
    import psycopg  # type: ignore
    from psycopg.rows import dict_row  # type: ignore
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore
    dict_row = None  # type: ignore

from .types import MemoryQueryResult, MemoryRecord

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Protocol describing the minimum surface of a vector store."""

    def add(self, records: Sequence[MemoryRecord]) -> None:
        ...

    def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[MemoryQueryResult]:
        ...

    def delete(self, record_ids: Sequence[str]) -> None:
        ...

    def persist(self) -> None:
        ...


class LocalVectorStore:
    """Simple vector store that keeps data locally and optionally persists to disk.

    This is not optimised for large corpora but provides a dependency-free
    implementation that can be swapped out with more capable backends later.
    """

    def __init__(
        self,
        persist_path: Optional[str | Path] = None,
        auto_flush: bool = True,
    ) -> None:
        self._records: List["MemoryRecord"] = []
        self.persist_path = Path(persist_path) if persist_path else None
        self.auto_flush = auto_flush
        if self.persist_path and self.persist_path.exists():
            self._load()

    def add(self, records: Sequence["MemoryRecord"]) -> None:
        if not records:
            return
        self._records.extend(records)
        if self.auto_flush:
            self.persist()

    def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List["MemoryQueryResult"]:
        if not self._records:
            return []
        query = np.asarray(embedding, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("Embedding must be a 1D vector.")
        matrix = np.asarray([rec.embedding for rec in self._records], dtype=np.float32)
        if matrix.size == 0:
            return []

        # Normalize for cosine similarity.
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            logger.warning("Query embedding has zero norm; returning empty search result.")
            return []
        matrix_norms = np.linalg.norm(matrix, axis=1)
        matrix_norms[matrix_norms == 0] = 1e-9
        scores = (matrix @ query) / (matrix_norms * query_norm)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[MemoryQueryResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < score_threshold:
                continue
            record = self._records[idx]
            results.append(MemoryQueryResult(record=record, score=score))
        return results

    def delete(self, record_ids: Sequence[str]) -> None:
        if not record_ids:
            return
        id_set = set(record_ids)
        self._records = [rec for rec in self._records if rec.record_id not in id_set]
        if self.auto_flush:
            self.persist()

    def persist(self) -> None:
        if not self.persist_path:
            return
        data = [self._serialise_record(record) for record in self._records]
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def _load(self) -> None:
        if not self.persist_path:
            return
        try:
            with self.persist_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            logger.debug("No persisted memory found at %s", self.persist_path)
            return
        except json.JSONDecodeError as exc:
            logger.warning("Failed to load persisted memory: %s", exc)
            return

        records: List[MemoryRecord] = []
        for item in payload:
            try:
                records.append(self._deserialise_record(item))
            except Exception as exc:  # pragma: no cover - defensive.
                logger.warning("Could not deserialise memory record: %s", exc)
        self._records = records

    @staticmethod
    def _serialise_record(record: MemoryRecord) -> dict:
        data = asdict(record)
        # `embedding` may be a numpy array; ensure it is JSON serialisable.
        data["embedding"] = list(map(float, record.embedding))
        return data

    @staticmethod
    def _deserialise_record(payload: dict) -> MemoryRecord:
        created_at = payload.get("created_at")
        if created_at is None:
            created_at = time.time()
        return MemoryRecord(
            content=payload["content"],
            embedding=list(map(float, payload["embedding"])),
            metadata=payload.get("metadata", {}),
            record_id=payload.get("record_id"),
            created_at=created_at,
        )


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


class PostgresVectorStore(VectorStore):
    """Vector store backed by PostgreSQL + pgvector."""

    _METRIC_OPERATORS: Dict[str, str] = {
        "cosine": "<#>",
        "l2": "<->",
        "inner": "<.>",
    }

    def __init__(
        self,
        dsn: str,
        *,
        table_name: str = "memory_records",
        ensure_schema: bool = False,
        vector_dimension: Optional[int] = None,
        metric: str = "cosine",
    ) -> None:
        if psycopg is None:
            raise ImportError(
                "psycopg[binary] is required for PostgresVectorStore."
            )
        if metric not in self._METRIC_OPERATORS:
            raise ValueError(f"Unsupported metric '{metric}'.")
        self.dsn = dsn
        self.table_name = table_name
        self.metric = metric
        self._vector_operator = self._METRIC_OPERATORS[metric]
        self._connection_kwargs: Dict[str, str] = {}

        if ensure_schema:
            if vector_dimension is None:
                raise ValueError("vector_dimension must be provided when ensure_schema=True")
            self._ensure_schema(vector_dimension)

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.dsn, **self._connection_kwargs) as conn:  # type: ignore[arg-type]
            yield conn

    def add(self, records: Sequence[MemoryRecord]) -> None:
        if not records:
            return
        sql = (
            f"INSERT INTO {self.table_name} (record_id, content, embedding, metadata, created_at) "
            "VALUES (%s, %s, %s::vector, %s::jsonb, %s) "
            "ON CONFLICT (record_id) DO UPDATE SET "
            "content = EXCLUDED.content, embedding = EXCLUDED.embedding, "
            "metadata = EXCLUDED.metadata, created_at = EXCLUDED.created_at"
        )
        payload = [
            (
                record.record_id,
                record.content,
                _vector_literal(record.embedding),
                json.dumps(record.metadata or {}),
                datetime.fromtimestamp(record.created_at),
            )
            for record in records
        ]
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, payload)
            conn.commit()

    def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[MemoryQueryResult]:
        if not embedding:
            return []
        vector_literal = _vector_literal(embedding)
        operator = self._vector_operator
        sql = f"""
            SELECT
                record_id,
                content,
                metadata,
                created_at,
                embedding::float4[] AS embedding_array,
                1 - (embedding {operator} %s::vector) AS similarity
            FROM {self.table_name}
            ORDER BY embedding {operator} %s::vector ASC
            LIMIT %s
        """
        results: List[MemoryQueryResult] = []
        with self._connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:  # type: ignore[arg-type]
                cur.execute(sql, (vector_literal, vector_literal, top_k))
                rows = cur.fetchall()
        for row in rows:
            score = float(row["similarity"])
            if score < score_threshold:
                continue
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            created_at = row["created_at"].timestamp() if row["created_at"] else time.time()
            record = MemoryRecord(
                content=row["content"],
                embedding=[float(x) for x in row["embedding_array"]],
                metadata=metadata or {},
                record_id=str(row["record_id"]),
                created_at=created_at,
            )
            results.append(MemoryQueryResult(record=record, score=score))
        return results

    def delete(self, record_ids: Sequence[str]) -> None:
        if not record_ids:
            return
        sql = f"DELETE FROM {self.table_name} WHERE record_id = ANY(%s)"
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (record_ids,))
            conn.commit()

    def persist(self) -> None:
        return

    def _ensure_schema(self, dimension: int) -> None:
        create_extension = "CREATE EXTENSION IF NOT EXISTS vector"
        create_table = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                record_id UUID PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR({dimension}) NOT NULL,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """
        ivfflat_index = (
            f"CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_ivfflat "
            f"ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)"
        )
        metadata_index = (
            f"CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_gin "
            f"ON {self.table_name} USING gin (metadata)"
        )
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_extension)
                cur.execute(create_table)
                if self.metric == "cosine":
                    cur.execute(ivfflat_index)
                cur.execute(metadata_index)
            conn.commit()
