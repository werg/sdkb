"""Transactional disk payload store and a streaming exact-search reference.

SQLite stores safetensors payloads, not Python pickle. Search scans eligible keys
on CPU in bounded chunks; it is a correctness baseline, not an ANN index or a
claim of cold-disk throughput. No source trajectory is exposed on the read path.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch
from torch import Tensor
from safetensors.torch import load, save


@dataclass(frozen=True)
class Selection:
    record_id: str
    score: float


@dataclass(frozen=True)
class ReadPlan:
    namespace: str
    space: str
    generation: str
    domain: str
    query_time: int
    selections: tuple[Selection, ...]


@dataclass
class StoredRecord:
    record_id: str
    key: Tensor
    payload: Tensor
    namespace: str = "default"
    space: str = "s0"
    generation: str = "v0"
    domain: str = "research"
    created_at: int = 0
    source_id: str = ""


class DiskStore:
    def __init__(self, path: str | Path, *, cache_mib: int = 32) -> None:
        self.path = Path(path)
        if cache_mib < 1:
            raise ValueError("SQLite cache budget must be positive")
        self.cache_mib = cache_mib
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS records (
                namespace TEXT NOT NULL, record_id TEXT NOT NULL, space TEXT NOT NULL,
                generation TEXT NOT NULL, domain TEXT NOT NULL, created_at INTEGER NOT NULL,
                key BLOB NOT NULL, key_dim INTEGER NOT NULL, payload BLOB NOT NULL,
                source_id TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(namespace,record_id,space,generation));
            CREATE INDEX IF NOT EXISTS eligible ON records
                (namespace,space,generation,domain,deleted,created_at);
            CREATE TABLE IF NOT EXISTS lineage (
                namespace TEXT NOT NULL, parent_id TEXT NOT NULL, child_id TEXT NOT NULL,
                PRIMARY KEY(namespace,parent_id,child_id));
            CREATE TABLE IF NOT EXISTS tombstones (
                namespace TEXT NOT NULL, record_id TEXT NOT NULL,
                PRIMARY KEY(namespace,record_id));
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute(f"PRAGMA cache_size={-self.cache_mib * 1024}")
        db.execute("PRAGMA mmap_size=0")  # no unaccounted application mmap cache
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, record: StoredRecord, *, children: tuple[str, ...] = ()) -> None:
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._put(db, record, children)

    def put_many(self, records: Iterable[StoredRecord]) -> None:
        """Stream raw records through one atomic transaction for offline bank creation.

        No partial batch is visible or retained after an exception. Derived
        compact records still use put(..., children=...) to record their lineage.
        """
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for record in records:
                self._put(db, record, ())

    @staticmethod
    def _event_digest(records: list[StoredRecord]) -> str:
        digest = hashlib.sha256()
        for record in sorted(records, key=lambda item: (item.record_id, item.space)):
            key = record.key.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
            payload = save({"payload": record.payload.detach().cpu().contiguous()})
            metadata = (record.namespace, record.record_id, record.space, record.generation,
                        record.domain, record.created_at, record.source_id)
            digest.update(json.dumps(metadata, separators=(",", ":")).encode())
            digest.update(key.tobytes())
            digest.update(payload)
        return digest.hexdigest()

    @staticmethod
    def _ensure_event_tables(db) -> None:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS event_streams (
                namespace TEXT NOT NULL, stream TEXT NOT NULL, generation TEXT NOT NULL,
                next_position INTEGER NOT NULL, visibility_time INTEGER NOT NULL,
                PRIMARY KEY(namespace,stream,generation));
            CREATE TABLE IF NOT EXISTS event_commits (
                namespace TEXT NOT NULL, stream TEXT NOT NULL, generation TEXT NOT NULL,
                position INTEGER NOT NULL, event_id TEXT NOT NULL, visibility_time INTEGER NOT NULL,
                record_ids TEXT NOT NULL, content_sha256 TEXT NOT NULL,
                PRIMARY KEY(namespace,stream,generation,position),
                UNIQUE(namespace,stream,generation,event_id));
        """)

    def commit_event(self, records: Iterable[StoredRecord], *, namespace: str, stream: str,
                     generation: str, position: int, event_id: str, visibility_time: int,
                     expected_spaces: tuple[str, ...]) -> bool:
        """Atomically advance a causal stream and publish all views of its writes.

        Returns false for an exact idempotent retry. The caller finishes event
        ``position`` before committing it; later searches use a strictly greater
        ``query_time`` to observe these records.
        """
        rows = list(records)
        if (not namespace or not stream or not generation or not event_id or position < 0
                or visibility_time < 0 or not expected_spaces
                or len(expected_spaces) != len(set(expected_spaces))):
            raise ValueError("Invalid temporal event commit")
        logical: dict[str, set[str]] = {}
        for record in rows:
            if (record.namespace != namespace or record.generation != generation
                    or record.created_at != visibility_time or record.space not in expected_spaces):
                raise ValueError("Event records differ from their visibility scope")
            logical.setdefault(record.record_id, set()).add(record.space)
        if any(spaces != set(expected_spaces) for spaces in logical.values()):
            raise ValueError("Every event record needs every expected space view")
        if len(rows) != len(logical) * len(expected_spaces):
            raise ValueError("Event contains duplicate record-space views")
        content_hash = self._event_digest(rows)
        record_ids = json.dumps(sorted(logical), separators=(",", ":"))
        with self.connect() as db:
            self._ensure_event_tables(db)
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("""SELECT position,event_id,visibility_time,record_ids,content_sha256
                FROM event_commits WHERE namespace=? AND stream=? AND generation=?
                AND (position=? OR event_id=?)""",
                (namespace, stream, generation, position, event_id)).fetchall()
            expected = (position, event_id, visibility_time, record_ids, content_hash)
            if prior:
                if len(prior) == 1 and tuple(prior[0]) == expected:
                    return False
                raise ValueError("Temporal event position or identity was already committed differently")
            state = db.execute("""SELECT next_position,visibility_time FROM event_streams
                WHERE namespace=? AND stream=? AND generation=?""",
                (namespace, stream, generation)).fetchone()
            if state is None:
                if position != 0:
                    raise ValueError("A temporal stream must begin at position zero")
            elif state[0] != position or visibility_time < state[1]:
                raise ValueError("Temporal events must commit once in monotonic order")
            for record in rows:
                self._put(db, record, ())
            db.execute("INSERT INTO event_commits VALUES (?,?,?,?,?,?,?,?)", (
                namespace, stream, generation, position, event_id, visibility_time,
                record_ids, content_hash))
            db.execute("""INSERT INTO event_streams VALUES (?,?,?,?,?)
                ON CONFLICT(namespace,stream,generation) DO UPDATE SET
                next_position=excluded.next_position, visibility_time=excluded.visibility_time""",
                (namespace, stream, generation, position + 1, visibility_time))
        return True

    def event_frontier(self, *, namespace: str, stream: str, generation: str) -> dict:
        with self.connect() as db:
            exists = db.execute("""SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='event_streams'""").fetchone()
            if exists is None:
                return {"next_position": 0, "visibility_time": -1}
            row = db.execute("""SELECT next_position,visibility_time FROM event_streams
                WHERE namespace=? AND stream=? AND generation=?""",
                (namespace, stream, generation)).fetchone()
        return {"next_position": 0, "visibility_time": -1} if row is None else {
            "next_position": row[0], "visibility_time": row[1]}

    def _put(self, db, record: StoredRecord, children: tuple[str, ...]) -> None:
        key = record.key.detach().float().cpu().contiguous()
        value = record.payload.detach().cpu().contiguous()
        if key.ndim != 1 or not value.is_floating_point() or not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError("Finite vector key and floating payload required")
        if not record.record_id or not record.namespace or not record.generation:
            raise ValueError("Record identity and version cannot be empty")
        key_blob = key.numpy().astype("<f4", copy=False).tobytes()
        value_blob = save({"payload": value})
        if db.execute("SELECT 1 FROM tombstones WHERE namespace=? AND record_id=?",
                      (record.namespace, record.record_id)).fetchone():
            raise ValueError("Deleted IDs cannot be resurrected; use a new generation ID")
        for child in children:
            rows = db.execute(
                "SELECT domain,deleted FROM records WHERE namespace=? AND record_id=?",
                (record.namespace, child)).fetchall()
            if not rows or any(d != record.domain or deleted for d, deleted in rows):
                raise PermissionError("Compaction requires live authorization-homogeneous children")
            if child == record.record_id:
                raise ValueError("A record cannot compact itself")
        # Immutable versions: overwriting a payload would invalidate captured plans.
        db.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,0)", (
            record.namespace, record.record_id, record.space, record.generation,
            record.domain, record.created_at, key_blob, key.numel(), value_blob, record.source_id))
        for child in children:
            db.execute("INSERT OR IGNORE INTO lineage VALUES (?,?,?)",
                       (record.namespace, record.record_id, child))

    def search(self, query: Tensor, *, top_k: int = 16, namespace: str = "default",
               space: str = "s0", generation: str = "v0", domain: str = "research",
               query_time: int = 2**62, key_chunk_size: int = 1024,
               exclude_ids: frozenset[str] = frozenset()) -> ReadPlan:
        if query.ndim != 1 or not torch.isfinite(query).all() or top_k < 0 or key_chunk_size < 1:
            raise ValueError("Invalid search parameters")
        if top_k == 0:
            return ReadPlan(namespace, space, generation, domain, query_time, ())
        q = query.detach().float().cpu().numpy()
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        best: list[tuple[float, str]] = []
        with self.connect() as db:
            cursor = db.execute("""SELECT record_id,key,key_dim FROM records
                WHERE namespace=? AND space=? AND generation=? AND domain=?
                AND created_at<? AND deleted=0 ORDER BY record_id""",
                (namespace, space, generation, domain, query_time))
            while rows := cursor.fetchmany(key_chunk_size):
                if any(dim != len(q) for _, _, dim in rows):
                    raise ValueError("Key dimensions incompatible with query generation")
                keys = np.stack([np.frombuffer(b, dtype="<f4") for _, b, _ in rows])
                keys /= np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12)
                scores = keys @ q
                best.extend((float(score), row[0]) for row, score in zip(rows, scores, strict=True)
                            if row[0] not in exclude_ids)
                best = sorted(best, key=lambda pair: (-pair[0], pair[1]))[:top_k]
        return ReadPlan(namespace, space, generation, domain, query_time,
                        tuple(Selection(record_id, score) for score, record_id in best))

    def fetch(self, plan: ReadPlan) -> list[Tensor]:
        """Revalidate visibility at use time; deletion invalidates outstanding plans."""
        return self.fetch_many((plan,))[0]

    def fetch_many(self, plans: Iterable[ReadPlan]) -> list[list[Tensor]]:
        """Revalidate and load several captured plans through one connection."""
        result = []
        with self.connect() as db:
            for plan in plans:
                values = []
                for selection in plan.selections:
                    row = db.execute("""SELECT payload FROM records WHERE namespace=? AND record_id=?
                        AND space=? AND generation=? AND domain=? AND created_at<? AND deleted=0""",
                        (plan.namespace, selection.record_id, plan.space, plan.generation,
                         plan.domain, plan.query_time)).fetchone()
                    if row is None:
                        raise KeyError(
                            f"Unavailable, stale or unauthorized record: {selection.record_id}")
                    values.append(load(row[0])["payload"])
                result.append(values)
        return result

    def iter_fetch(self, plan: ReadPlan, chunk_size: int = 32):
        """Reload and revalidate at most one selected chunk at a time on the host."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        for start in range(0, len(plan.selections), chunk_size):
            part = ReadPlan(plan.namespace, plan.space, plan.generation, plan.domain,
                            plan.query_time, plan.selections[start:start + chunk_size])
            values = self.fetch(part)
            yield torch.stack(values)[None], torch.ones(1, len(values))

    def delete(self, namespace: str, record_id: str) -> set[str]:
        """Tombstone a source and invalidate all transitive compacted derivatives."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("""WITH RECURSIVE affected(id) AS (
                SELECT ? UNION SELECT lineage.parent_id FROM lineage JOIN affected
                ON lineage.child_id=affected.id WHERE lineage.namespace=?)
                SELECT id FROM affected""", (record_id, namespace)).fetchall()
            affected = {row[0] for row in rows}
            for rid in affected:
                db.execute("UPDATE records SET deleted=1 WHERE namespace=? AND record_id=?", (namespace, rid))
                db.execute("INSERT OR IGNORE INTO tombstones VALUES (?,?)", (namespace, rid))
        return affected

    def sizes(self) -> dict[str, int]:
        with self.connect() as db:
            count, payload, keys = db.execute("""SELECT count(*),coalesce(sum(length(payload)),0),
                coalesce(sum(length(key)),0) FROM records WHERE deleted=0""").fetchone()
        physical = sum(p.stat().st_size for p in self.path.parent.glob(self.path.name + "*") if p.is_file())
        return {"records": count, "serialized_payload_bytes": payload,
                "key_bytes": keys, "physical_sqlite_bytes": physical,
                "sqlite_cache_budget_bytes_per_connection": self.cache_mib * 1024**2}


class AsyncRetriever:
    """CPU/disk future API only. No automatic backbone scheduling or latency claim."""
    def __init__(self, store: DiskStore, workers: int = 2) -> None:
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def submit(self, query: Tensor, **kwargs) -> Future:
        query = query.detach().float().cpu().clone()
        def read():
            plan = self.store.search(query, **kwargs)
            return plan, self.store.fetch(plan)
        return self.pool.submit(read)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> AsyncRetriever:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def lookup_record(store: DiskStore, record_id: str, *, namespace: str, space: str,
                  generation: str, domain: str = "research", query_time: int = 2**62) -> StoredRecord:
    """Fetch one immutable compatible record (used for training caches and manifests)."""
    with store.connect() as db:
        row = db.execute("""SELECT key,payload,created_at,source_id FROM records
            WHERE namespace=? AND record_id=? AND space=? AND generation=? AND domain=?
            AND created_at<? AND deleted=0""",
            (namespace, record_id, space, generation, domain, query_time)).fetchone()
    if row is None:
        raise KeyError(record_id)
    key = torch.from_numpy(np.frombuffer(row[0], dtype="<f4").copy())
    return StoredRecord(record_id, key, load(row[1])["payload"], namespace, space,
                        generation, domain, row[2], row[3])
