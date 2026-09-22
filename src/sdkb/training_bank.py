"""Checkpointed mutable key/payload overlays for continuous bank learning.

The published snapshot is currently a frozen physical recovery base. Training
updates materialized logical records in the overlay; the ordinary inference API
sees the newest stored tensors and never re-encodes a source trajectory. This is a
transition toward a durable mutation journal rather than the target bank lifecycle.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from safetensors.torch import load, save

from .key_index import PublishedKeyIndex
from .store import DiskStore, ReadPlan, StoredRecord


class TrainingBank:
    def __init__(self, base: DiskStore, cache: DiskStore, index: PublishedKeyIndex) -> None:
        self.base, self.cache, self.index = base, cache, index
        self._positions = {
            space: {str(record_id): position for position, record_id in enumerate(array.ids)}
            for space, array in index.spaces.items()
        }
        with cache.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS training_bank_overlay (
                space TEXT NOT NULL, record_id TEXT NOT NULL, key BLOB NOT NULL,
                key_dim INTEGER NOT NULL, payload BLOB NOT NULL,
                PRIMARY KEY(space,record_id))''')
            rows = db.execute(
                'SELECT space,record_id,key,key_dim FROM training_bank_overlay').fetchall()
        for space, record_id, blob, dimension in rows:
            self._install_key(space, record_id, blob, dimension)

    def _install_key(self, space: str, record_id: str, blob: bytes,
                     dimension: int) -> None:
        if space not in self.index.spaces or record_id not in self._positions[space]:
            raise ValueError('Training overlay refers to an unknown bank record')
        key = np.frombuffer(blob, dtype='<f4').copy()
        if key.shape != (dimension,) or dimension != self.index.spaces[space].keys.shape[1]:
            raise ValueError('Training overlay key dimension changed')
        norm = np.linalg.norm(key)
        if not np.isfinite(key).all() or norm <= 0:
            raise ValueError('Training overlay contains an invalid key')
        self.index.spaces[space].keys[self._positions[space][record_id]] = key / norm

    def fetch_many(self, plans: Iterable[ReadPlan]) -> list[list[torch.Tensor]]:
        plans = tuple(plans)
        base_rows = self.base.fetch_many(plans)
        requested: dict[str, set[str]] = {}
        for plan in plans:
            requested.setdefault(plan.space, set()).update(
                selection.record_id for selection in plan.selections)
        overlay = {}
        with self.cache.connect() as db:
            for space, record_ids in requested.items():
                ordered = sorted(record_ids)
                for start in range(0, len(ordered), 900):
                    chunk = ordered[start:start + 900]
                    placeholders = ','.join('?' for _ in chunk)
                    rows = db.execute(
                        f'''SELECT record_id,payload FROM training_bank_overlay
                            WHERE space=? AND record_id IN ({placeholders})''',
                        (space, *chunk)).fetchall()
                    overlay.update(((space, record_id), load(blob)['payload'])
                                   for record_id, blob in rows)
        for row, plan in zip(base_rows, plans, strict=True):
            for position, selection in enumerate(plan.selections):
                value = overlay.get((plan.space, selection.record_id))
                if value is not None:
                    row[position] = value
        return base_rows

    def update(self, records: Iterable[StoredRecord]) -> int:
        encoded = []
        for record in records:
            if (record.space not in self.index.spaces
                    or record.record_id not in self._positions[record.space]):
                raise KeyError('Cannot update an unknown published record')
            key = record.key.detach().float().cpu().contiguous()
            payload = record.payload.detach().cpu().contiguous()
            if (key.ndim != 1 or not torch.isfinite(key).all()
                    or not payload.is_floating_point() or not torch.isfinite(payload).all()):
                raise ValueError('Overlay updates need finite vector keys and payloads')
            key = torch.nn.functional.normalize(key, dim=-1)
            blob = key.numpy().astype('<f4', copy=False).tobytes()
            encoded.append((record.space, record.record_id, blob, key.numel(),
                            save({'payload': payload})))
        if not encoded:
            return 0
        with self.cache.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.executemany('''INSERT INTO training_bank_overlay VALUES (?,?,?,?,?)
                              ON CONFLICT(space,record_id) DO UPDATE SET
                              key=excluded.key,key_dim=excluded.key_dim,
                              payload=excluded.payload''', encoded)
        for space, record_id, blob, dimension, _ in encoded:
            self._install_key(space, record_id, blob, dimension)
        return len(encoded)

    def sizes(self) -> dict[str, int]:
        with self.cache.connect() as db:
            count, payload_bytes = db.execute(
                'SELECT COUNT(*),COALESCE(SUM(LENGTH(payload)+LENGTH(key)),0) '
                'FROM training_bank_overlay').fetchone()
        return {'views': count, 'tensor_bytes': payload_bytes}
