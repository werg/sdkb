"""Resident exact key arrays for a verified physical bank snapshot.

This is a CPU exact scan, not ANN. Stored payloads remain in DiskStore and every
fetch revalidates namespace, domain, physical revision, time and deletion status.
Spatial training may patch these arrays from its checkpointed mutable overlay.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from .store import DiskStore, ReadPlan, Selection


@dataclass
class _Space:
    ids: np.ndarray
    keys: np.ndarray
    domains: np.ndarray
    times: np.ndarray
    source_ids: np.ndarray
    deleted: np.ndarray


class PublishedKeyIndex:
    """Load complete raw keys once; search with the SQLite reference semantics."""

    def __init__(self, store: DiskStore, *, namespace: str, generation: str,
                 spaces: tuple[str, ...], expected_sources: int):
        if not namespace or not generation or not spaces or expected_sources < 1:
            raise ValueError('Published index needs a scoped nonempty generation')
        self.namespace, self.generation = namespace, generation
        self.bank_cursor: int | None = None
        self._space_names = spaces
        self._expected_sources = expected_sources
        self.reload(store)

    def reload(self, store: DiskStore) -> None:
        """Restore the physical base view before applying journal revisions."""
        self.spaces = {}
        common_ids = None
        for space in self._space_names:
            with store.connect() as db:
                rows = db.execute('''SELECT record_id,key,key_dim,domain,created_at,source_id,deleted
                                     FROM records WHERE namespace=? AND generation=? AND space=?
                                     ORDER BY record_id''',
                                  (self.namespace, self.generation, space)).fetchall()
            if len(rows) != self._expected_sources:
                raise ValueError('Published bank space is not complete')
            ids = np.asarray([row[0] for row in rows], dtype=object)
            if common_ids is not None and not np.array_equal(ids, common_ids):
                raise ValueError('Published bank spaces have different logical source IDs')
            common_ids = ids
            dimensions = {row[2] for row in rows}
            if len(dimensions) != 1:
                raise ValueError('Published key dimensions differ within one space')
            keys = np.stack([np.frombuffer(row[1], dtype='<f4') for row in rows]).copy()
            if keys.shape[1] != next(iter(dimensions)) or not np.isfinite(keys).all():
                raise ValueError('Published key bytes are incompatible or nonfinite')
            keys /= np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12)
            self.spaces[space] = _Space(ids, keys,
                                        np.asarray([row[3] for row in rows], dtype=object),
                                        np.asarray([row[4] for row in rows], dtype=np.int64),
                                        np.asarray([row[5] for row in rows], dtype=object),
                                        np.asarray([bool(row[6]) for row in rows]))
        if len(self.spaces) != len(self._space_names):
            raise ValueError('Published spaces must be distinct')
        self.key_bytes = sum(array.keys.nbytes for array in self.spaces.values())

    def upsert(self, space: str, record_id: str, key: bytes, key_dim: int, *,
               domain: str, created_at: int, source_id: str = '',
               deleted: bool = False) -> None:
        """Apply one committed journal head to the resident exact index."""
        if space not in self.spaces or not record_id or not domain or created_at < 0:
            raise ValueError('Invalid mutable index revision')
        vector = np.frombuffer(key, dtype='<f4').copy()
        array = self.spaces[space]
        if vector.shape != (key_dim,) or key_dim != array.keys.shape[1]:
            raise ValueError('Mutable key dimension differs from the base index')
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm <= 0:
            raise ValueError('Mutable key is nonfinite or zero')
        vector /= norm
        positions = np.flatnonzero(array.ids == record_id)
        if len(positions) > 1:
            raise ValueError('Resident index contains duplicate logical IDs')
        if len(positions) == 1:
            position = int(positions[0])
            array.keys[position] = vector
            array.domains[position] = domain
            array.times[position] = created_at
            if source_id:
                array.source_ids[position] = source_id
            array.deleted[position] = deleted
        else:
            array.ids = np.append(array.ids, record_id)
            array.keys = np.concatenate((array.keys, vector[None]), axis=0)
            array.domains = np.append(array.domains, domain)
            array.times = np.append(array.times, created_at)
            array.source_ids = np.append(array.source_ids, source_id)
            array.deleted = np.append(array.deleted, deleted)
            order = np.argsort(array.ids, kind='stable')
            array.ids, array.keys = array.ids[order], array.keys[order]
            array.domains, array.times = array.domains[order], array.times[order]
            array.source_ids = array.source_ids[order]
            array.deleted = array.deleted[order]
        self.key_bytes = sum(item.keys.nbytes for item in self.spaces.values())

    def upsert_many(self, space: str, rows: Sequence[tuple]) -> None:
        """Apply a journal snapshot with one vectorized existing-ID patch."""
        if space not in self.spaces or not rows:
            return
        array = self.spaces[space]
        ids = np.asarray([row[0] for row in rows], dtype=object)
        if len(set(ids)) != len(ids):
            raise ValueError('Mutable index batch contains duplicate logical IDs')
        dimensions = {row[2] for row in rows}
        if dimensions != {array.keys.shape[1]}:
            raise ValueError('Mutable key dimension differs from the base index')
        keys = np.stack([np.frombuffer(row[1], dtype='<f4') for row in rows]).copy()
        norms = np.linalg.norm(keys, axis=1, keepdims=True)
        if not np.isfinite(keys).all() or np.any(norms <= 0):
            raise ValueError('Mutable key is nonfinite or zero')
        keys /= norms
        positions = np.searchsorted(array.ids, ids)
        existing = positions < len(array.ids)
        existing &= np.asarray([
            array.ids[position] == record_id if position < len(array.ids) else False
            for position, record_id in zip(positions, ids, strict=True)
        ])
        old_positions = positions[existing]
        array.keys[old_positions] = keys[existing]
        array.domains[old_positions] = np.asarray(
            [row[3] for row, present in zip(rows, existing, strict=True) if present],
            dtype=object)
        array.times[old_positions] = np.asarray(
            [row[4] for row, present in zip(rows, existing, strict=True) if present],
            dtype=np.int64)
        sources = np.asarray(
            [row[5] for row, present in zip(rows, existing, strict=True) if present],
            dtype=object)
        replace_source = sources != ''
        array.source_ids[old_positions[replace_source]] = sources[replace_source]
        array.deleted[old_positions] = np.asarray(
            [row[6] for row, present in zip(rows, existing, strict=True) if present],
            dtype=bool)
        if not np.all(existing):
            fresh = ~existing
            array.ids = np.concatenate((array.ids, ids[fresh]))
            array.keys = np.concatenate((array.keys, keys[fresh]), axis=0)
            array.domains = np.concatenate((array.domains, np.asarray(
                [row[3] for row, present in zip(rows, fresh, strict=True) if present],
                dtype=object)))
            array.times = np.concatenate((array.times, np.asarray(
                [row[4] for row, present in zip(rows, fresh, strict=True) if present],
                dtype=np.int64)))
            array.source_ids = np.concatenate((array.source_ids, np.asarray(
                [row[5] for row, present in zip(rows, fresh, strict=True) if present],
                dtype=object)))
            array.deleted = np.concatenate((array.deleted, np.asarray(
                [row[6] for row, present in zip(rows, fresh, strict=True) if present],
                dtype=bool)))
            order = np.argsort(array.ids, kind='stable')
            array.ids, array.keys = array.ids[order], array.keys[order]
            array.domains, array.times = array.domains[order], array.times[order]
            array.source_ids, array.deleted = (
                array.source_ids[order], array.deleted[order])
        self.key_bytes = sum(item.keys.nbytes for item in self.spaces.values())

    def set_deleted(self, record_ids: set[str], deleted: bool) -> None:
        for array in self.spaces.values():
            if record_ids:
                array.deleted[np.isin(array.ids, tuple(record_ids))] = deleted

    def keys_for_ids(self, space: str, record_ids: Sequence[str], *, domain: str,
                     query_time: int) -> Tensor:
        """Return fixed candidate keys while enforcing the index visibility snapshot."""
        if space not in self.spaces or not record_ids:
            raise ValueError('Published key lookup needs a known space and record IDs')
        array = self.spaces[space]
        positions = np.searchsorted(array.ids, np.asarray(record_ids))
        if any(position >= len(array.ids) or array.ids[position] != record_id
               for position, record_id in zip(positions, record_ids, strict=True)):
            raise KeyError('Record ID is absent from the published generation')
        if any(array.domains[position] != domain or array.times[position] >= query_time
               or array.deleted[position] for position in positions):
            raise PermissionError('Record key is outside its causal authorization scope')
        return torch.from_numpy(array.keys[positions].copy())

    def search(self, query: Tensor, *, top_k: int = 16, namespace: str = 'corpus',
               space: str = 's0', generation: str = '', domain: str = 'research',
               query_time: int = 2**62, exclude_ids: frozenset[str] = frozenset()) -> ReadPlan:
        return self.search_batch(
            query[None], top_k=top_k, namespace=namespace, space=space,
            generation=generation, domains=(domain,), query_times=(query_time,),
            exclude_ids=(exclude_ids,),
        )[0]

    def search_batch(self, queries: Tensor, *, top_k: int = 16,
                     namespace: str = 'corpus', space: str = 's0', generation: str = '',
                     domains: Sequence[str], query_times: Sequence[int],
                     exclude_ids: Sequence[frozenset[str]] | None = None) -> tuple[ReadPlan, ...]:
        """Search several site queries with one dense key matrix operation."""
        if namespace != self.namespace or generation != self.generation or space not in self.spaces:
            raise ValueError('Key index scope differs from its published generation')
        if (queries.ndim != 2 or not torch.isfinite(queries).all() or top_k < 0
                or len(domains) != queries.shape[0] or len(query_times) != queries.shape[0]):
            raise ValueError('Invalid published batched key search request')
        excluded = tuple(exclude_ids or (frozenset(),) * queries.shape[0])
        if len(excluded) != queries.shape[0]:
            raise ValueError('One exclusion set is required per published query')
        array = self.spaces[space]
        q = queries.detach().float().cpu().numpy().copy()
        if q.shape[1] != array.keys.shape[1]:
            raise ValueError('Key dimensions incompatible with query generation')
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        scores = q @ array.keys.T
        plans = []
        for row, (domain, query_time, omitted) in enumerate(
                zip(domains, query_times, excluded, strict=True)):
            eligible = (array.domains == domain) & (array.times < query_time) & ~array.deleted
            if omitted:
                eligible &= ~np.isin(array.ids, tuple(omitted))
            indices = np.flatnonzero(eligible)
            order = np.lexsort((array.ids[indices], -scores[row, indices]))[:top_k]
            chosen = indices[order]
            plans.append(ReadPlan(
                namespace, space, generation, domain, query_time,
                tuple(Selection(str(array.ids[i]), float(scores[row, i])) for i in chosen),
                self.bank_cursor,
            ))
        return tuple(plans)
