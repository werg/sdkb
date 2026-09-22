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
    deleted: np.ndarray


class PublishedKeyIndex:
    """Load complete raw keys once; search with the SQLite reference semantics."""

    def __init__(self, store: DiskStore, *, namespace: str, generation: str,
                 spaces: tuple[str, ...], expected_sources: int):
        if not namespace or not generation or not spaces or expected_sources < 1:
            raise ValueError('Published index needs a scoped nonempty generation')
        self.namespace, self.generation = namespace, generation
        self.spaces = {}
        common_ids = None
        for space in spaces:
            with store.connect() as db:
                rows = db.execute('''SELECT record_id,key,key_dim,domain,created_at,deleted
                                     FROM records WHERE namespace=? AND generation=? AND space=?
                                     ORDER BY record_id''', (namespace, generation, space)).fetchall()
            if len(rows) != expected_sources:
                raise ValueError('Published bank space is not complete')
            ids = np.asarray([row[0] for row in rows])
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
                                        np.asarray([row[3] for row in rows]),
                                        np.asarray([row[4] for row in rows], dtype=np.int64),
                                        np.asarray([bool(row[5]) for row in rows]))
        if len(self.spaces) != len(spaces):
            raise ValueError('Published spaces must be distinct')
        self.key_bytes = sum(array.keys.nbytes for array in self.spaces.values())

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
            ))
        return tuple(plans)
