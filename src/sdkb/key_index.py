"""Resident exact key arrays for an immutable published bank generation.

This is a CPU exact scan, not ANN. Stored payloads remain in DiskStore and every
fetch revalidates namespace, domain, generation, time and deletion status.
"""
from __future__ import annotations

from dataclasses import dataclass

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

    def search(self, query: Tensor, *, top_k: int = 16, namespace: str = 'corpus',
               space: str = 's0', generation: str = '', domain: str = 'research',
               query_time: int = 2**62, exclude_ids: frozenset[str] = frozenset()) -> ReadPlan:
        if namespace != self.namespace or generation != self.generation or space not in self.spaces:
            raise ValueError('Key index scope differs from its published generation')
        if query.ndim != 1 or not torch.isfinite(query).all() or top_k < 0:
            raise ValueError('Invalid published key search request')
        if top_k == 0:
            return ReadPlan(namespace, space, generation, domain, query_time, ())
        array = self.spaces[space]
        q = query.detach().float().cpu().numpy().copy()
        if q.size != array.keys.shape[1]:
            raise ValueError('Key dimensions incompatible with query generation')
        q /= max(float(np.linalg.norm(q)), 1e-12)
        eligible = (array.domains == domain) & (array.times < query_time) & ~array.deleted
        if exclude_ids:
            eligible &= ~np.isin(array.ids, tuple(exclude_ids))
        scores = array.keys @ q
        indices = np.flatnonzero(eligible)
        order = np.lexsort((array.ids[indices], -scores[indices]))[:top_k]
        chosen = indices[order]
        return ReadPlan(namespace, space, generation, domain, query_time,
                        tuple(Selection(str(array.ids[i]), float(scores[i])) for i in chosen))
