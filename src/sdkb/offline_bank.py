"""Atomic identity and integrity checks for resumable frozen offline writes."""
from collections.abc import Callable, Iterable
import hashlib
import json

from .store import DiskStore, StoredRecord


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _contents(db, namespace, generation, spaces):
    digest, count = hashlib.sha256(), 0
    placeholders = ','.join('?' for _ in spaces)
    rows = db.execute(f'''SELECT * FROM records WHERE namespace=? AND generation=?
                         AND space IN ({placeholders}) ORDER BY space,record_id''',
                      (namespace, generation, *spaces))
    for row in rows:
        count += 1
        for value in row:
            encoded = value if isinstance(value, bytes) else canonical_json(value).encode()
            digest.update(len(encoded).to_bytes(8, 'big'))
            digest.update(encoded)
    return count, digest.hexdigest()


def ensure_offline_records(store: DiskStore, factory: Callable[[], Iterable[StoredRecord]], *,
                           identity: dict, namespace: str, generation: str,
                           spaces: tuple[str, ...], expected_count: int) -> bool:
    """Return True only when this call writes a new bank.

    Caller supplies verified frozen-writer identity and source/config identity.
    Records and manifest commit together. Resuming validates the raw bytes without
    invoking factory; derived code views are outside this raw-bank digest. Existing
    banks without this manifest fail closed: use a new path for an explicit offline
    rebuild. This helper neither adopts legacy data nor changes immutable inserts.
    """
    if not spaces or len(spaces) != len(set(spaces)) or expected_count < 0:
        raise ValueError('Distinct raw spaces and nonnegative count required')
    serialized = canonical_json(identity)
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('''CREATE TABLE IF NOT EXISTS offline_banks (
            namespace TEXT NOT NULL, generation TEXT NOT NULL,
            identity TEXT NOT NULL, records INTEGER NOT NULL, digest TEXT NOT NULL,
            PRIMARY KEY(namespace,generation))''')
        previous = db.execute('SELECT identity,records,digest FROM offline_banks WHERE namespace=? AND generation=?',
                              (namespace, generation)).fetchone()
        if previous is not None and previous[0] != serialized:
            raise ValueError('Offline bank identity changed')
        count, digest = _contents(db, namespace, generation, spaces)
        if previous is not None:
            if (count, digest) != previous[1:] or count != expected_count:
                raise ValueError('Offline bank raw contents changed')
            return False
        if count:
            raise ValueError('Existing bank has no offline manifest; rebuild offline at a new path')
        for record in factory():
            if record.namespace != namespace or record.generation != generation or record.space not in spaces:
                raise ValueError('Offline writer emitted an unexpected record scope')
            store._put(db, record, ())
        count, digest = _contents(db, namespace, generation, spaces)
        if count != expected_count:
            raise ValueError('Offline writer record count differs')
        db.execute('INSERT INTO offline_banks VALUES (?,?,?,?,?)',
                   (namespace, generation, serialized, count, digest))
    return True
