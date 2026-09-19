"""Causally separated synthetic support/query episodes with counterfactual worlds.

No teacher API is required. These fixtures test information transfer and composition,
not real-world coding ability or capacity substitution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import random


@dataclass(frozen=True)
class Source:
    record_id: str
    text: str
    created_at: int
    kind: str


@dataclass(frozen=True)
class Episode:
    episode_id: str
    environment: str
    supports: tuple[Source, ...]
    query: str
    answer: str
    required_ids: tuple[str, ...]
    restore: bool
    allowed_capability: int
    capability: int
    query_time: int = 10


def _opaque(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def make_episode(seed: int, *, split: str = "train", distractors: int = 2,
                 restore: bool | None = None, allowed_capability: int | None = None,
                 capability: int | None = None) -> Episode:
    rng = random.Random(f"{split}:{seed}")
    environment = "env_" + _opaque(f"{split}:{seed}:env")[:8]
    restore = bool(rng.randrange(2)) if restore is None else restore
    allowed = rng.randrange(2) if allowed_capability is None else allowed_capability
    cap = rng.randrange(2) if capability is None else capability
    if allowed not in {0, 1} or cap not in {0, 1}:
        raise ValueError("Capabilities are binary")
    rid_a, rid_b = (_opaque(f"{split}:{seed}:{kind}") for kind in ("A", "B"))
    a = Source(rid_a, f"Experience in {environment}: after a failed attempt, "
               + ("restoring the snapshot is required before retry." if restore
                  else "restoring the snapshot is forbidden before retry."), 1, "restoration")
    b = Source(rid_b, f"Experience in {environment}: retry is permitted exactly when "
               f"capability={allowed}; otherwise stop.", 2, "permission")
    supports = [a, b]
    for i in range(distractors):
        decoy = "env_" + _opaque(f"{split}:{seed}:decoy:{i}")[:8]
        supports.append(Source(_opaque(f"{split}:{seed}:d:{i}"),
            f"Experience in {decoy}: retry requires capability={rng.randrange(2)}; "
            f"restoration is {'required' if rng.randrange(2) else 'forbidden'}.", 3 + i, "irrelevant"))
    rng.shuffle(supports)
    answer = "STOP" if cap != allowed else ("RESTORE_RETRY" if restore else "RETRY")
    query = (f"In {environment}, an attempt failed and capability={cap}. "
             "Using the environment's rules, choose exactly one action: STOP, RETRY, RESTORE_RETRY.")
    return Episode(_opaque(f"{split}:{seed}:query"), environment, tuple(supports), query, answer,
                   (rid_a, rid_b), restore, allowed, cap, 4 + distractors)


def counterfactual(episode: Episode, which: str) -> Episode:
    """Same IDs and metadata; flip one rule and recompute the correct later action."""
    if which not in {"restoration", "permission"}:
        raise ValueError("Choose restoration or permission")
    restore = not episode.restore if which == "restoration" else episode.restore
    allowed = 1 - episode.allowed_capability if which == "permission" else episode.allowed_capability
    sources = []
    for s in episode.supports:
        text = s.text
        if which == "restoration" and s.kind == which:
            text = (f"Experience in {episode.environment}: after a failed attempt, "
                    + ("restoring the snapshot is required before retry." if restore
                       else "restoring the snapshot is forbidden before retry."))
        if which == "permission" and s.kind == which:
            text = (f"Experience in {episode.environment}: retry is permitted exactly when "
                    f"capability={allowed}; otherwise stop.")
        sources.append(replace(s, text=text))
    answer = "STOP" if episode.capability != allowed else ("RESTORE_RETRY" if restore else "RETRY")
    return replace(episode, supports=tuple(sources), answer=answer, restore=restore, allowed_capability=allowed)


def save_episodes(path: str | Path, episodes: list[Episode]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(asdict(episode), sort_keys=True) + "\n")


def load_episodes(path: str | Path) -> list[Episode]:
    episodes = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row.setdefault("environment", row["episode_id"])
            row.setdefault("restore", False)
            row.setdefault("allowed_capability", 0)
            row.setdefault("capability", 0)
            row["supports"] = tuple(Source(**({"kind": "experience"} | source)) for source in row["supports"])
            row["required_ids"] = tuple(row["required_ids"])
            episode = Episode(**row)
            if any(s.created_at >= episode.query_time for s in episode.supports):
                raise ValueError("Source is not causally prior to the query")
            if episode.episode_id in {s.record_id for s in episode.supports}:
                raise ValueError("Query cannot be its own source")
            ids = [s.record_id for s in episode.supports]
            if len(ids) != len(set(ids)) or not set(episode.required_ids) <= set(ids):
                raise ValueError("Support IDs must be unique and include required_ids")
            if not episode.answer or not episode.query:
                raise ValueError("Query and answer cannot be empty")
            episodes.append(episode)
    seen = {}
    episode_ids = set()
    for episode in episodes:
        if episode.episode_id in episode_ids:
            raise ValueError("Episode IDs must be unique")
        episode_ids.add(episode.episode_id)
        for source in episode.supports:
            if source.record_id in seen and seen[source.record_id] != source:
                raise ValueError("A shared source ID must identify exactly the same experience")
            seen[source.record_id] = source
    if not episodes:
        raise ValueError("Episode file is empty")
    return episodes
