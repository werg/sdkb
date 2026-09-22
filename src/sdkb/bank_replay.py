"""Selective writer replay and post-update refresh for mutable training banks."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from .agent import SDKBAgent
from .replay import ReplayTape
from .store import StoredRecord
from .training_bank import TrainingBank


class BankWriterReplay:
    def __init__(self, agent: SDKBAgent, writer_inputs: Mapping[str, Tensor], *,
                 verify_outputs: bool = True) -> None:
        self.agent = agent
        self.writer_inputs = writer_inputs
        self.tape = ReplayTape(verify_outputs=verify_outputs)
        self.record_ids: list[str] = []

    def _produce(self, record_ids: tuple[str, ...]) -> tuple[Tensor, ...]:
        outputs = self.agent.produce_batch([self.writer_inputs[record_id]
                                            for record_id in record_ids])
        dtype = getattr(torch, self.agent.config.memory.storage_dtype)
        serialized = []
        for space in range(len(self.agent.config.memory.payload_dims)):
            serialized.extend((outputs[2 * space].float(),
                               outputs[2 * space + 1].to(dtype).float()))
        return tuple(serialized)

    def capture(self, record_ids: tuple[str, ...]) -> dict[str, tuple[Tensor, ...]]:
        if not record_ids or len(record_ids) != len(set(record_ids)):
            raise ValueError('Writer replay needs distinct record IDs')
        missing = set(record_ids) - self.writer_inputs.keys()
        if missing:
            raise KeyError(f'Writer inputs missing for {sorted(missing)[:3]}')
        leaves = self.tape.capture(self.agent, lambda: self._produce(record_ids))
        self.record_ids.extend(record_ids)
        return {record_id: tuple(output[row:row + 1] for output in leaves)
                for row, record_id in enumerate(record_ids)}

    def backward(self) -> None:
        self.tape.backward()

    @torch.no_grad()
    def refresh(self, bank: TrainingBank, *, batch_size: int = 16) -> int:
        """Regenerate touched views after the optimizer step and commit atomically."""
        ids = tuple(dict.fromkeys(self.record_ids))
        records = []
        dtype = getattr(torch, self.agent.config.memory.storage_dtype)
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            outputs = self.agent.produce_batch([self.writer_inputs[record_id]
                                                for record_id in batch])
            for row, record_id in enumerate(batch):
                for space in range(len(self.agent.config.memory.payload_dims)):
                    records.append(StoredRecord(
                        record_id, outputs[2 * space][row].detach(),
                        outputs[2 * space + 1][row].to(dtype).detach(),
                        namespace=bank.index.namespace, space=f's{space}',
                        generation=bank.index.generation))
        return bank.update(records)
