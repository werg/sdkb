from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    backend: str = "hf"
    model_id: str = "LiquidAI/LFM2.5-230M"
    revision: str = "main"  # resolved commit is logged; pin before scientific runs
    loops: int = 1
    tiny_width: int = 64
    tiny_layers: int = 2
    tiny_heads: int = 4
    gradient_checkpointing: bool = True
    freeze_backbone: bool = False


@dataclass
class MemoryConfig:
    key_dim: int = 64
    write_slots: int = 8
    read_slots: int = 8
    payload_dims: list[int] = field(default_factory=lambda: [256])
    neighbors: list[int] = field(default_factory=lambda: [16])
    reader_width: int = 128
    reader_rounds: int = 3
    reader: str = "mlp"
    chunk_size: int = 32
    checkpoint_chunks: bool = True
    storage_dtype: str = "bfloat16"
    compaction: str = "none"
    compact_records: int = 2
    compaction_probability: float = 0.25
    compaction_loss_weight: float = 0.1
    compaction_warmup: int = 100
    compaction_objective: str = "interleaved"  # paired adds raw-task anchoring
    compaction_grouping: str = "whole"
    compaction_group_size: int = 4
    field_memberships: int = 2
    compact_task_weight: float = 1.0
    behavior_kl_weight: float = 0.0
    merge_loss_weight: float = 0.0
    read_steps: int = 1  # scheduled causal reads, not a replay memory limit
    stream_reads: bool = False  # single-space stored-only bounded staging
    read_top_k: int = 1
    noise_std: float = 0.0
    quantization_step: float = 0.0


@dataclass
class TrainConfig:
    seed: int = 17
    steps: int = 200
    learning_rate: float = 0.0001
    backbone_learning_rate: float = 0.00001
    gradient_accumulation: int = 4
    clip_grad_norm: float = 1.0
    precision: str = "bf16"
    device: str = "cuda"
    replay: bool = True
    verify_replay: bool = False
    live_fraction: float = 0.5
    train_worlds: int = 128
    distractors: int = 2
    eval_worlds: int = 16
    log_every: int = 10
    max_source_tokens: int = 512
    max_prompt_tokens: int = 1024
    arm: str = "memory"  # memory/no_memory/oracle_text/direct_latent
    retrieval: str = "oracle"  # oracle or learned (candidate-local top-k)
    routing_weight: float = 0.1
    routing_warmup: int = 100
    threads: int = 4
    optimization_scope: str = "all"  # compactor freezes writer, reader and controller
    checkpoint_every: int = 50
    keep_checkpoints: int = 2
    max_target_tokens: int = 512
    oracle_anchor_weight: float = 0.0
    episodes_file: str | None = None  # optional general support/query JSONL


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        m, r, t = self.model, self.memory, self.train
        if m.backend not in {"tiny", "hf"} or m.loops < 1:
            raise ValueError("Invalid backbone configuration")
        if len(r.payload_dims) != len(r.neighbors) or not r.payload_dims:
            raise ValueError("payload_dims and neighbors must have the same nonzero length")
        if min(*r.payload_dims, *r.neighbors, r.key_dim, r.write_slots, r.read_slots,
               r.reader_width, r.reader_rounds, r.chunk_size) < 1:
            raise ValueError("All memory dimensions and counts must be positive")
        if r.reader not in {"mlp", "attention"} or r.compaction not in {"none", "mean", "synthetic"}:
            raise ValueError("Invalid reader/compactor")
        if len(r.payload_dims) > 1 and r.compaction != "none":
            raise ValueError("Integrated compaction runner is single-space; multi-space APIs are separate")
        if r.storage_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Invalid storage precision")
        if t.precision not in {"fp32", "bf16"} or t.device not in {"cpu", "cuda"}:
            raise ValueError("Use fp32/bf16 and cpu/cuda")
        if not 0 <= t.live_fraction <= 1 or not 0 <= r.compaction_probability <= 1:
            raise ValueError("Probabilities must be between zero and one")
        if min(t.steps, t.train_worlds, t.eval_worlds, t.gradient_accumulation, t.log_every,
               t.max_source_tokens, t.max_prompt_tokens, t.max_target_tokens, t.threads, t.checkpoint_every, t.keep_checkpoints) < 1:
            raise ValueError("Training counts must be positive")
        if t.arm not in {"memory", "no_memory", "oracle_text", "direct_latent", "shared_compute"}:
            raise ValueError("Unknown comparison arm")
        if t.retrieval not in {"oracle", "learned"}:
            raise ValueError("Unknown retrieval mode")
        if r.noise_std < 0 or r.quantization_step < 0 or r.compact_records < 1:
            raise ValueError("Invalid noise or compaction settings")
        if r.compaction_objective not in {"interleaved", "paired"}:
            raise ValueError("Invalid compaction objective")
        if r.compaction_grouping not in {"whole", "random", "local", "overlap"}:
            raise ValueError("Invalid compaction grouping")
        if min(r.compaction_group_size, r.field_memberships, r.read_steps, r.read_top_k) < 1:
            raise ValueError("Positive grouping/read counts required")
        if min(r.compact_task_weight, r.behavior_kl_weight, r.merge_loss_weight) < 0:
            raise ValueError("Loss weights cannot be negative")
        if len(r.payload_dims) > 1 and r.merge_loss_weight:
            raise ValueError("Merge objective currently supports one memory space")
        if r.stream_reads and (len(r.payload_dims) != 1 or t.arm != "memory"):
            raise ValueError("Streaming currently supports one-space latent-memory inference")
        if r.read_steps > 1 and (r.compaction != "none" or t.arm != "memory"):
            raise ValueError("Scheduled multi-read is initially a raw-memory comparison")
        if t.optimization_scope not in {"all", "compactor"}:
            raise ValueError("Invalid optimization scope")
        if t.optimization_scope == "compactor" and (r.compaction != "synthetic" or r.compaction_probability != 1.0 or r.compaction_warmup != 0):
            raise ValueError("Compactor-only runs require synthetic compaction on every step without warmup")
        if t.oracle_anchor_weight < 0:
            raise ValueError("Negative oracle anchor")
        if t.distractors < 0 or min(t.learning_rate, t.backbone_learning_rate) <= 0:
            raise ValueError("Invalid distractor count or learning rate")


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) - {"model", "memory", "train"}:
        raise ValueError("Unknown configuration sections")
    config = Config(ModelConfig(**raw.get("model", {})), MemoryConfig(**raw.get("memory", {})),
                    TrainConfig(**raw.get("train", {})))
    config.validate()
    return config
