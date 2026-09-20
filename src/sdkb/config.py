from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    backend: str = "hf"
    model_id: str = "LiquidAI/LFM2.5-230M"
    revision: str = "main"  # resolved commit is logged; pin before scientific runs
    loops: int = 1
    recurrence_mode: str = "full_stack"  # legacy reference or native middle_block
    recurrent_start: int = 4
    recurrent_end: int = 10
    recurrent_input_mix: float = 0.1
    recurrent_update_mix: float = 0.1
    writer_loops: int | None = None  # None preserves old checkpoints; new recipes use 1
    backbone_train_scope: str = "all"  # all or recurrent_core
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
    read_timing: str = "prefix"  # prefix or loop_boundary
    read_steps: int = 1  # scheduled causal reads, not a replay memory limit
    stream_reads: bool = False  # single-space stored-only bounded staging
    read_top_k: int = 1
    noise_std: float = 0.0
    quantization_step: float = 0.0
    independent_routing_query: bool = False


@dataclass
class TrainConfig:
    seed: int = 17
    steps: int = 200
    learning_rate: float = 0.0001
    backbone_learning_rate: float = 0.00001
    optimizer: str = 'adamw'  # historical default; new Muon recipes opt in explicitly
    weight_decay: float = .01
    adam_betas: list[float] = field(default_factory=lambda: [.9, .999])
    adam_eps: float = 1e-8
    muon_momentum: float = .95
    muon_ns_steps: int = 5
    gradient_accumulation: int = 4
    clip_grad_norm: float = 1.0
    precision: str = "bf16"
    device: str = "cuda"
    replay: bool = True
    verify_replay: bool = False
    live_fraction: float = 0.5
    selected_producers_only: bool = False  # explicit all-live oracle compute policy
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
    cuda_memory_fraction: float | None = None
    min_system_available_bytes: int = 0
    stall_timeout_seconds: float = 0
    optimization_scope: str = "all"  # all, compactor, or routing address projections only
    checkpoint_every: int = 1000
    keep_checkpoints: int = 2
    max_target_tokens: int = 512
    loop_counts: list[int] = field(default_factory=list)  # sampled once per optimizer step
    oracle_anchor_loops: int | None = None
    allow_recurrence_conversion: bool = False
    reinitialize_reader: bool = False  # explicit warm-start fork; never applied on resume
    parent_kl_weight: float = 0.0  # fixed one-pass parent, only while native base is frozen
    oracle_anchor_weight: float = 0.0
    oracle_alignment_weight: float = 0.0  # detached selected-text answer-state cosine loss
    episodes_file: str | None = None  # optional general support/query JSONL
    evidence_scope: str = 'required'  # oracle-selected supports or all causally available candidates
    archive_dir: str | None = None  # existing external storage root; never auto-mount
    archive_keep_checkpoints: int = 3
    min_free_disk_bytes: int = 1024 ** 3
    wandb_mode: str = 'disabled'  # opt-in: offline or online
    wandb_project: str = 'sdkb'
    wandb_entity: str | None = None
    wandb_group: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        m, r, t = self.model, self.memory, self.train
        if (t.cuda_memory_fraction is not None and not 0 < t.cuda_memory_fraction <= 1
                or t.min_system_available_bytes < 0 or t.stall_timeout_seconds < 0):
            raise ValueError('Invalid runtime resource limits')
        if t.optimizer not in {'adamw', 'muon'}:
            raise ValueError('Optimizer must be adamw or muon')
        if not math.isfinite(t.oracle_alignment_weight) or t.oracle_alignment_weight < 0:
            raise ValueError('Invalid oracle alignment weight')
        if t.oracle_alignment_weight and (t.arm != 'memory' or t.retrieval != 'oracle'
                or r.read_timing != 'loop_boundary' or r.compaction != 'none'
                or not math.isfinite(t.oracle_anchor_weight) or t.oracle_anchor_weight <= 0):
            raise ValueError('Oracle alignment requires anchored oracle memory with native reads and no compaction')
        if t.selected_producers_only and (t.arm != 'memory' or t.retrieval != 'oracle' or t.live_fraction != 1.):
            raise ValueError('Selected-only producers require fully live oracle memory training')
        if (t.weight_decay < 0 or t.adam_eps <= 0 or len(t.adam_betas) != 2
                or any(not 0 <= b < 1 for b in t.adam_betas)
                or not 0 <= t.muon_momentum < 1 or t.muon_ns_steps < 1):
            raise ValueError('Invalid optimizer hyperparameters')
        if m.backend not in {"tiny", "hf"} or m.loops < 1:
            raise ValueError("Invalid backbone configuration")
        if m.recurrence_mode not in {"full_stack", "middle_block"}:
            raise ValueError("Invalid recurrence mode")
        if m.writer_loops is not None and m.writer_loops < 1:
            raise ValueError("Writer depth must be positive")
        if m.backbone_train_scope not in {"all", "recurrent_core"}:
            raise ValueError("Invalid backbone train scope")
        if m.recurrence_mode == "middle_block":
            if not 0 <= m.recurrent_start < m.recurrent_end:
                raise ValueError("Invalid recurrent partition")
            if m.backend == "tiny" and m.recurrent_end > m.tiny_layers:
                raise ValueError("Recurrent partition exceeds tiny decoder depth")
            if not 0 < m.recurrent_input_mix < 1 or not 0 < m.recurrent_update_mix < 1:
                raise ValueError("Recurrent gates must start live, strictly between zero and one")
        elif m.backbone_train_scope == "recurrent_core":
            raise ValueError("Core-only training requires middle-block recurrence")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in t.loop_counts):
            raise ValueError("Sampled loop counts must be positive integers")
        if t.oracle_anchor_loops is not None and t.oracle_anchor_loops < 1:
            raise ValueError("Anchor depth must be positive")
        if r.read_timing not in {"prefix", "loop_boundary"}:
            raise ValueError("Invalid read timing")
        if r.read_timing == "loop_boundary":
            if m.recurrence_mode != "middle_block":
                raise ValueError("In-loop reads require the native recurrent core")
            if r.stream_reads or t.arm == "direct_latent":
                raise ValueError("In-loop training requires materialized set reads")
            counts = t.loop_counts or [m.loops]
            if t.arm in {"memory", "shared_compute"} and min(counts) < 2:
                raise ValueError("Memory training requires a read followed by another core pass")
            if r.read_steps > 1 and min(counts) <= r.read_steps:
                raise ValueError("A scheduled read must leave at least one core pass to use it")
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
        if t.evidence_scope not in {'required', 'available'}:
            raise ValueError('Invalid evidence scope')
        if t.evidence_scope == 'available' and t.retrieval != 'oracle':
            raise ValueError('Available evidence scope requires oracle candidate access; it is not learned routing')
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
        if t.optimization_scope not in {"all", "compactor", "routing"}:
            raise ValueError("Invalid optimization scope")
        if t.optimization_scope == 'routing' and (t.arm != 'memory' or t.retrieval != 'learned'
                                                  or t.routing_weight <= 0 or r.compaction != 'none'):
            raise ValueError('Routing-only runs require learned memory routing, positive routing weight and raw records')
        if t.optimization_scope == "compactor" and (r.compaction != "synthetic" or r.compaction_probability != 1.0 or r.compaction_warmup != 0):
            raise ValueError("Compactor-only runs require synthetic compaction on every step without warmup")
        if t.oracle_anchor_weight < 0 or t.parent_kl_weight < 0:
            raise ValueError("Negative retention loss weight")
        if t.parent_kl_weight and (not m.freeze_backbone or m.recurrence_mode != "middle_block"
                                    or t.arm not in {"oracle_text", "no_memory"}):
            raise ValueError("Parent KL needs a frozen native backbone and identical text inputs")
        if t.loop_counts and m.recurrence_mode == "middle_block" and m.writer_loops is None:
            raise ValueError("Specify writer_loops independently of sampled consumer depth")
        if t.distractors < 0 or min(t.learning_rate, t.backbone_learning_rate) <= 0:
            raise ValueError("Invalid distractor count or learning rate")
        if t.archive_keep_checkpoints < 1 or t.min_free_disk_bytes < 0:
            raise ValueError('Invalid archive retention/free disk reserve')
        if t.wandb_mode not in {'disabled', 'offline', 'online'}:
            raise ValueError('wandb_mode must be disabled, offline or online')


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) - {"model", "memory", "train"}:
        raise ValueError("Unknown configuration sections")
    config = Config(ModelConfig(**raw.get("model", {})), MemoryConfig(**raw.get("memory", {})),
                    TrainConfig(**raw.get("train", {})))
    config.validate()
    return config
