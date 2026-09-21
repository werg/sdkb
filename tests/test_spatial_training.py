import json

import torch

from sdkb.agent import SDKBAgent
from sdkb.data import make_episode
from sdkb.key_index import PublishedKeyIndex
from sdkb.spatial_data import pack_spatial_trajectory
from sdkb.spatial_training import spatial_bank_forward
from sdkb.store import DiskStore, StoredRecord


class StableChatTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        text, mask = "B", [0]
        for message in messages:
            rendered = f"<{message['role']}>" + str(message.get("content", ""))
            if message.get("tool_calls"):
                rendered += json.dumps(message["tool_calls"], sort_keys=True)
            rendered += "</>"
            text += rendered
            mask.extend([int(message["role"] == "assistant")] * len(rendered))
        return {"input_ids": [ord(char) for char in text], "assistant_masks": mask}


def test_spatial_bank_forward_reads_all_sites_without_writer(tiny_config, tmp_path):
    tiny_config.model.loops = 2
    tiny_config.model.recurrence_mode = "middle_block"
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.memory.read_timing = "loop_boundary"
    tiny_config.memory.neighbors = [3]
    tiny_config.train.routing_weight = .1
    agent = SDKBAgent(tiny_config)
    episodes = [make_episode(index, distractors=0) for index in range(2)]
    row = pack_spatial_trajectory(StableChatTokenizer(), episodes, read_slots=2,
                                  generation="g1", levels=(1, 1))
    store = DiskStore(tmp_path / "bank.sqlite")
    sources = {source.record_id: source for episode in episodes for source in episode.supports}
    store.put_many(StoredRecord(
        record_id, torch.randn(tiny_config.memory.key_dim),
        torch.randn(tiny_config.memory.payload_dims[0], dtype=torch.bfloat16),
        namespace="corpus", space="s0", generation="g1", domain="research",
        created_at=source.created_at, source_id=record_id,
    ) for record_id, source in sources.items())
    index = PublishedKeyIndex(store, namespace="corpus", generation="g1",
                              spaces=("s0",), expected_sources=len(sources))
    agent.produce = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("spatial stored reads must not call the writer"))

    result = spatial_bank_forward(agent, store, index, [row], limits=(3,),
                                  routing_candidates=3)
    assert result.loss.isfinite() and result.metrics["read_sites"] == 2
    assert result.metrics["selected_counts"] == [3]
    result.loss.backward()
    assert agent.query_maps[0].weight.grad is not None
    assert agent.query_maps[0].weight.grad.abs().sum() > 0
    assert next(agent.reader.parameters()).grad is not None


def test_spatial_forward_projects_only_supervised_positions(tiny_config, tmp_path, monkeypatch):
    tiny_config.model.loops = 2
    tiny_config.model.recurrence_mode = "middle_block"
    tiny_config.model.recurrent_start = 0
    tiny_config.model.recurrent_end = 1
    tiny_config.memory.read_timing = "loop_boundary"
    agent = SDKBAgent(tiny_config)
    episode = make_episode(0, distractors=0)
    row = pack_spatial_trajectory(StableChatTokenizer(), [episode], read_slots=2,
                                  generation="g1")
    store = DiskStore(tmp_path / "bank.sqlite")
    store.put_many(StoredRecord(
        source.record_id, torch.randn(tiny_config.memory.key_dim),
        torch.randn(tiny_config.memory.payload_dims[0], dtype=torch.bfloat16),
        namespace="corpus", space="s0", generation="g1", created_at=source.created_at,
    ) for source in episode.supports)
    index = PublishedKeyIndex(store, namespace="corpus", generation="g1", spaces=("s0",),
                              expected_sources=len(episode.supports))
    original, projected = agent.backbone.logits, []

    def observe(hidden):
        projected.append(hidden.shape)
        return original(hidden)

    monkeypatch.setattr(agent.backbone, "logits", observe)
    result = spatial_bank_forward(agent, store, index, [row], limits=(2,),
                                  routing_candidates=2)
    assert projected == [(row["supervised_tokens"], agent.width)]
    assert result.metrics["supervised_tokens"] == row["supervised_tokens"]
