import pytest
import torch

from sdkb.key_geometry import (RetrievalLoad, effective_rank, geometry_summary, koleo_loss,
                               mean_direction_cosine, occurrence_statistics, uniformity,
                               variance_covariance_loss)


def _spread(n=512, d=32, seed=0):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def _collapsed(n=512, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    direction = torch.randn(d, generator=g)
    return direction + 0.01 * torch.randn(n, d, generator=g)


def test_collapse_metrics_separate_spread_from_collinear_keys():
    spread, collapsed = _spread(), _collapsed()
    assert mean_direction_cosine(collapsed) > 0.99 > 0.3 > mean_direction_cosine(spread)
    assert effective_rank(spread) > 20
    assert uniformity(spread) < uniformity(collapsed)
    low_rank = _spread()[:, :2] @ torch.randn(2, 32)
    assert effective_rank(low_rank) < 2.5


def test_regularizers_are_lower_for_spread_keys_and_differentiable():
    spread, collapsed = _spread(), _collapsed()
    variance_s, covariance_s = variance_covariance_loss(spread)
    variance_c, _ = variance_covariance_loss(collapsed)
    assert variance_s < variance_c
    assert covariance_s < 0.1
    assert koleo_loss(spread) < koleo_loss(collapsed)
    keys = collapsed.clone().requires_grad_(True)
    variance, covariance = variance_covariance_loss(keys)
    (variance + covariance + koleo_loss(keys)).backward()
    assert torch.isfinite(keys.grad).all() and keys.grad.abs().sum() > 0


def test_correlated_dimensions_raise_covariance():
    x = _spread()
    x[:, 1] = x[:, 0] + 0.01 * x[:, 1]
    assert variance_covariance_loss(x)[1] > variance_covariance_loss(_spread())[1]


def test_occurrence_statistics_detect_hubs():
    balanced = torch.arange(100).reshape(25, 4)
    hubbed = torch.cat([torch.zeros(25, 1, dtype=torch.long),
                        torch.arange(75).reshape(25, 3) % 10 + 1], 1)
    even, hub = occurrence_statistics(balanced, 100), occurrence_statistics(hubbed, 100)
    assert even['gini'] == pytest.approx(0.0, abs=1e-6)
    assert even['never_retrieved'] == 0
    assert hub['gini'] > 0.8 and hub['never_retrieved'] == pytest.approx(0.89)
    assert hub['max_share'] == pytest.approx(0.25) and hub['skewness'] > 3


def test_geometry_summary_reports_every_metric():
    summary = geometry_summary(_spread(256), _spread(64, seed=1), k=8)
    assert {'key_effective_rank', 'query_effective_rank', 'key_uniformity',
            'hubness_at_8'} <= summary.keys()


def test_retrieval_load_penalizes_only_hubs_and_explores_cold_records():
    load = RetrievalLoad(10, decay=0.9, threshold=3.0)
    for _ in range(20):
        load.update(torch.tensor([0, 0, 0, 1]))
    overload = load.overload(torch.arange(10))
    assert overload[0] > 0 and (overload[1:] == 0).all()
    queries = torch.randn(4, 8)
    keys = torch.randn(10, 8, requires_grad=True)
    load.penalty(queries, keys, torch.arange(10)).backward()
    assert keys.grad[0].abs().sum() > 0 and keys.grad[1:].abs().sum() == 0
    g = torch.Generator().manual_seed(0)
    picks = torch.cat([load.explore(torch.arange(10), 3, generator=g) for _ in range(200)])
    counts = torch.bincount(picks, minlength=10)
    assert counts[0] < counts[5] and counts[1] < counts[5]
    restored = RetrievalLoad(10, decay=0.9, threshold=3.0)
    restored.load_state_dict(load.state_dict())
    assert torch.equal(restored.counts, load.counts)
    with pytest.raises(ValueError):
        RetrievalLoad(11, decay=0.9, threshold=3.0).load_state_dict(load.state_dict())
