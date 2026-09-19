from types import SimpleNamespace

import pytest

from sdkb import runtime


def test_unified_memory_uses_available_host_memory(monkeypatch):
    monkeypatch.setattr(runtime, 'available_host_memory', lambda: 100)
    config = SimpleNamespace(device='cpu', cuda_memory_fraction=None, min_system_available_bytes=101)
    with pytest.raises(MemoryError, match='MemAvailable'):
        runtime.configure_memory(config)
    config.min_system_available_bytes = 50
    assert runtime.configure_memory(config)['host_available_bytes'] == 100


def test_watchdog_is_disarmed_on_exception(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.faulthandler, 'dump_traceback_later', lambda *a, **kw: calls.append('armed'))
    monkeypatch.setattr(runtime.faulthandler, 'cancel_dump_traceback_later', lambda: calls.append('cancelled'))
    with pytest.raises(ValueError):
        with runtime.compute_watchdog(60):
            raise ValueError('compute failed')
    assert calls == ['armed', 'cancelled']
