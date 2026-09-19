"""Queued evaluations tolerate registration races but do not hide failed runs."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def waiter(monkeypatch):
    path = Path(__file__).parents[1] / 'scripts/evaluate_causal_stages.py'
    spec = importlib.util.spec_from_file_location('evaluation_wait', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    now = [0.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    return module, now


def state(status, *, stop=False):
    return {'status': status, 'running': status == 'running', 'stop_requested': stop}


def test_wait_allows_delayed_process_registration(waiter, monkeypatch):
    module, now = waiter
    states = iter([state('not_managed'), state('running'), state('complete')])
    monkeypatch.setattr(module, 'run_status', lambda run: next(states))
    module.wait_for_completion(Path('/unused'))
    assert now[0] == 20


def test_wait_unregistered_job_times_out(waiter, monkeypatch):
    module, now = waiter
    monkeypatch.setattr(module, 'run_status', lambda run: state('not_managed'))
    with pytest.raises(RuntimeError, match='register'):
        module.wait_for_completion(Path('/unused'), startup_timeout=25)
    assert now[0] == 25


def test_wait_stop_before_registration_is_immediate(waiter, monkeypatch):
    module, now = waiter
    monkeypatch.setattr(module, 'run_status', lambda run: state('not_managed', stop=True))
    with pytest.raises(RuntimeError, match='did not complete'):
        module.wait_for_completion(Path('/unused'), startup_timeout=30)
    assert now[0] == 0


def test_wait_does_not_hide_lost_registered_state(waiter, monkeypatch):
    module, now = waiter
    states = iter([state('running'), state('not_managed')])
    monkeypatch.setattr(module, 'run_status', lambda run: next(states))
    with pytest.raises(RuntimeError, match='not_managed'):
        module.wait_for_completion(Path('/unused'), startup_timeout=30)
    assert now[0] == 10
