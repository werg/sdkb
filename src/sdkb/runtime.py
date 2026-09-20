"""Portable opt-in resource rails and stack diagnostics for stalled compute."""
from contextlib import contextmanager
from pathlib import Path
import faulthandler

import torch


def available_host_memory():
    path = Path('/proc/meminfo')
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return None


def configure_memory(config):
    available = available_host_memory()
    if available is not None and available < config.min_system_available_bytes:
        raise MemoryError('Host MemAvailable is below the configured reserve; '
                          'on unified-memory systems this includes CUDA and filesystem-cache pressure')
    if config.cuda_memory_fraction is not None and config.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(config.cuda_memory_fraction)
    return dict(host_available_bytes=available, cuda_memory_fraction=config.cuda_memory_fraction,
                min_system_available_bytes=config.min_system_available_bytes)


def memory_metrics(device):
    """Allocator counters without device synchronization; never sum host and CUDA RAM."""
    result = {}
    available = available_host_memory()
    if available is not None:
        result['host_available_bytes'] = available
    if torch.device(device).type == 'cuda':
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                      cuda_attempt_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    return result


@contextmanager
def compute_watchdog(seconds, *, device=None):
    """Dump stacks on a stall; never hard-kill a process with unsaved work.

    Armed around the declared compute region, excluding setup and saves.
    With a CUDA device, completion is awaited before disarming an enabled guard.
    Cooperative stopping still needs the current native kernel to return.
    """
    if seconds:
        faulthandler.dump_traceback_later(seconds, repeat=True)
    try:
        yield
        if seconds and device is not None and torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
    finally:
        if seconds:
            faulthandler.cancel_dump_traceback_later()
