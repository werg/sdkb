"""Local run ownership and cooperative stops, independent of GPU/container runtime."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with pending.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, path)


def control_dir(run):
    run = Path(run).resolve()
    # Same control directory through host/container bind mounts of the run root.
    key = hashlib.sha256(run.name.encode()).hexdigest()[:24]
    path = run.parent / '.sdkb-control' / key
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def file_lock(path):
    """OS lock released on process death; an old file is never a stale lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        try:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                handle.write(b'\0')
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Run is already running: {path}') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def run_lock(run, *, clear_stop=True):
    directory = control_dir(run)
    with file_lock(directory / 'owner.lock'):
        if clear_stop:
            (directory / 'STOP').unlink(missing_ok=True)
        yield


def stop_requested(run):
    return (control_dir(run) / 'STOP').exists()


def request_stop(run):
    path = control_dir(run) / 'STOP'
    atomic_json(path, {'requested_at': time.time()})
    return {'status': 'stop_requested', 'output': str(Path(run).resolve()),
            'behavior': 'Finish the current optimizer step and checkpoint; evaluations stop at the next stage boundary.'}


def run_status(run):
    directory = control_dir(run)
    path = directory / 'process.json'
    state = json.loads(path.read_text()) if path.exists() else {'status': 'not_managed'}
    try:
        with file_lock(directory / 'owner.lock'):
            running = False
    except RuntimeError:
        running = True
    if not running and state['status'] == 'running':
        state['status'] = 'starting' if time.time() - state.get('started_at', 0) < 60 else 'interrupted'
    if not running and state['status'] == 'starting':
        pid_file = directory / 'pid.json'
        if pid_file.exists() and os.name != 'nt':
            try:
                os.kill(json.loads(pid_file.read_text())['pid'], 0)
            except ProcessLookupError:
                state['status'] = 'interrupted'
            except PermissionError:
                pass
        if time.time() - state.get('started_at', 0) > 60:
            state['status'] = 'interrupted'
    return state | {'running': running, 'stop_requested': stop_requested(run),
                    'log': str(directory / 'console.log')}


def start_run(recipe, run, *, resume=False):
    """Start one detached local worker; use foreground launch inside a container."""
    run = Path(run).resolve()
    recipe = Path(recipe).resolve()
    if not recipe.is_file():
        raise FileNotFoundError(recipe)
    directory = control_dir(run)
    with file_lock(directory / 'start.lock'):
        state = run_status(run)
        if state['running'] or state['status'] == 'starting':
            raise RuntimeError('Run is already running or starting')
        if run.exists() and not resume:
            raise FileExistsError('Output exists; use --resume')
        atomic_json(directory / 'process.json', {'status': 'starting', 'recipe': str(recipe),
                                               'output': str(run), 'started_at': time.time()})
        # Only this explicit start acknowledges an earlier stop, before spawn.
        (directory / 'STOP').unlink(missing_ok=True)
        command = [sys.executable, '-m', 'sdkb.operations', str(recipe), str(run)]
        if resume:
            command.append('--resume')
        try:
            with (directory / 'console.log').open('ab', buffering=0) as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=os.name != 'nt',
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
                    env=os.environ | {'PYTHONUNBUFFERED': '1'})
        except BaseException:
            atomic_json(directory / 'process.json', {'status': 'failed_to_start'})
            raise
        atomic_json(directory / 'pid.json', {'pid': child.pid})
    return {'status': 'starting', 'pid': child.pid, 'log': str(directory / 'console.log')}


def worker(recipe, run, resume=False):
    from .launch import launch
    directory = control_dir(run)
    state = {'status': 'running', 'pid': os.getpid(), 'recipe': str(recipe), 'output': str(run),
             'started_at': time.time()}
    atomic_json(directory / 'process.json', state)
    try:
        result = launch(recipe, run, resume=resume, preserve_stop=True)
    except BaseException:
        atomic_json(directory / 'process.json', state | {'status': 'failed', 'finished_at': time.time()})
        raise
    atomic_json(directory / 'process.json', state | {'status': result['status'], 'finished_at': time.time()})


if __name__ == '__main__':
    worker(sys.argv[1], sys.argv[2], '--resume' in sys.argv[3:])
