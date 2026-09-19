"""The Docker wrapper must not silently fall back from configured storage."""
from pathlib import Path
import shutil
import subprocess
import os

import pytest

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='Linux/macOS Bash wrapper')


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    (root / 'scripts').mkdir(parents=True)
    (root / '.sdkb').mkdir()
    script = root / 'scripts/spark.sh'
    shutil.copyfile(Path(__file__).parents[1] / 'scripts/spark.sh', script)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    docker = bin_dir / 'docker'
    docker.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    docker.chmod(0o755)
    capture = tmp_path / 'arguments'
    monkeypatch.setenv('CAPTURE', str(capture))
    monkeypatch.setenv('PATH', str(bin_dir) + ':/usr/bin:/bin')
    monkeypatch.setenv('SDKB_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.delenv('SDKB_RUNS_DIR', raising=False)
    monkeypatch.delenv('SDKB_ARCHIVE_DIR', raising=False)
    return root, script, capture


def test_machine_storage_default_and_environment_override(wrapper, tmp_path, monkeypatch):
    root, script, capture = wrapper
    external = tmp_path / 'external runs'
    external.mkdir()
    (root / '.sdkb/runs-dir').write_text(str(external) + '\n')
    args = ['bash', str(script), 'start', '--recipe', 'recipe.yaml', '--output', '/runs/test']
    subprocess.run(args, check=True, capture_output=True)
    assert f'type=bind,src={external},dst=/runs' in capture.read_text().splitlines()
    assert not (root / 'runs').exists()
    override = tmp_path / 'override'
    override.mkdir()
    monkeypatch.setenv('SDKB_RUNS_DIR', str(override))
    subprocess.run(args, check=True, capture_output=True)
    assert f'type=bind,src={override},dst=/runs' in capture.read_text().splitlines()


def test_missing_configured_storage_fails_without_creating_it(wrapper, tmp_path):
    root, script, capture = wrapper
    absent = tmp_path / 'unmounted' / 'runs'
    (root / '.sdkb/runs-dir').write_text(str(absent) + '\n')
    result = subprocess.run(['bash', str(script), 'run', 'sdkb', 'doctor'], capture_output=True)
    assert result.returncode != 0
    assert not absent.exists()
    assert not capture.exists()


@pytest.mark.parametrize('command', [['start'], ['run', 'sdkb', 'launch'],
                                     ['run', 'sdkb', 'train'], ['run', 'sdkb', 'runs', 'start']])
@pytest.mark.parametrize('output', [['--output', 'runs/accidental-local'],
                                   ['--output=runs/accidental-local'], ['--output', '']])
def test_relative_run_output_cannot_bypass_the_storage_mount(wrapper, tmp_path, command, output):
    root, script, capture = wrapper
    external = tmp_path / 'external'
    external.mkdir()
    (root / '.sdkb/runs-dir').write_text(str(external) + '\n')
    result = subprocess.run(['bash', str(script), *command, *output], capture_output=True)
    assert result.returncode != 0
    assert b'/runs/' in result.stderr
    assert not capture.exists()
