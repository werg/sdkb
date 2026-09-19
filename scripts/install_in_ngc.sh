#!/usr/bin/env bash
# Isolate HF dependencies while preserving NVIDIA's native torch/CUDA installation.
set -euo pipefail
repo="${1:-/workspace/sdkb}"
python -m venv --system-site-packages /opt/sdkb-venv
python - <<'PY' >/opt/sdkb-runtime-constraints.txt
from importlib.metadata import distributions
for d in distributions():
    name = d.metadata['Name'].lower().replace('_', '-')
    if name in {'torch', 'torchvision', 'torchaudio', 'triton'} or name.startswith('nvidia-'):
        print(f'{name}=={d.version}')
PY
before="$(python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.__file__)')"
# Remove general vendor pins only inside this venv, keeping explicit GPU runtime pins.
env -u PIP_CONSTRAINT -u PIP_BUILD_CONSTRAINT /opt/sdkb-venv/bin/python -m pip install \
    -c /opt/sdkb-runtime-constraints.txt -e "$repo[hf,data,dev]"
after="$(/opt/sdkb-venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.__file__)')"
[[ "$before" == "$after" ]] || { echo 'Vendor torch changed; refusing build.' >&2; exit 1; }
/opt/sdkb-venv/bin/python - <<'PY'
from transformers import AutoTokenizer, Lfm2ForCausalLM
import datasets, sdkb
print('SDKB imports validated:', sdkb.__version__, datasets.__version__)
PY
/opt/sdkb-venv/bin/python -m pip freeze >/opt/sdkb-python-freeze.txt
