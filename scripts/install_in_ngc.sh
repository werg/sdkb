#!/usr/bin/env bash
# Preserve NVIDIA's torch/CUDA stack. New Python dependencies are isolated.
set -euo pipefail
repo="${1:-/workspace/elm}"
python -m venv --system-site-packages /opt/elm-venv
python - <<'PY' > /tmp/elm-torch-constraints.txt
from importlib.metadata import version, PackageNotFoundError
for name in ('torch', 'torchvision', 'torchaudio', 'triton'):
    try:
        print(f'{name}=={version(name)}')
    except PackageNotFoundError:
        pass
PY
before="$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
# Keep an inherited PIP_CONSTRAINT as well; conflicts should fail, not silently
# replace the vendor runtime. Choose another documented base image if needed.
/opt/elm-venv/bin/python -m pip install -c /tmp/elm-torch-constraints.txt -e "$repo[hf,dev]"
after="$(/opt/elm-venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
[[ "$before" == "$after" ]] || { echo 'NVIDIA torch runtime changed; refusing build.' >&2; exit 1; }
