# DGX Spark development

## Hardware accounting

NVIDIA documents DGX Spark as a GB10 system with a 20-core ARM CPU, Blackwell GPU,
128 GB coherent unified memory, and NVMe storage. See the
[hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
There is no independent host-RAM reservoir to add to the 128 GB GPU-visible memory.
CPU-side keys, SQLite caches, page cache, reader buffers, optimizer state and CUDA
allocations compete for that physical pool.

`elm doctor` records host architecture, Torch/CUDA versions, GPU name, capability,
BF16 support, a finite BF16 matrix multiplication, process peak RSS, CUDA allocator
peaks, and available system memory. These are complementary counters, not disjoint
terms that can be blindly summed. RSS/page-cache residency needs additional system
profiling before a unified-memory capacity claim.

## Native container

The Dockerfile defaults to NVIDIA PyTorch `26.08-py3`, matching the dated
[release documentation](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-08.html).
The image and host driver must be compatible; this repository does not modify the
host driver. Start with `nvidia-smi`, a functioning Docker daemon and NVIDIA
Container Toolkit installed through the vendor-supported route. The repository
has not built or run this image on real Spark hardware.

```bash
./scripts/spark.sh build
./scripts/spark.sh run elm doctor --require-spark
./scripts/spark.sh shell
```

The build creates a virtual environment exposing the NVIDIA system packages and
pins the installed Torch family while installing the project. It compares the
Torch and CUDA versions before/after installation and refuses a changed runtime.
An inherited NVIDIA pip constraint is respected; a conflict fails rather than
silently replacing the vendor stack. Use a compatible base image and inspect the
conflicting package when a build fails.

```bash
ELM_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.08-py3 ./scripts/spark.sh build
ELM_CACHE_DIR=/path/on/nvme/elm-cache ./scripts/spark.sh run elm doctor
```

Use the same `ELM_CACHE_DIR` for later runs to avoid redownloading checkpoints.
The shell script runs the container as the invoking UID/GID so host run directories
remain writable. The optional VS Code devcontainer uses its default container user;
check ownership when mixing the two entry paths. The workspace is the only project
bind mount; authentication directories are not mounted.

The `--shm-size=8g` value is an upper bound, not a promise that 8 GB is unused or
free. Reduce it for competing workloads. No `--privileged` or host IPC is required.

## First model checks

```bash
./scripts/spark.sh run python -m pytest -q
./scripts/spark.sh run python scripts/pin_model.py \
  --config configs/lfm25_230m_spark.yaml --output runs/lfm-pinned.yaml
./scripts/spark.sh run bash scripts/smoke_lfm.sh \
  runs/lfm-pinned.yaml runs/lfm-smoke
```

The smoke performs device checks, a random-LFM structural test, two real-model
training steps, checkpoint save/reload, bank creation and stored-only evaluation.
The default trains FP32 master parameters under BF16 autocast, with small
microbatches, gradient accumulation, producer replay and reader/backbone
checkpointing. It does not use FP8, quantized optimizers, FlashAttention packages
or custom convolution extensions. Optimize these only after parity is established.

Pin image digest and HF model revision for measured runs. `main` is only a setup
convenience. Save `environment.json`, `config.json`, `data_manifest.json`, run commit,
metrics and device profiling. Torch, transformers, tokenizers and CUDA must all be
recorded when comparing devices or models.

## Storage harness

```bash
./scripts/spark.sh run elm io-bench --path runs/io.sqlite \
  --records 10000 --payload-dim 256 --key-dim 64 --neighbors 16 \
  --reads 100 --cache-mib 32 --output runs/io-report.json
```

This is an exact CPU cosine scan plus SQLite payload fetch. It measures search and
fetch separately and reports serialized bytes and physical database size. It does
**not** clear OS caches; a just-populated bank will normally be warm. The SQLite
cache setting is per connection and does not bound the OS page cache. It is not
DiskANN, an SSD random-I/O benchmark, or evidence that a 10,000-record bank exceeds
RAM. To substantiate a disk-resident result, use a representative working set,
controlled memory budget, concurrency and access pattern on the actual hardware.
Do not run privileged cache-dropping commands automatically on a shared machine.

## Troubleshooting and safe defaults

An unavailable Torch CUDA backend is a setup failure, not a reason to benchmark
CPU silently. An ARM64 check failure means the process is not running natively as
intended. Pure PyTorch LFM convolution can be slower than tuned kernels; measure
rather than assuming a kernel package supports GB10. Leave model caching disabled
in the recurrent path until a separately tested conv/KV cache design exists.

The manual GitHub Spark workflow requires the owner to register a persistent
runner labeled `spark`. Do not expose that runner to untrusted pull-request jobs.
No remote Spark access, runner registration, or host installation was performed
by the bootstrap.
