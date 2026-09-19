# SDKB 0.3 handoff and publication

The rewritten repository is delivered as `sdkb-v0.3.bundle` with complete Git history,
plus source ZIP/tarball and a patch from the previous version.

The existing `werg/sdkb` GitHub repository was readable and empty when checked.
An actual contents write was rejected with **HTTP 403: Resource not accessible by
integration**. Consequently the handoff does **not** claim a remote commit or push.
The connection must permit repository contents writes, or use authenticated local Git.

Start from the supplied bundle on the Spark:

```bash
git clone sdkb-v0.3.bundle sdkb
cd sdkb
./scripts/start_spark.sh --recipe recipes/spark_smoke.yaml --output runs/spark-smoke
./scripts/start_spark.sh --recipe recipes/starter.yaml --output runs/starter
```

A bundle clone has a local-file `origin`. To publish to the currently empty repository
with an authenticated Git installation, inspect and rename that remote first:

```bash
git remote -v
git remote rename origin handoff-bundle
git remote add origin https://github.com/werg/sdkb.git
git push -u origin main
```

A normal non-forced push fails rather than discarding any unrelated upstream work.
Do not use `--force`. `scripts/publish_github.sh` is an optional authenticated local
helper with an existing-repository and ancestry check. It never changes visibility,
replaces an unexpected origin, creates an unrelated repository or bypasses permissions.

Credentials belong in normal Git/HF credential tooling, not source, Docker build args,
configuration or issues. Data/download/checkpoint locations are ignored by Git.
