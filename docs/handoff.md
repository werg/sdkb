# Repository handoff

The v0.2 work is committed locally and supplied as a self-contained Git bundle,
source ZIP/tarball, and a patch from the v0.1 handoff. The connected GitHub tools
now include writes to existing repositories, but not repository creation. A fresh
lookup of `werg/external-latent-memory` returned 404 (absent or inaccessible). No
remote was created or pushed. No authenticated shell GitHub CLI was available.

The bundle `external-latent-memory-v0.2.bundle` preserves all commits and `main`.
Source archives exclude Git metadata, model checkpoints, caches and dependencies.

To update an unchanged v0.1 checkout without replacing local work:

```bash
git fetch /path/to/external-latent-memory-v0.2.bundle main
git merge --ff-only FETCH_HEAD
# Only after verifying your origin is the intended GitHub repository:
git remote -v
git push origin main
```

A divergent checkout intentionally fails the fast-forward merge rather than
silently overwriting work. The provided patch is an alternative for review.

To preserve the committed history:

```bash
git clone external-latent-memory-v0.2.bundle external-latent-memory
cd external-latent-memory
# Cloning a bundle creates an origin pointing at the local bundle. Remove only
# that local origin before asking the helper to create the actual GitHub remote.
git remote -v
git remote remove origin
gh auth login  # omit when already authenticated
./scripts/publish_github.sh werg/external-latent-memory
```

Check the printed remote URL and `isPrivate: true`. The script uses the standard
[GitHub CLI creation workflow](https://cli.github.com/manual/gh_repo_create),
creates a new private repo, pushes existing commits, fetches the remote main branch,
and checks commit equality. It refuses an existing repo or remote and never force
pushes. Authentication happens locally through GitHub CLI; do not put tokens in
this repository or paste them into a research configuration.

For source-only initialization instead:

```bash
mkdir external-latent-memory
cd external-latent-memory
tar -xzf ../external-latent-memory-v0.2.tar.gz
git init -b main
git add .
git commit -m 'Initialize external latent memory research project'
./scripts/publish_github.sh werg/external-latent-memory
```

Git author identity must already be configured for the latter option. The supplied
bundle avoids reconstructing the initial commits. Neither archive includes weights
or experiment databases. Recorded diagnostics are in `experiments/bootstrap` and `experiments/development-v2`.
