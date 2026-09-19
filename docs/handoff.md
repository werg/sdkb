# Repository handoff

The bootstrap environment authenticated a read-capable GitHub connector as `werg`.
The exposed actions did not include creation, file writes or pushing commits. No
GitHub CLI or shell credentials were available, and the execution environment
could not connect to Git/Hugging Face from its shell. The project was therefore
implemented, tested and committed **locally**. No remote repo was created or pushed.

Two equivalent handoff formats are supplied:

* `external-latent-memory.bundle` preserves commits and the `main` branch.
* `external-latent-memory.tar.gz` is a source-only `git archive`, excluding Git
  metadata, model checkpoints, training caches and dependencies.

To preserve the committed history:

```bash
git clone external-latent-memory.bundle external-latent-memory
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
tar -xzf ../external-latent-memory.tar.gz
git init -b main
git add .
git commit -m 'Initialize external latent memory research project'
./scripts/publish_github.sh werg/external-latent-memory
```

Git author identity must already be configured for the latter option. The supplied
bundle avoids reconstructing the initial commits. Neither archive includes weights
or experiment databases. Recorded diagnostic JSON remains in `experiments/bootstrap`.
