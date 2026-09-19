#!/usr/bin/env bash
# Explicitly creates a NEW PRIVATE repo. Never force pushes, never changes auth.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
repository="${1:-werg/external-latent-memory}"
[[ "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || {
  echo 'Use owner/repository.' >&2; exit 2;
}
command -v gh >/dev/null || { echo 'Install GitHub CLI and run gh auth login first.' >&2; exit 1; }
gh auth status
git rev-parse --verify HEAD >/dev/null
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit or remove uncommitted changes first.' >&2; exit 1; }
if git remote get-url origin >/dev/null 2>&1; then
  echo 'An origin already exists. Review it and publish manually; no overwrite performed.' >&2
  exit 1
fi
if gh repo view "$repository" --json name >/dev/null 2>&1; then
  echo "Repository already exists: $repository. No changes made." >&2
  exit 1
fi
gh repo create "$repository" --private --source=. --remote=origin --push \
  --description 'Transfer-trained external latent memory, recurrent composition, selective replay and conditional compaction'
git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
gh repo view "$repository" --json url,isPrivate,defaultBranchRef
