#!/usr/bin/env bash
# Clone or update vendored native libraries to the pinned commits in vendor/native/*.lock
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT/vendor/native"
LOCK_FILE="$VENDOR_DIR/parakeet.lock"
SRC_DIR="$VENDOR_DIR/parakeet.cpp"

if [[ ! -f "$LOCK_FILE" ]]; then
  echo "Lock file not found: $LOCK_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$LOCK_FILE"

: "${PARAKEET_CPP_REPO:?missing in parakeet.lock}"
: "${PARAKEET_CPP_COMMIT:?missing in parakeet.lock}"

echo "==> parakeet.cpp @ ${PARAKEET_CPP_COMMIT:0:12} (${PARAKEET_CPP_COMMIT_DATE:-pinned})"

if [[ ! -d "$SRC_DIR/.git" ]]; then
  echo "    Cloning $PARAKEET_CPP_REPO"
  git clone --recursive "$PARAKEET_CPP_REPO" "$SRC_DIR"
fi

git -C "$SRC_DIR" fetch origin --tags
git -C "$SRC_DIR" checkout "$PARAKEET_CPP_COMMIT"
git -C "$SRC_DIR" submodule update --init --recursive

CURRENT="$(git -C "$SRC_DIR" rev-parse HEAD)"
if [[ "$CURRENT" != "$PARAKEET_CPP_COMMIT" ]]; then
  echo "Checkout failed: expected $PARAKEET_CPP_COMMIT, got $CURRENT" >&2
  exit 1
fi

echo "    Source ready: $SRC_DIR"
