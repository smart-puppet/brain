#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

exec python -m puppet.main --config "${PUPPET_CONFIG:-config}" "$@"
