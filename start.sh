#!/usr/bin/env bash
set -euo pipefail

# One-click startup for LiteLLM Proxy (bash).
# Uses the repo-local config in this directory.

cd "$(dirname "$0")"

CONFIG="${LITELLM_CONFIG:-./proxy_iflow_config.yaml}"
HOST="${LITELLM_HOST:-127.0.0.1}"
PORT="${LITELLM_PORT:-4000}"

if command -v litellm >/dev/null 2>&1; then
  exec litellm --config "$CONFIG" --host "$HOST" --port "$PORT"
fi

echo "ERROR: 'litellm' command not found. Install with: pip install 'litellm[proxy]'" >&2
exit 127

