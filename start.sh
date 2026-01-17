#!/usr/bin/env bash
set -euo pipefail

# One-click startup for LiteLLM Proxy (bash).
# - Starts in background
# - Redirects logs to ./logs/litellm.log
# - Binds to 0.0.0.0 by default

cd "$(dirname "$0")"

CONFIG="${LITELLM_CONFIG:-./proxy_iflow_config.yaml}"
HOST="${LITELLM_HOST:-0.0.0.0}"
PORT="${LITELLM_PORT:-4000}"
LOG_DIR="${LITELLM_LOG_DIR:-./logs}"
LOG_FILE="${LITELLM_LOG_FILE:-$LOG_DIR/litellm.log}"
PID_FILE="${LITELLM_PID_FILE:-$LOG_DIR/litellm.pid}"

if command -v litellm >/dev/null 2>&1; then
  mkdir -p "$LOG_DIR"

  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "LiteLLM already running (pid=$pid). Log: $LOG_FILE"
      exit 0
    fi
    rm -f "$PID_FILE"
  fi

  : > "$LOG_FILE"
  echo "Starting LiteLLM Proxy in background..."
  echo "  config: $CONFIG"
  echo "  bind:   $HOST:$PORT"
  echo "  log:    $LOG_FILE"

  nohup litellm --config "$CONFIG" --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"

  echo "Started (pid=$(cat "$PID_FILE"))."
  exit 0
fi

echo "ERROR: 'litellm' command not found. Install with: pip install 'litellm[proxy]'" >&2
exit 127
