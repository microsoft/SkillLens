#!/bin/bash
# Start the two tool servers required by SEAL-0 inference.
#
# What this does:
#   1. Starts search_server (FastAPI on port $SEARCH_SERVER_PORT, default 8001)
#      in the background, logging to logs/seal0/search_server.log.
#   2. Starts browser_server (FastAPI on port $BROWSER_SERVER_PORT, default 8002)
#      in the background, logging to logs/seal0/browser_server.log.
#   3. Writes PIDs to logs/seal0/{search,browser}_server.pid so they can be
#      stopped later with `kill $(cat ...)`.
#
# Required env (export, or put in .env):
#   SEARCH_PROVIDER         "serper" (default) or "microsoft"
#   SERPER_KEY_ID           Serper API key (when SEARCH_PROVIDER=serper)
#   MICROSOFT_SEARCH_API_KEY  Microsoft AI key (when SEARCH_PROVIDER=microsoft)
#   BROWSE_PROVIDER         "scrapedo" (default) or "microsoft"
#   SCRAPEDO_API_KEY        ScrapeDo API key (when BROWSE_PROVIDER=scrapedo)
#   MICROSOFT_BROWSE_API_KEY  Microsoft AI key (when BROWSE_PROVIDER=microsoft)
#
# Optional env:
#   SKILLLENS_CONDA_ENV     conda env name (default: skilllens)
#   SEARCH_SERVER_PORT      default 8001
#   BROWSER_SERVER_PORT     default 8002
#
# Usage:
#   bash scripts/setup_seal0.sh                 # start
#   bash scripts/setup_seal0.sh --stop          # stop running servers
#   bash scripts/setup_seal0.sh --status        # check status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Load .env if present
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; . "$REPO_ROOT/.env"; set +a
fi

CONDA_ENV="${SKILLLENS_CONDA_ENV:-skilllens}"
LOG_DIR="$REPO_ROOT/logs/seal0"
mkdir -p "$LOG_DIR"

SEARCH_PORT="${SEARCH_SERVER_PORT:-8001}"
BROWSER_PORT="${BROWSER_SERVER_PORT:-8002}"

ACTION="start"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stop)   ACTION="stop";   shift ;;
        --status) ACTION="status"; shift ;;
        -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

stop_one() {
    local name="$1"
    local pid_file="$LOG_DIR/${name}.pid"
    if [[ -f "$pid_file" ]]; then
        local pid; pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[setup_seal0] Stopping $name (pid $pid)..."
            kill "$pid" || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
}

status_one() {
    local name="$1"
    local pid_file="$LOG_DIR/${name}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "  $name : RUNNING (pid $(cat "$pid_file"))"
    else
        echo "  $name : not running"
    fi
}

if [[ "$ACTION" == "stop" ]]; then
    stop_one search_server
    stop_one browser_server
    echo "[setup_seal0] Stopped."
    exit 0
fi

if [[ "$ACTION" == "status" ]]; then
    status_one search_server
    status_one browser_server
    exit 0
fi

# --------- start ---------
# Sanity check provider credentials
SEARCH_PROVIDER="${SEARCH_PROVIDER:-serper}"
BROWSE_PROVIDER="${BROWSE_PROVIDER:-scrapedo}"

if [[ "$SEARCH_PROVIDER" == "serper" && -z "${SERPER_KEY_ID:-}" ]]; then
    echo "ERROR: SEARCH_PROVIDER=serper but SERPER_KEY_ID not set." >&2
    exit 1
fi
if [[ "$SEARCH_PROVIDER" == "microsoft" && -z "${MICROSOFT_SEARCH_API_KEY:-}" ]]; then
    echo "ERROR: SEARCH_PROVIDER=microsoft but MICROSOFT_SEARCH_API_KEY not set." >&2
    exit 1
fi
if [[ "$BROWSE_PROVIDER" == "scrapedo" && -z "${SCRAPEDO_API_KEY:-}" ]]; then
    echo "ERROR: BROWSE_PROVIDER=scrapedo but SCRAPEDO_API_KEY not set." >&2
    exit 1
fi
if [[ "$BROWSE_PROVIDER" == "microsoft" && -z "${MICROSOFT_BROWSE_API_KEY:-}" ]]; then
    echo "ERROR: BROWSE_PROVIDER=microsoft but MICROSOFT_BROWSE_API_KEY not set." >&2
    exit 1
fi

# Refuse to start if already running
for name in search_server browser_server; do
    pid_file="$LOG_DIR/${name}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "[setup_seal0] $name already running (pid $(cat "$pid_file")). Run --stop first." >&2
        exit 1
    fi
done

echo "[setup_seal0] Starting search_server (port $SEARCH_PORT, provider $SEARCH_PROVIDER)..."
SEARCH_SERVER_PORT="$SEARCH_PORT" \
    nohup conda run --no-capture-output -n "$CONDA_ENV" \
    python -m skilllens.benchmarks.seal0.search_server \
    > "$LOG_DIR/search_server.log" 2>&1 &
echo $! > "$LOG_DIR/search_server.pid"

echo "[setup_seal0] Starting browser_server (port $BROWSER_PORT, provider $BROWSE_PROVIDER)..."
BROWSER_SERVER_PORT="$BROWSER_PORT" \
    nohup conda run --no-capture-output -n "$CONDA_ENV" \
    python -m skilllens.benchmarks.seal0.browser_server \
    > "$LOG_DIR/browser_server.log" 2>&1 &
echo $! > "$LOG_DIR/browser_server.pid"

# Wait for servers to come up
for port in "$SEARCH_PORT" "$BROWSER_PORT"; do
    for _ in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:$port/docs" >/dev/null 2>&1 || \
           curl -sf "http://127.0.0.1:$port/" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
done

cat <<EOF

[setup_seal0] Servers started.
  search_server  : http://127.0.0.1:$SEARCH_PORT   (log: $LOG_DIR/search_server.log)
  browser_server : http://127.0.0.1:$BROWSER_PORT   (log: $LOG_DIR/browser_server.log)

Stop with : bash scripts/setup_seal0.sh --stop
Status    : bash scripts/setup_seal0.sh --status
EOF
