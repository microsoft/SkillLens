#!/bin/bash
# Prepare SpreadsheetBench dependencies for the SkillLens runner.
#
# This script is the automated equivalent of
# skilllens/benchmarks/spreadsheetbench/README.md sections 2 + 3. It:
#
#   1. Downloads the verified_400 tarball (~15 MB) from the upstream
#      RUCKBReasoning/SpreadsheetBench repo and unpacks it into
#      data/test_pool/spreadsheetbench/sb_root/data/spreadsheetbench_verified_400/.
#   2. Builds the two Docker images (sb-api + sb-executor).
#   3. Writes skilllens/benchmarks/spreadsheetbench/code_exec_docker/config.json
#      so sb-api bind-mounts the just-unpacked spreadsheet/ directory.
#   4. Starts the sb-api container on localhost:8081 with --network host
#      (required so sb-api can reach the random ports of the executor sibling
#      containers it spawns).
#
# After this script finishes, the 200 held-out test ids live at
# data/test_pool/spreadsheetbench/testset_v1.json (already in the repo); the
# baseline runner reads that file directly via --dataset_json.
#
# Usage:
#   bash scripts/setup_spreadsheetbench.sh
#   bash scripts/setup_spreadsheetbench.sh --skip-docker     # if you already built images
#   bash scripts/setup_spreadsheetbench.sh --port 8081       # change host port
#
# Prerequisites: docker, wget (or curl), tar.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --------- args ---------
SKIP_DOCKER=0
SB_PORT="${SB_PORT:-8081}"
TARBALL_URL="https://raw.githubusercontent.com/RUCKBReasoning/SpreadsheetBench/main/data/spreadsheetbench_verified_400.tar.gz"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-docker) SKIP_DOCKER=1; shift ;;
        --port) SB_PORT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SB_ROOT="$REPO_ROOT/data/test_pool/spreadsheetbench/sb_root"
DATA_DIR="$SB_ROOT/data/spreadsheetbench_verified_400"
TARBALL_DEST="$REPO_ROOT/data/test_pool/spreadsheetbench/raw/verified_400.tar.gz"

mkdir -p "$SB_ROOT/data" "$(dirname "$TARBALL_DEST")"

# --------- 1. Download + unpack ---------
if [[ -d "$DATA_DIR/spreadsheet" ]] && [[ $(ls "$DATA_DIR/spreadsheet" | wc -l) -ge 400 ]]; then
    echo "[setup_sb] Data already unpacked at $DATA_DIR (skipping download)."
else
    echo "[setup_sb] Downloading verified_400 tarball from RUCKBReasoning/SpreadsheetBench..."
    if command -v wget >/dev/null 2>&1; then
        wget -O "$TARBALL_DEST" "$TARBALL_URL"
    else
        curl -L --fail --progress-bar -o "$TARBALL_DEST" "$TARBALL_URL"
    fi
    echo "[setup_sb] Unpacking into $SB_ROOT/data/..."
    tar -xzf "$TARBALL_DEST" -C "$SB_ROOT/data/"
    rm -f "$TARBALL_DEST"

    n=$(ls "$DATA_DIR/spreadsheet" 2>/dev/null | wc -l)
    echo "[setup_sb] Unpacked $n spreadsheet directories."
    if [[ $n -lt 400 ]]; then
        echo "[setup_sb] WARNING: expected 400 directories, got $n." >&2
    fi
fi

# --------- 2. Build Docker images ---------
if [[ $SKIP_DOCKER -eq 1 ]]; then
    echo "[setup_sb] --skip-docker given, skipping image build + container start."
else
    if ! command -v docker >/dev/null 2>&1; then
        echo "[setup_sb] docker not found on PATH. Install Docker, then re-run." >&2
        exit 1
    fi

    DOCKER_DIR="$REPO_ROOT/skilllens/benchmarks/spreadsheetbench/code_exec_docker"
    pushd "$DOCKER_DIR" >/dev/null

    if ! docker image inspect sb-executor >/dev/null 2>&1; then
        echo "[setup_sb] Building sb-executor (~14 GB, includes jupyter+torch+pandas, ~10 min)..."
        docker build -t sb-executor -f Dockerfile.executor .
    else
        echo "[setup_sb] Image sb-executor already present."
    fi

    if ! docker image inspect sb-api >/dev/null 2>&1; then
        echo "[setup_sb] Building sb-api..."
        docker build -t sb-api -f Dockerfile.api .
    else
        echo "[setup_sb] Image sb-api already present."
    fi
    popd >/dev/null

    # --------- 3. Write config.json + (re)start sb-api ---------
    # volumes_path MUST be the DATASET ROOT (not the spreadsheet/ subdir),
    # because the runner tells the model to read /mnt/data/spreadsheet/<id>/...
    # and write /mnt/data/outputs/<run_tag>/...; the runner then reads those
    # outputs back from <DATA_DIR>/outputs/<run_tag>/ on the host. Pointing
    # volumes_path one level too deep (at spreadsheet/) makes the executor
    # land outputs in <DATA_DIR>/spreadsheet/outputs/<run_tag>/, and every
    # sample evaluates to 0 because the runner can't find them.
    CONFIG_PATH="$DOCKER_DIR/config.json"
    cat > "$CONFIG_PATH" <<EOF
{
  "volumes_path": "$DATA_DIR"
}
EOF
    echo "[setup_sb] Wrote $CONFIG_PATH (volumes_path=$DATA_DIR)"

    if docker ps -a --format '{{.Names}}' | grep -q '^sb-api$'; then
        echo "[setup_sb] Removing existing sb-api container..."
        docker rm -f sb-api >/dev/null
    fi

    echo "[setup_sb] Starting sb-api container (--network host, port $SB_PORT)..."
    docker run -d \
        --name sb-api \
        --restart unless-stopped \
        --network host \
        -v "$CONFIG_PATH:/app/config.json:ro" \
        -v /var/run/docker.sock:/var/run/docker.sock \
        sb-api \
        python3 /app/api.py --port "$SB_PORT" >/dev/null

    # Wait until healthy
    for i in {1..30}; do
        if curl -sf "http://localhost:$SB_PORT/" >/dev/null 2>&1 || \
           docker logs sb-api 2>&1 | grep -q "Starting Tornado server" ; then
            break
        fi
        sleep 1
    done
    echo "[setup_sb] sb-api container status: $(docker inspect -f '{{.State.Status}}' sb-api)"
fi

cat <<EOF

[setup_sb] Done.
Data directory : $DATA_DIR
Test split     : data/test_pool/spreadsheetbench/testset_v1.json (200 items)
sb-api         : http://localhost:$SB_PORT/execute

You can now run:
    python -m skilllens infer --benchmark spreadsheetbench --model gpt-5.4 \
        --endpoint "\$AZURE_OPENAI_ENDPOINT" --workers 32 --num-rounds 1
EOF
