#!/bin/bash
# Prepare ALFWorld game data needed by the SkillLens ALFWorld runner.
#
# What this does:
#   1. Picks a data directory (env $ALFWORLD_DATA, falling back to ~/.cache/alfworld).
#   2. Runs `alfworld-download` (shipped with the `alfworld` PyPI package) to
#      fetch json_2.1.1, pddl, tw_pddl, and the Mask R-CNN detector model
#      (~180 MB total) from the upstream GitHub Releases.
#   3. Verifies the resulting directory layout matches what the runner expects.
#
# The runner reads `$ALFWORLD_DATA` at import time — `skilllens/benchmarks/alfworld/config_tw.yaml`
# references it as `$ALFWORLD_DATA/json_2.1.1/...`. After this script finishes,
# export the printed line in your shell (or source it from your .env).
#
# Usage:
#   bash scripts/setup_alfworld.sh                 # uses ~/.cache/alfworld
#   ALFWORLD_DATA=/data/alfworld bash scripts/setup_alfworld.sh
#   bash scripts/setup_alfworld.sh --conda-env skilllens   # run inside a specific conda env

set -euo pipefail

CONDA_ENV=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,17p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Default data location
: "${ALFWORLD_DATA:=$HOME/.cache/alfworld}"
mkdir -p "$ALFWORLD_DATA"

echo "[setup_alfworld] ALFWORLD_DATA=$ALFWORLD_DATA"
export ALFWORLD_DATA

# Pick the python that has alfworld installed.
if [[ -n "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "[setup_alfworld] conda not found but --conda-env was given." >&2
        exit 1
    fi
    DOWNLOAD_CMD=(conda run --no-capture-output -n "$CONDA_ENV" alfworld-download)
else
    if ! command -v alfworld-download >/dev/null 2>&1; then
        cat >&2 <<'EOF'
[setup_alfworld] `alfworld-download` not found on PATH.
Install the alfworld package first, e.g.:
    pip install "alfworld[full]"
or pass --conda-env <name> to run inside a conda environment that has it.
EOF
        exit 1
    fi
    DOWNLOAD_CMD=(alfworld-download)
fi

echo "[setup_alfworld] Running: ${DOWNLOAD_CMD[*]}"
"${DOWNLOAD_CMD[@]}"

# ---- Verify ----
REQUIRED_PATHS=(
    "$ALFWORLD_DATA/json_2.1.1/valid_unseen"
    "$ALFWORLD_DATA/json_2.1.1/valid_seen"
    "$ALFWORLD_DATA/json_2.1.1/train"
    "$ALFWORLD_DATA/logic/alfred.pddl"
    "$ALFWORLD_DATA/logic/alfred.twl2"
    "$ALFWORLD_DATA/detectors/mrcnn.pth"
)

missing=0
for p in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "$p" ]]; then
        echo "[setup_alfworld]   MISSING: $p" >&2
        missing=1
    fi
done

if [[ $missing -ne 0 ]]; then
    echo "[setup_alfworld] One or more required files are missing — see above." >&2
    exit 1
fi

# Count games in valid_unseen (should be 134 for ALFWorld v2.1.1)
n_unseen=$(find "$ALFWORLD_DATA/json_2.1.1/valid_unseen" -name "game.tw-pddl" | wc -l)
echo "[setup_alfworld] valid_unseen game count: $n_unseen (expect 134)"

cat <<EOF

[setup_alfworld] Done.
Add this to your shell or .env so the runner can find the data:

    export ALFWORLD_DATA="$ALFWORLD_DATA"

EOF
