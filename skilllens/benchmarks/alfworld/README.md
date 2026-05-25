# ALFWorld Benchmark Integration

TextWorld wrapper for ALFWorld, consumed by `skilllens/inference/alfworld/`.

## Prerequisites

```bash
pip install "skilllens[alfworld]"

# Download game data (~180 MB) into $ALFWORLD_DATA (default: ~/.cache/alfworld).
bash scripts/setup_alfworld.sh
export ALFWORLD_DATA="$HOME/.cache/alfworld"
```

Also set `AZURE_OPENAI_ENDPOINT` + (`AZURE_OPENAI_API_KEY` or `AZURE_CLIENT_ID`) in `.env`.

## Pipeline

### Raw experience generation

```bash
python -m skilllens infer \
    --benchmark alfworld \
    --model gpt-5.4 \
    --endpoint "$AZURE_OPENAI_ENDPOINT" \
    --workers 16 --num-rounds 1 --reasoning-effort medium
```

Enumerates the 134 `valid_unseen` games. Outputs land in
`inference_output/alfworld/gpt-5.4_baseline_<ts>/`.

### Schema normalization

```bash
RUN=inference_output/alfworld/gpt-5.4_baseline_<your-ts>

python -m skilllens convert \
    --trajectory-dir "$RUN" \
    --benchmark      alfworld \
    --model-name     gpt-5.4 \
    -o data/experience_pool/alfworld/gpt-5.4_baseline_134.json
```

ALFWorld embeds the success signal per trajectory, so no `--eval-result`
is needed.

### Skill extraction

```bash
python -m skilllens extract \
    -c configs/examples/alfworld_parallel.yaml \
    -i data/experience_pool/alfworld/gpt-5.4_baseline_134.json \
    -o extraction_output/alfworld_parallel/
```

### Skill consumption

```bash
SKILL=extraction_output/alfworld_parallel/alfworld/parallel_b0_g8_<extractor>_<pool>_<ts>/skill_set.json

python -m skilllens infer \
    --benchmark alfworld \
    --model gpt-5.4 \
    --endpoint "$AZURE_OPENAI_ENDPOINT" \
    --workers 16 --num-rounds 1 --reasoning-effort medium \
    --skill-set "$SKILL" \
    --output-dir inference_output/alfworld/gpt-5.4_with_skill_134
```
