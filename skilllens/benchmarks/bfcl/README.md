# BFCL v4 Benchmark Integration

Embedded BFCL v4 evaluation harness, consumed by `skilllens/inference/bfcl/`.

## Prerequisites

```bash
pip install "skilllens[bfcl]"
```

BFCL multi-turn test data is bundled inside the embedded `bfcl_eval`
package; no external download is needed. The SkillLens held-out test
split (300 ids across the 3 multi-turn categories) lives at
`data/test_pool/bfcl/testset_v1.json`.

Set `AZURE_OPENAI_ENDPOINT` + (`AZURE_OPENAI_API_KEY` or `AZURE_CLIENT_ID`)
in `.env`.

## Pipeline

### Raw experience generation

```bash
python -m skilllens infer \
    --benchmark bfcl --model gpt-5.4 \
    --num-rounds 1 --workers 20 --reasoning-effort medium \
    --output-dir inference_output/bfcl/gpt-5.4_baseline \
    --benchmark-args include-ids-file=data/test_pool/bfcl/testset_v1.json
```

Default categories: `multi_turn_long_context`, `multi_turn_miss_func`,
`multi_turn_miss_param`. Override with
`--benchmark-args test-category=cat1,cat2,...`.

### Schema normalization

```bash
RUN=inference_output/bfcl/gpt-5.4_baseline_<your-ts>/round_0

python -m skilllens convert \
    --trajectory-dir "$RUN" \
    --benchmark      bfcl \
    --model-name     gpt-5.4 \
    -o data/experience_pool/bfcl/gpt-5.4_baseline_300.json
```

No `--eval-result` is needed — the converter walks
`<run>/score/**/*_score.json` to backfill each trajectory's outcome.

### Skill extraction

```bash
python -m skilllens extract \
    -c configs/examples/bfcl_parallel.yaml \
    -i data/experience_pool/bfcl/gpt-5.4_baseline_300.json \
    -o extraction_output/bfcl_parallel/
```

### Skill consumption

```bash
SKILL=extraction_output/bfcl_parallel/bfcl/parallel_b0_g8_<extractor>_<pool>_<ts>/skill_set.json

python -m skilllens infer \
    --benchmark bfcl --model gpt-5.4 \
    --num-rounds 1 --workers 20 --reasoning-effort medium \
    --skill-set "$SKILL" \
    --output-dir inference_output/bfcl/gpt-5.4_with_skill \
    --benchmark-args include-ids-file=data/test_pool/bfcl/testset_v1.json
```

The runner renders the skill to a temp file and exports
`SKILL_INJECT_FILE`; `bfcl_eval`'s prompt builder appends the skill body
to the developer / system message of every turn.
