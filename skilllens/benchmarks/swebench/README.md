# SWE-bench Benchmark Integration

Vendored fork of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
driving SWE-bench Verified inside per-instance Docker containers, consumed
by `skilllens/inference/swebench/`.

## Prerequisites

```bash
pip install "skilllens[swebench]"
pip install swebench        # official harness (kept optional in extras)
```

Set in `.env`:

```bash
AZURE_OPENAI_ENDPOINT=...                    # plus AZURE_OPENAI_API_KEY or AZURE_CLIENT_ID
AZURE_API_BASE=$AZURE_OPENAI_ENDPOINT        # litellm requires these two
AZURE_API_VERSION=2025-04-01-preview
```

The 250 held-out test ids are committed at
`data/test_pool/swebench/testset_v1.json`. Instance metadata + patches
are pulled lazily from `princeton-nlp/SWE-bench_Verified` on HuggingFace.

Docker must be running. Each instance image is ~2–8 GB; the full 250-id
split needs ~500 GB once warm. Verify with `docker ps`.

## Pipeline

### Raw experience generation

```bash
python -m skilllens infer \
    --benchmark swebench --model gpt-5.4 \
    --endpoint "$AZURE_OPENAI_ENDPOINT" \
    --workers 8 --num-rounds 1 \
    --benchmark-args \
        subset=verified \
        split=test \
        instance-ids=data/test_pool/swebench/testset_v1.json \
        c=skilllens/benchmarks/swebench/minisweagent/config/benchmarks/swebench.yaml
```

Outputs land in `inference_output/swebench/gpt-5.4_baseline_<ts>/`, with
per-instance `<instance_id>/<instance_id>.traj.json`, `preds.json`, and
the swebench-harness report `<run_id>.json`.

### Schema normalization

```bash
RUN=inference_output/swebench/gpt-5.4_baseline_<your-ts>

python -m skilllens convert \
    --trajectory-dir "$RUN" \
    --eval-result    "$RUN/<run_id>.json" \
    --benchmark      swebench \
    --model-name     gpt-5.4 \
    -o data/experience_pool/swebench/gpt-5.4_baseline_250.json
```

The `<run_id>.json` filename matches `OUTPUT_DIR`'s basename (e.g.
`azure__gpt-5.4.gpt-5.4_baseline_<ts>.json`).

### Skill extraction

```bash
python -m skilllens extract \
    -c configs/examples/swebench_parallel.yaml \
    -i data/experience_pool/swebench/gpt-5.4_baseline_250.json \
    -o extraction_output/swebench_parallel/
```

### Skill consumption

```bash
SKILL=extraction_output/swebench_parallel/swebench/parallel_b0_g8_<extractor>_<pool>_<ts>/skill_set.json

python -m skilllens infer \
    --benchmark swebench --model gpt-5.4 \
    --endpoint "$AZURE_OPENAI_ENDPOINT" \
    --workers 8 --num-rounds 1 \
    --skill-set "$SKILL" \
    --output-dir inference_output/swebench/gpt-5.4_with_skill_250 \
    --benchmark-args \
        subset=verified \
        split=test \
        instance-ids=data/test_pool/swebench/testset_v1.json \
        c=skilllens/benchmarks/swebench/minisweagent/config/benchmarks/swebench.yaml
```
