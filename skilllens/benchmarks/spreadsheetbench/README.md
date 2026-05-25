# SpreadsheetBench Benchmark Integration

Inference runner + docker-sandboxed Jupyter executor for
[SpreadsheetBench](https://github.com/RUCKBReasoning/SpreadsheetBench).

## Prerequisites

### Python dependencies

```bash
pip install "skilllens[spreadsheet]"
```

### Dataset (~15 MB tarball → ~1.3 GB extracted)

```bash
mkdir -p data/test_pool/spreadsheetbench/raw \
         data/test_pool/spreadsheetbench/sb_root/data

wget -O data/test_pool/spreadsheetbench/raw/verified_400.tar.gz \
    https://raw.githubusercontent.com/RUCKBReasoning/SpreadsheetBench/main/data/spreadsheetbench_verified_400.tar.gz

tar -xzf data/test_pool/spreadsheetbench/raw/verified_400.tar.gz \
    -C data/test_pool/spreadsheetbench/sb_root/data
```

The held-out 200 ids in `data/test_pool/spreadsheetbench/testset_v1.json`
are a strict subset of the 400-id `dataset.json` from this download.

### Docker sandbox

`sb-api` (long-running) spawns `sb-executor` sibling containers per
conversation.

```bash
cd skilllens/benchmarks/spreadsheetbench/code_exec_docker
docker build -t sb-executor -f Dockerfile.executor .   # ~14 GB
docker build -t sb-api      -f Dockerfile.api      .   # ~1.2 GB
cd -

# Point sb-api at the dataset root (must be ABSOLUTE).
DATA_ROOT_ABS="$(pwd)/data/test_pool/spreadsheetbench/sb_root/data/spreadsheetbench_verified_400"
python3 -c "import json; json.dump({'volumes_path': '$DATA_ROOT_ABS'}, \
  open('skilllens/benchmarks/spreadsheetbench/code_exec_docker/config.json','w'), indent=2)"

docker run -d --name sb-api --network host \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$(pwd)/skilllens/benchmarks/spreadsheetbench/code_exec_docker/config.json:/app/config.json" \
    sb-api python3 api.py --port 8081
```

> The `volumes_path` must point at the directory containing
> `dataset.json` and `spreadsheet/` — **not** one level deeper into
> `spreadsheet/`. Otherwise outputs land in the wrong place and every
> sample evaluates to 0.

Set `AZURE_OPENAI_ENDPOINT` + (`AZURE_OPENAI_API_KEY` or `AZURE_CLIENT_ID`)
in `.env`.

## Pipeline

### Raw experience generation

```bash
python -m skilllens infer \
    --benchmark spreadsheetbench --model gpt-5.4 \
    --endpoint "$AZURE_OPENAI_ENDPOINT" \
    --workers 32 --num-rounds 1 --reasoning-effort medium \
    --max-turn-num 10
```

Reads the 200 held-out ids from `testset_v1.json`. Outputs land in
`inference_output/spreadsheetbench/gpt-5.4_baseline_<ts>/`.

### Schema normalization

```bash
RUN=inference_output/spreadsheetbench/gpt-5.4_baseline_<your-ts>

python -m skilllens convert \
    --trajectory-dir "$RUN" \
    --eval-result    "$RUN/eval_result.json" \
    --benchmark      spreadsheetbench \
    --model-name     gpt-5.4 \
    -o data/experience_pool/spreadsheetbench/gpt-5.4_baseline_200.json
```

### Skill extraction

```bash
python -m skilllens extract \
    -c configs/examples/spreadsheetbench_parallel.yaml \
    -i data/experience_pool/spreadsheetbench/gpt-5.4_baseline_200.json \
    -o extraction_output/spreadsheetbench_parallel/
```

### Skill consumption

```bash
SKILL=extraction_output/spreadsheetbench_parallel/spreadsheetbench/parallel_b0_g8_<extractor>_<pool>_<ts>/skill_set.json

python -m skilllens infer \
    --benchmark spreadsheetbench --model gpt-5.4 \
    --endpoint "$AZURE_OPENAI_ENDPOINT" \
    --workers 32 --num-rounds 1 --reasoning-effort medium \
    --max-turn-num 10 \
    --skill-set "$SKILL" \
    --output-dir inference_output/spreadsheetbench/gpt-5.4_with_skill_200
```

## Cleanup

```bash
docker rm -f sb-api
docker ps --format '{{.Names}}' | grep '^conv-' | xargs -r docker rm -f
```
