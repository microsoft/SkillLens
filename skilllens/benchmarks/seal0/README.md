# SEAL-0 (LiteResearcher) Benchmark Integration

LiteResearcher web-research agent with local search + browse FastAPI
shims, consumed by `skilllens/inference/seal0/`.

## Prerequisites

```bash
pip install "skilllens[seal0]"
```

### Tool providers

The agent calls two local FastAPI shims (`search_server.py` /
`browser_server.py`) which front pluggable upstream providers. The
default is a Google-backed search + a public browse provider:

| Tool | Default provider | Env var | Sign up |
|------|------------------|---------|---------|
| Search | [Serper](https://serper.dev) (Google Search API) | `SERPER_KEY_ID` | https://serper.dev |
| Browse | [ScrapeDo](https://scrape.do) | `SCRAPEDO_API_KEY` | https://scrape.do |

Put the keys in `.env`:

```bash
SEARCH_PROVIDER=serper
SERPER_KEY_ID=<your-serper-key>

BROWSE_PROVIDER=scrapedo
SCRAPEDO_API_KEY=<your-scrapedo-key>

AZURE_OPENAI_ENDPOINT=...                    # plus AZURE_OPENAI_API_KEY or AZURE_CLIENT_ID
```

> A second, internal search + browse backend is also wired in (selectable
> via `SEARCH_PROVIDER=microsoft` / `BROWSE_PROVIDER=microsoft`), as a
> drop-in replacement for the Serper / ScrapeDo pair. The SkillLens
> experiments in the paper were run against this internal backend.

### Start the tool servers (ports 8001 / 8002)

```bash
bash scripts/setup_seal0.sh             # start
bash scripts/setup_seal0.sh --status    # verify
bash scripts/setup_seal0.sh --stop      # stop
```

### Test split

The held-out test split is `data/test_pool/seal0/testset_v1.json` (54
ids); the materialized JSONL is
`data/test_pool/seal0/sealqa_seal_0_testset_v1.jsonl`.

## Pipeline

### Raw experience generation

```bash
python -m skilllens infer \
    --benchmark seal0 --model gpt-5.4 \
    --workers 20 --num-rounds 1 --reasoning-effort medium \
    --benchmark-args \
        dataset=data/test_pool/seal0/sealqa_seal_0_testset_v1.jsonl
```

Outputs land in `inference_output/seal0/gpt-5.4_baseline_<ts>/`.

### Schema normalization

```bash
RUN=inference_output/seal0/gpt-5.4_baseline_<your-ts>

python -m skilllens convert \
    --trajectory-dir "$RUN" \
    --benchmark      seal0 \
    --model-name     gpt-5.4 \
    -o data/experience_pool/seal0/gpt-5.4_baseline_54.json
```

The judge verdict is stored inline in each rollout, so no
`--eval-result` is needed.

### Skill extraction

```bash
python -m skilllens extract \
    -c configs/examples/seal0_parallel.yaml \
    -i data/experience_pool/seal0/gpt-5.4_baseline_54.json \
    -o extraction_output/seal0_parallel/
```

### Skill consumption

```bash
SKILL=extraction_output/seal0_parallel/seal0/parallel_b0_g8_<extractor>_<pool>_<ts>/skill_set.json

python -m skilllens infer \
    --benchmark seal0 --model gpt-5.4 \
    --workers 20 --num-rounds 1 --reasoning-effort medium \
    --skill-set "$SKILL" \
    --output-dir inference_output/seal0/gpt-5.4_with_skill \
    --benchmark-args \
        dataset=data/test_pool/seal0/sealqa_seal_0_testset_v1.jsonl
```

The runner renders the skill to a temp file and exports
`SKILL_INJECT_FILE`; `agent.py` reads it and appends the skill body to
the LiteResearcher system prompt.
