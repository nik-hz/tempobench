# Tempo-Bench

Formally grounded **LLM benchmark** for temporal reasoning over automata/traces. Runs locally via a clean CLI or Python API. Keeps the wheel thin (code + tiny samples) and lets you plug in **any model** (OpenAI-compatible, Hugging Face, vLLM, or your own class/function).

---

## Features
- Tasks: **trace acceptance** & **temporal causality** (with per-feature metrics).
- Backends: OpenRouter/OpenAI (OpenAI-compatible), **HF pipelines**, **vLLM**, or **custom** Python adapters.
- Outputs: row-wise JSONL + CSV with accuracy and F1s (AP & timestep).
- Reproducible runs: fixed seeds, manifest-friendly outputs, small packaged sample datasets.

The benchmarking datasets can be found at `https://huggingface.co/datasets/nikolausholzer/tempobench`.

---

## Install

```bash
pip install tempobench
```

Python ≥ 3.10 recommended.

---

## Quickstart (CLI)

```bash
# OpenRouter (OpenAI-compatible) example
export OPENROUTER_API_KEY=YOUR_KEY

tempobench run \
  --dataset_path src/tempobench/data/causal-done.jsonl \
  --task causal \
  --backend openrouter \
  --model-id openai/gpt-4o-mini \
  --gen-args '{"temperature":0.0,"max_tokens":256}' \
  --outdir benchmark_results --console-prints
```

Other backends:
This is not implemented yet and is an open **issue** currently.

```bash
# OpenAI
export OPENAI_API_KEY=YOUR_KEY
tempobench run --dataset_path ... --task causal --backend openai --model-id gpt-4o-mini

# Hugging Face (local model)
tempobench run --dataset_path ... --task causal \
  --backend hf --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model-args '{"device":0}' --gen-args '{"max_new_tokens":256}'

# vLLM server (OpenAI API compatible)
tempobench run --dataset_path ... --task trace \
  --backend vllm --model-id my-vllm \
  --model-args '{"base_url":"http://127.0.0.1:8000/v1","api_key":"nokey"}'
```

**Outputs** land under `benchmark_results/<task>/` as both `.jsonl` and `.csv`.

---

## Python API
You can use the benchmarker to build custom benchmarking workflows that use tempobench logic. Check out the benchmark.py file out on the project github.


```python
from tempobench import Benchmark

bench = Benchmark(
    dataset_path="src/tempobench/data/causal-done.jsonl",
    task="causal",
    model_id="openai/gpt-4o-mini",
    results_dir="benchmark_results",
    console_prints=True,
)

df = bench.evaluate()
print(df.head())
```

---

## Datasets

Check my huggingface for the tempobench public benchmarking datasets.

If you are interested in access to our datasets for reasoning SFT, reach out to me.

---

## Env vars
You will need to have these env vars set for this to work properly.

- `OPENROUTER_API_KEY` (for `--backend openrouter`)
- `OPENAI_API_KEY` (for `--backend openai`)

---

## Results schema (per row)

`results_*.jsonl` contains:
```json
{
  "model": "openai/gpt-4o-mini",
  "gold": "... (gold JSON) ...",
  "pred": "... (raw text) ...",
  "GT": { "...parsed..." },
  "PRED": { "...parsed..." },
  "correct": true,
  "precision_ap": 1.0,
  "recall_ap": 1.0,
  "F1_ap": 1.0,
  "precision_timestep": 1.0,
  "recall_timestep": 1.0,
  "F1_timestep": 1.0,
  "cost": 0.0023,
  "generation_id": "gen_...",
  "native_prompt_tokens": 123,
  "native_completion_tokens": 45
}
```

## License

MIT (see `LICENSE`).

If you use this benchmark, please cite our paper!
