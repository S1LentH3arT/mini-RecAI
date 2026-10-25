# mini-RecAI

A standalone working copy of the InteRecAgent framework from
[Microsoft RecAI](https://github.com/microsoft/RecAI), with Steam data generation,
single-turn evaluation, and diagnostic replay scripts.

This is an experimental reproduction workspace. Successful execution does not
establish reproduction of the paper's results; the current evaluator uses fuzzy
title matching over answer text rather than strict catalog-ID Recall@5.

## Setup

Use Python 3.9. The shell scripts expect a sibling `.recai` environment:

```bash
python3.9 -m venv ../.recai
../.recai/bin/python -m pip install -r requirements.txt
```

Restore the resources named by `resources/<domain>/settings.json`: item metadata,
SASRec checkpoint, and item similarity matrix. Raw datasets, model binaries,
generated evaluation inputs and results are intentionally not committed.
Embedding models used by the gallery and dynamic demonstrations must also be
available locally or downloadable.

## API configuration

```bash
export DASHSCOPE_API_KEY='your-api-key'
export OPENAI_API_BASE='https://YOUR_WORKSPACE.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
```

Choose a model available to your account in the relevant shell script. For
separate providers, set `API_KEY` / `API_BASE` for the interactive and single-turn
scripts, and `OPENAI_API_KEY` / `OPENAI_API_BASE` plus `SIMULATOR_API_KEY` /
`SIMULATOR_API_BASE` for simulator evaluation. Each script optionally sources
its own ignored `*.local.env` file, which can override exported settings.

## Run

```bash
bash run.sh
bash eval_single_turn.sh
bash eval_simulator.sh
```

Single-turn retrieval expects
`test/steam/retrieval_eval_data_500.jsonl`, with one `context` / `target` object
per line. Data generation and conversion entrypoints are in `eval/`.

To replay exactly the first 40 records using the single-turn script's settings:

```bash
../.recai/bin/python test/steam/repro_first40/run.py
```

The replay disables proxy environment variables and writes `trace.jsonl`,
`run.log`, `answers.jsonl`, `input.jsonl`, and `manifest.json` beside the runner.
These files are overwritten on each run; archive previous results first.

## Offline checks

```bash
../.recai/bin/python -m unittest discover -s eval -p 'test_*.py'
../.recai/bin/python test/test_open_ai_response.py
```

The replay self-check additionally requires the local retrieval dataset.

Upstream source retains its Microsoft copyright notices and MIT license.
