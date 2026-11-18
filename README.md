# ExpGuard
Official implementation of "ExpGuard: Guarding Long-Horizon LLM Agents against Corrupted Experience".

## How to Use

Install dependencies with Python 3.11 or later.

```bash
python -m pip install .
```

Extract experiences from recorded trajectories.

```bash
python -m expguard extract --input path/to/trajectories.jsonl \
  --reasoningbank path/to/reasoning-bank --library experiences.jsonl \
  --endpoint "$LLM_BASE_URL" --operator-model "$LLM_MODEL"
```

Build graphs and labels from recorded replay pairs.

```bash
python -m expguard import-replays --benchmark MemForm \
  --input path/to/paired_cases.jsonl --library experiences.jsonl --output labels.jsonl
```

Train the risk model.

```bash
python -m expguard train --benchmark MemForm --input labels.jsonl --checkpoint risk_model.pt
```

Evaluate detection on the held-out split.

```bash
python -m expguard evaluate-detection --benchmark MemForm \
  --input labels.jsonl --checkpoint risk_model.pt
```

Evaluate online defense from state checkpoints.

```bash
python -m expguard evaluate-online --benchmark STATE-Bench \
  --input path/to/online_cases.jsonl --library experiences.jsonl \
  --checkpoint risk_model.pt --output online_runs.jsonl \
  --endpoint "$LLM_BASE_URL" --operator-model "$LLM_MODEL"
```

## External Dependencies

- [ReasoningBank](https://github.com/google-research/reasoning-bank) supplies the extraction prompts.
- [BGE-M3](https://github.com/FlagOpen/FlagEmbedding) supplies the encoder weights published as `BAAI/bge-m3`.
- [MemForm](https://github.com/opanhw/memform-code) supplies the offline benchmark data.
- [STATE-Bench](https://github.com/microsoft/STATE-Bench) supplies the online benchmark data.

Prepare benchmark cases in ExpGuard JSONL format. Set `LLM_BASE_URL` and `LLM_MODEL` for an OpenAI-compatible endpoint, and set `EXPGUARD_API_KEY` if required.
