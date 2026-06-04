# Ollama Model Tester

A small, dependency-free CLI for running the same prompt against your local
[Ollama](https://ollama.com) models and saving every response to disk — so you
can compare models (or compare repeated runs of one model) side by side.

It uses only the Python standard library: no `pip install` required.

## Requirements

- Python 3.7 or newer
- [Ollama](https://ollama.com) running locally (the default `http://localhost:11434`)
- At least one model pulled, e.g. `ollama pull llama3.1:8b`

## Quick start

Make sure Ollama is running, then:

```bash
python3 ollama_model_test.py
```

You'll be asked, in order:

1. **Which model** to use (pick a number from your installed models)
2. **The prompt** — type as many lines as you like, then put `/done` on its own
   line to finish
3. **How many times** to run the prompt
4. **Temperature** (`0.0`–`2.0`), or press Enter to use Ollama's default
5. Whether to **stream** the responses live to the terminal

It then runs the prompt the requested number of times and writes the results
under `ollama-runs/`.

## Command-line flags (optional)

Every prompt above can be supplied up front, which makes the tool scriptable.
Anything you omit is still asked interactively.

| Flag | Description |
| --- | --- |
| `--model NAME` | Local model to use (must already be installed) |
| `--runs N` | Number of generations to run |
| `--temperature T` | Temperature, `0.0`–`2.0` |
| `--prompt-file PATH` | Read the prompt from a UTF-8 text file |
| `--stream` / `--no-stream` | Stream responses live, or don't |

Example — run a saved prompt three times, fully non-interactive:

```bash
python3 ollama_model_test.py \
  --model llama3.1:8b \
  --prompt-file prompt.txt \
  --runs 3 \
  --temperature 0.7 \
  --no-stream
```

## Output

Results are grouped into one folder per prompt:

```text
ollama-runs/
  what-are-the-main-tradeoffs-between_835562a4/
    prompt.md         # the prompt, with its hash and timestamp
    metadata.json     # every run against this prompt (model, timing, options)
    llama3.1-8b.md    # responses + raw Ollama metadata for this model
    gemma3-1b.md
```

The folder name is the first few words of the prompt plus a short hash of the
full prompt. Because the folder is keyed on the prompt, **running the same
prompt against a different model drops its output into the same folder** —
making model-to-model comparison easy. Each model's file records every run's
response alongside Ollama's raw metadata (token counts, timings, and so on).
