# Ollama Model Tester

This small CLI asks for a prompt, lists the Ollama models installed on this machine, lets you choose a model, asks you how many times you want to run the prompt, and saves the results.

## Usage

Make sure Ollama is running, then run:

```bash
python3 ollama_model_test.py
```

Enter a multiline prompt, then put `/done` on its own line to finish prompt entry.

Results are written under `ollama-runs/` in one folder per prompt:

```text
ollama-runs/
  what-are-the-main-tradeoffs_a83f21c4/
    prompt.md
    metadata.json
    llama3.1-8b.md
    gemma3-1b.md
```

The folder name includes the first few words of the prompt and a short hash of the full prompt. Reusing the exact same prompt with a different model saves that model's output in the same folder.
