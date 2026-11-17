# Ollama Model Tester

A small, dependency-free CLI for running the same prompt — optionally with
images, text or code files — against your local [Ollama](https://ollama.com)
models and saving every response to disk, so you can compare models (or compare
repeated runs of one model) side by side.

It uses only the Python standard library: no `pip install` required.

## Requirements

- Python 3.7 or newer
- [Ollama](https://ollama.com) running locally (the default `http://localhost:11434`),
  or elsewhere with `OLLAMA_HOST` set — see [Using another host](#using-another-host)
- At least one model pulled, e.g. `ollama pull llama3.1:8b`

## Quick start

Make sure Ollama is running, then:

```bash
python3 ollama_model_test.py
```

You'll be asked, in order:

1. **Which model(s)** to use — one or more numbers from your installed models
   (e.g. `1,3`), or `all`
2. **The prompt** — type as many lines as you like, then put `/done` on its own
   line to finish
3. **Files** to add to the prompt — enter a path (or drag a file into the
   terminal), one per line; press Enter on an empty line to move on. See
   [Files](#files)
4. **How many times** to run the prompt
5. **Temperature** (`0.0`–`2.0`), or press Enter to use Ollama's default
6. **Thinking** — only asked for models that support it (qwen3, deepseek-r1,
   gpt-oss, …), once per such model. The choices come from the model itself:
   on/off for some, levels such as low/medium/high for others. Press Enter for
   the model's default
7. Whether to **stream** the responses live to the terminal

It then runs the prompt the requested number of times on each model and writes
the results under `ollama-runs/`.

Results are saved after every run, so pressing Ctrl-C (or a crash) partway
through keeps everything that had finished. Only the run in progress is lost.

## Command-line flags (optional)

Every prompt above can be supplied up front, which makes the tool scriptable.
Anything you omit is still asked interactively.

| Flag | Description |
| --- | --- |
| `--model NAME` | Local model to use (must already be installed). Repeat it to run several models, or pass `all` for every local model that can generate text |
| `--file PATH` | File to add to the prompt: an image, or a text or code file. Repeat it to attach several. `--image` is accepted as another name for it |
| `--runs N` | Number of generations to run |
| `--temperature T` | Temperature, `0.0`–`2.0` |
| `--think VALUE` | Thinking control: `on`, `off`, a level the model supports (e.g. `low`, `medium`, `high`), or `default` to leave it to the model |
| `--prompt-file PATH` | Read the prompt from a UTF-8 text file. No files question is asked; use `--file` |
| `--option NAME=VALUE` | Advanced and never required: set one of Ollama's generation options. See [Advanced options](#advanced-options) |
| `--stream` / `--no-stream` | Stream responses live, or don't |

Example — run a saved prompt three times, fully non-interactive:

```bash
python3 ollama_model_test.py \
  --model llama3.1:8b \
  --prompt-file prompt.txt \
  --runs 3 \
  --temperature 0.7 \
  --think default \
  --no-stream
```

`--think default` is only needed for thinking-capable models, where leaving it
out triggers the thinking question. `--think` applies to every selected model
that supports thinking and is left out for the ones that don't.

Example — compare every local model on the same prompt:

```bash
python3 ollama_model_test.py --model all --prompt-file prompt.txt --runs 3
```

## Files

Files can be added when you type the prompt, or up front with `--file`:

```bash
python3 ollama_model_test.py \
  --model gemma3:4b \
  --model qwen3-vl:8b \
  --file chart.png \
  --file notes.md \
  --prompt-file prompt.txt
```

What happens to a file depends on what it is, judged from its contents rather
than its name:

- **Images** (PNG, JPEG, WebP, GIF) are sent to the model as images. Selected
  models that Ollama reports as not accepting images are skipped, so
  `--model all --file chart.png` runs just the vision models.
- **Text and code** (anything that is valid UTF-8) is placed in the prompt text
  ahead of what you typed, each file labelled with its name.
- Anything else, such as a PDF or a Word document, is refused.

Ollama has no separate channel for text files, so they count against the
model's context window like any other prompt text. If a file is too large for
it, Ollama silently drops the beginning of the input. Your prompt goes last so
that it survives; compare `prompt_eval_count` in a run's metadata against the
size of what you sent if the answers look like the model missed something.

## Advanced options

You never have to set any of these. They are not part of the questions, and
without them every model runs with its own defaults, which is the right choice
unless you have a specific reason to change something.

When you do, pass `--option NAME=VALUE`, once per option:

```bash
python3 ollama_model_test.py --option num_ctx=16384 --option seed=42
```

The ones most likely to be useful:

| Option | What it does | When to reach for it |
| --- | --- | --- |
| `num_ctx` | Size of the model's context window, in tokens: how much prompt plus answer it can hold at once | The tool tells you the prompt filled the context window, which happens with large files. Try `8192` or `16384`; bigger uses more memory |
| `seed` | Fixes the randomness, so the same prompt and settings give the same answer | You want repeatable results, or a fair like-for-like rerun |
| `num_predict` | Maximum number of tokens in the answer | A model rambles or never stops, or you want short answers |
| `top_p`, `top_k` | Narrow or widen the range of words the model picks from | Fine-tuning variety beyond what temperature does |
| `repeat_penalty` | Discourages repeating the same phrases | A model gets stuck in loops |
| `stop` | Text that ends the answer when the model produces it. Can be given more than once | You want output cut at a marker |

The name is checked before anything runs, so a typo such as `num_ctxt` is
reported with a suggestion instead of being silently ignored. Temperature keeps
its own flag, `--temperature`. Whatever you set is recorded with the run and
shown in the comparison table.

## Using another host

The tool reads the same `OLLAMA_HOST` environment variable as the `ollama`
command, so it follows Ollama to another port or machine:

```bash
OLLAMA_HOST=192.168.1.20:11434 python3 ollama_model_test.py
```

## Output

Results are grouped into one folder per prompt:

```text
ollama-runs/
  what-are-the-main-tradeoffs-between_835562a4/
    prompt.md         # the prompt, with its hash and timestamp
    summary.md        # comparison table: one row per model file
    metadata.json     # every run against this prompt (model, timing, options)
    llama3.1-8b.md    # responses + Ollama metadata for this model
    gemma3-1b.md
    image-1.png       # copies of any attached images
```

The folder name is the first few words of the prompt plus a short hash of the
full prompt. Because the folder is keyed on the prompt, **running the same
prompt against a different model drops its output into the same folder** —
making model-to-model comparison easy. Each model's file records every run's
response alongside Ollama's run metadata (token counts, timings, and so on).

`summary.md` is rewritten after every run. It has one row for each model file
in the folder, including ones from earlier sessions, with the settings used,
output tokens, tokens per second, time, model load time, and how much of the
context window the prompt took. If a prompt filled the whole window, the
summary and the terminal both say so, since the model then probably did not
see all of the input. Rows for results saved by older versions of the tool show
`–` where the figures were not recorded.

Added files count as part of the prompt: the hash covers them too, so the same
question about a different picture or document gets its own folder. Images are
copied into the folder (as `image-1.png`, `image-2.jpg`, … in the order given).
Text files are not copied separately; `prompt.md` holds the complete text that
was sent, files included. Each file's original name and SHA-256 hash is listed
in `prompt.md` and `metadata.json`.

Each model's digest is recorded in its file and in `metadata.json`. Tags such
as `llama3.1:8b` can point at a different build after an `ollama pull`; the
digest says exactly which one produced a result.

For thinking models, each run's reasoning is saved under its own `Thinking`
heading ahead of the response, and streamed runs show it live (dimmed) before
the answer. The thinking setting used is recorded in both the model's file and
`metadata.json` (`null` there means the model's default).
