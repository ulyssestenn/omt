#!/usr/bin/env python3
"""Run the same prompt against a local Ollama model and save the outputs."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import difflib
import hashlib
import http.client
import io
import json
import math
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, TextIO


OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_PORT = 11434


def resolve_ollama_host(value: str | None) -> str:
    """Turn an ``OLLAMA_HOST`` value into a base URL, the way the Ollama CLI does.

    Accepts ``host``, ``host:port``, ``:port`` and full ``scheme://host[:port]``
    forms. Anything unparseable falls back to the local default.
    """
    text = (value or "").strip().strip("\"'").strip()
    if not text:
        return OLLAMA_DEFAULT_HOST

    scheme, separator, rest = text.partition("://")
    if separator:
        scheme = scheme.lower()
        default_port = {"http": 80, "https": 443}.get(scheme, OLLAMA_DEFAULT_PORT)
    else:
        scheme, rest, default_port = "http", text, OLLAMA_DEFAULT_PORT

    try:
        parts = urllib.parse.urlsplit(f"{scheme}://{rest}")
        hostname = parts.hostname or "localhost"
        port = parts.port or default_port
    except ValueError:
        return OLLAMA_DEFAULT_HOST

    # 0.0.0.0 is what people set to make the server listen on every interface;
    # as a destination it only works on some platforms, so talk to loopback.
    if hostname == "0.0.0.0":
        hostname = "127.0.0.1"
    elif hostname == "::":
        hostname = "::1"
    if ":" in hostname:
        hostname = f"[{hostname}]"

    return f"{scheme}://{hostname}:{port}{parts.path.rstrip('/')}"


OLLAMA_HOST = resolve_ollama_host(os.environ.get("OLLAMA_HOST"))
OUTPUT_ROOT = Path("ollama-runs")
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0
INVALID_TEMPERATURE = object()

# Spellings accepted for the on/off/default forms of Ollama's ``think`` request
# field. Anything else is passed through as a model-defined level ("low", ...).
THINK_ON_WORDS = frozenset({"on", "true", "yes", "y"})
THINK_OFF_WORDS = frozenset({"off", "false", "no", "n", "none"})
THINK_DEFAULT_WORDS = frozenset({"", "default"})
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"

# Fields dropped from the saved Ollama metadata: ``context`` is a large array of
# token IDs only useful for resuming a conversation, while ``response`` and
# ``thinking`` duplicate the text already written above it as the run output.
SAVED_METADATA_OMIT_KEYS = frozenset({"context", "response", "thinking"})

# Generation options Ollama accepts in a request's ``options``, and the type of
# value each takes. Ollama silently ignores names it doesn't know, so --option
# is checked against this list to catch typos. ``temperature`` is left out
# because it has its own flag and question.
OLLAMA_OPTION_TYPES = {
    "num_ctx": int,
    "seed": int,
    "num_predict": int,
    "top_k": int,
    "top_p": float,
    "min_p": float,
    "typical_p": float,
    "repeat_last_n": int,
    "repeat_penalty": float,
    "presence_penalty": float,
    "frequency_penalty": float,
    "stop": str,
    "num_keep": int,
    "num_batch": int,
    "num_gpu": int,
    "main_gpu": int,
    "num_thread": int,
    "draft_num_predict": int,
    "use_mmap": bool,
}
OPTION_TRUE_WORDS = frozenset({"true", "yes", "on", "1"})
OPTION_FALSE_WORDS = frozenset({"false", "no", "off", "0"})

# Figures copied from Ollama's response into metadata.json for each run, which
# is what the summary table is built from.
RUN_STAT_KEYS = (
    "eval_count",
    "eval_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "load_duration",
    "total_duration",
    "done_reason",
)
# Listing models and looking up their details should take moments, so those
# calls give up quickly. Generation has no time limit unless --timeout sets one:
# a slow run that is still working is worth more than one cut off.
QUICK_TIMEOUT_SECONDS = 30.0
# Wording Ollama uses when it refuses input that is too long for the context
# window, which it does (instead of cutting the input) when asked not to truncate.
CONTEXT_ERROR_PHRASES = (
    "context length",
    "context size",
    "context window",
    "exceeds the available context",
)
SUMMARY_SYSTEM_PREVIEW_CHARS = 40
SUMMARY_FILE_NAME = "summary.md"
MISSING = "–"

# ``--model all`` (or answering "all" interactively) selects every local model.
ALL_MODELS = "all"

# How the start of a file identifies it as an image Ollama may be able to read.
# Anything else has to be UTF-8 text, which is sent as part of the prompt.
IMAGE_SIGNATURES = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")

# Subfolder of a prompt's folder that holds copies of the files sent with it.
ATTACHMENTS_DIR = "files"
EXIT_INTERRUPTED = 130


class InputEnded(Exception):
    """A question could not be asked because there is no more input to read."""

    def __init__(self, flag: str) -> None:
        super().__init__(flag)
        self.flag = flag


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except InputEnded as exc:
        print()
        print(
            "Input ended before that question was answered. "
            f"Pass {exc.flag} to answer it up front.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print()
        print("Cancelled.", file=sys.stderr)
        return EXIT_INTERRUPTED


def run(args: argparse.Namespace) -> int:
    print("Ollama model tester")
    print("===================")
    print()

    extra_options = parse_options(args.option or [])
    if extra_options is None:
        return 1

    system = resolve_system(args.system, args.system_file)
    if system is None:
        return 1

    timeout = args.timeout
    if timeout is not None and not (math.isfinite(timeout) and timeout > 0):
        print("--timeout must be a number of seconds greater than zero.", file=sys.stderr)
        return 1

    try:
        installed = list_ollama_models()
    except RuntimeError as exc:
        print(f"Could not list Ollama models at {OLLAMA_HOST}: {exc}", file=sys.stderr)
        print("Make sure Ollama is running, then try again.", file=sys.stderr)
        return 1

    if not installed:
        print("No local Ollama models found. Install one with `ollama pull <model>`.")
        return 1

    models = resolve_models(args.model, list(installed))
    if models is None:
        return 1

    prompt = resolve_prompt(args.prompt_file)
    if prompt is None:
        return 1
    if not prompt.strip():
        print("No prompt entered. Exiting.")
        return 1

    # Files are asked about as part of typing the prompt; a prompt read from a
    # file gets only the files named with --file.
    attachments = resolve_attachments(args.file, ask=args.prompt_file is None)
    if attachments is None:
        return 1
    images = [item for item in attachments if item["kind"] == "image"]
    text_files = [item for item in attachments if item["kind"] == "text"]
    full_prompt = compose_prompt(prompt, text_files)
    if images:
        models = models_accepting_images(models)
        if not models:
            print("None of the selected models accept images.", file=sys.stderr)
            return 1

    runs = resolve_runs(args.runs)
    if runs is None:
        return 1

    temperature = resolve_temperature(args.temperature)
    if temperature is INVALID_TEMPERATURE:
        return 1
    options: dict[str, Any] = dict(extra_options)
    if temperature is not None:
        options["temperature"] = temperature

    # The thinking question is newer than the others. A command line that already
    # answers all of those was written to run unattended, so it is not asked then.
    unattended = (
        bool(args.model)
        and args.prompt_file is not None
        and args.runs is not None
        and args.temperature is not None
        and args.stream is not None
    )
    think_by_model = {
        model: resolve_think(args.think, model, ask=not unattended) for model in models
    }

    stream = resolve_stream(args.stream)

    run_dir = create_prompt_run_dir(prompt, images, text_files)
    prompt_path = run_dir / "prompt.md"
    metadata_path = run_dir / "metadata.json"

    copy_attachments(run_dir, attachments)
    if not prompt_path.exists():
        write_prompt_file(prompt_path, prompt, now_local(), images, text_files)

    print()
    print(f"Saving results to: {run_dir}")

    encoded_images = [base64.b64encode(image["data"]).decode("ascii") for image in images]
    interrupted = False
    for model in models:
        think = think_by_model[model]
        digest = installed.get(model, "")
        output_path = unique_model_output_path(run_dir, model)
        model_system = model_system_prompt(model)
        started_at = now_local()
        results: list[dict[str, Any]] = []
        context_length: int | None = None
        warned_about_context = False
        warned_about_length = False

        print()
        try:
            for index in range(1, runs + 1):
                print(f"Run {index}/{runs} using {model}...")
                try:
                    result = generate_once(
                        model,
                        full_prompt,
                        options,
                        stream=stream,
                        think=think,
                        images=encoded_images,
                        system=system,
                        timeout=timeout,
                    )
                except KeyboardInterrupt:
                    interrupted = True
                    print()
                    break
                results.append(result)
                status = "ok" if result["ok"] else "error"
                elapsed = result["elapsed_seconds"]
                print(f"  {status} in {elapsed:.2f}s")
                if not result["ok"]:
                    print(f"  {result['error']}", file=sys.stderr)
                if context_length is None:
                    context_length = loaded_context_length(model, options)

                input_did_not_fit = is_context_error(result)
                if input_did_not_fit:
                    size = f" ({context_length:,} tokens)" if context_length else ""
                    print(
                        f"  The prompt and its files do not fit in {model}'s context window{size}. "
                        "Nothing was cut: the run was stopped so that you don't get an answer "
                        "based on only part of the input. Give the model more room with "
                        "--option num_ctx=<a larger number>, or send less."
                    )
                elif not warned_about_context and prompt_filled_context(result, context_length):
                    warned_about_context = True
                    print(
                        f"  Note: the prompt filled {model}'s whole context window "
                        f"({context_length:,} tokens), so the model probably did not see all "
                        "of it. Give it more room with --option num_ctx=<a larger number>."
                    )
                if not warned_about_length and stopped_at_length(result):
                    warned_about_length = True
                    print(
                        "  Note: this answer stopped at a length limit, so it is probably "
                        "incomplete. Either the context window filled up (raise it with "
                        "--option num_ctx=<a larger number>) or num_predict was reached."
                    )

                # Save after every run so an interruption or crash later on loses
                # nothing that has already been generated.
                write_model_output_file(
                    output_path,
                    model,
                    prompt,
                    results,
                    options,
                    stream,
                    started_at,
                    think=think,
                    runs_requested=runs,
                    digest=digest,
                    images=images,
                    text_files=text_files,
                    context_length=context_length,
                    system=system,
                    model_system=model_system,
                )
                metadata = save_metadata_run(
                    metadata_path,
                    {
                        "model": model,
                        "digest": digest,
                        "runs_requested": runs,
                        "runs_completed": len(results),
                        "started_at": started_at.isoformat(),
                        "finished_at": now_local().isoformat(),
                        "options": options,
                        "think": think,
                        "system": system or None,
                        "model_system": model_system,
                        "stream": stream,
                        "context_length": context_length,
                        "results": [run_stats(item) for item in results],
                        "files": {
                            "model_output": output_path.name,
                        },
                    },
                    prompt,
                    prompt_path.name,
                    images,
                    text_files,
                )
                write_summary_file(run_dir / SUMMARY_FILE_NAME, metadata)

                if input_did_not_fit:
                    if index < runs:
                        print("  Skipping this model's remaining runs: they would fail the same way.")
                    break
        finally:
            unload_model(model)

        if results:
            finished = len(results) == runs
            print(f"Done. Wrote {output_path}" if finished else f"Wrote {output_path}")
        if interrupted:
            break

    print()
    if interrupted:
        print(
            "Interrupted. Completed runs are saved; the run in progress was discarded.",
            file=sys.stderr,
        )
        return EXIT_INTERRUPTED
    print(f"Comparison table: {run_dir / SUMMARY_FILE_NAME}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the same prompt against a local Ollama model and save the outputs."
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="NAME",
        help=(
            "Name of a local Ollama model to use. Repeat the flag to run several models, "
            f"or pass '{ALL_MODELS}' for every local model. If omitted, choose interactively."
        ),
    )
    parser.add_argument(
        "--file",
        "--image",
        dest="file",
        action="append",
        type=Path,
        metavar="PATH",
        help=(
            "File to add to the prompt. Images go to vision models; text and code files "
            "are included in the prompt text. Repeat the flag to attach several files. "
            "If omitted, you are asked when you type the prompt interactively."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        help="Number of generations to run. If omitted, enter this interactively.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help=(
            f"Generation temperature ({TEMPERATURE_MIN:.1f} to {TEMPERATURE_MAX:.1f}). "
            "If omitted, choose interactively or press Enter for the model's default."
        ),
    )
    parser.add_argument(
        "--think",
        metavar="VALUE",
        help=(
            "Thinking control for thinking-capable models: on, off, a level the model "
            "supports (such as low, medium or high), or default to leave it to the model. "
            "If omitted, you are asked when the model supports thinking, unless the other "
            "questions were all answered by flags."
        ),
    )
    parser.add_argument(
        "--option",
        action="append",
        metavar="NAME=VALUE",
        help=(
            "Advanced, and never required: set one of Ollama's generation options, such as "
            "num_ctx=8192 (context window size) or seed=42 (repeatable output). Repeat the "
            "flag for several. Left out, each model's own defaults apply. "
            "The README explains the ones worth knowing."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help=(
            "Path to a UTF-8 text file containing the prompt. "
            "If omitted, enter the prompt interactively."
        ),
    )
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        "--system",
        metavar="TEXT",
        help=(
            "Optional system prompt: a standing instruction that replaces the model's "
            "built-in one. Never asked for; left out, the model's own is used."
        ),
    )
    system_group.add_argument(
        "--system-file",
        type=Path,
        metavar="PATH",
        help="Read the system prompt from a UTF-8 text file.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help=(
            "Give up on a run when Ollama sends nothing for this many seconds. Without "
            "streaming nothing is sent until the answer is complete, so this then limits "
            "the whole run. Default: wait as long as it takes."
        ),
    )
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        help="Stream responses from /api/generate.",
    )
    stream_group.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Do not stream responses from /api/generate.",
    )
    parser.set_defaults(stream=None)
    return parser.parse_args(argv)


def parse_options(pairs: list[str]) -> dict[str, Any] | None:
    """Turn ``--option NAME=VALUE`` arguments into Ollama options; ``None`` if invalid."""
    options: dict[str, Any] = {}
    for pair in pairs:
        try:
            name, value = parse_option(pair)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return None
        if name == "stop":
            options.setdefault("stop", []).append(value)
        else:
            options[name] = value
    return options


def parse_option(pair: str) -> tuple[str, Any]:
    name, separator, text = pair.partition("=")
    name = name.strip().lower().replace("-", "_")
    text = text.strip()
    if not separator or not name or not text:
        raise ValueError(
            f"--option {pair!r}: options are written NAME=VALUE, for example num_ctx=8192."
        )
    if name == "temperature":
        raise ValueError("Set the temperature with --temperature, not --option.")

    value_type = OLLAMA_OPTION_TYPES.get(name)
    if value_type is None:
        close = difflib.get_close_matches(name, OLLAMA_OPTION_TYPES, n=1)
        hint = f" Did you mean {close[0]}?" if close else ""
        known = ", ".join(OLLAMA_OPTION_TYPES)
        raise ValueError(f"--option {name}: not an option Ollama knows.{hint}\nKnown options: {known}")

    if value_type is str:
        return name, text
    if value_type is bool:
        if text.lower() in OPTION_TRUE_WORDS:
            return name, True
        if text.lower() in OPTION_FALSE_WORDS:
            return name, False
        raise ValueError(f"--option {name}: needs true or false, not {text!r}.")
    try:
        number = value_type(text)
    except ValueError:
        number = None
    if number is None or not math.isfinite(number):
        kind = "a whole number" if value_type is int else "a number"
        raise ValueError(f"--option {name}: needs {kind}, not {text!r}.")
    return name, number


def loaded_context_length(model: str, options: dict[str, Any]) -> int | None:
    """Return the context window, in tokens, that a model is running with.

    Ollama reports it for loaded models, so this is asked after a run. Failing
    that, an explicit ``num_ctx`` option is the best available answer.
    """
    try:
        loaded = ollama_get_json("/api/ps").get("models", [])
    except RuntimeError:
        loaded = []
    for item in loaded if isinstance(loaded, list) else []:
        if isinstance(item, dict) and model in (item.get("name"), item.get("model")):
            if is_positive_int(item.get("context_length")):
                return item["context_length"]

    num_ctx = options.get("num_ctx")
    return num_ctx if is_positive_int(num_ctx) else None


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_context_error(result: dict[str, Any]) -> bool:
    """Whether a run failed because its input was too long for the context window."""
    if result.get("ok"):
        return False
    error = str(result.get("error") or "").lower()
    return any(phrase in error for phrase in CONTEXT_ERROR_PHRASES)


def stopped_at_length(result: dict[str, Any]) -> bool:
    """Whether an answer ended because it hit a limit, not because it was finished."""
    if not result.get("ok"):
        return False
    stats = result.get("raw", result)
    return stats.get("done_reason") == "length"


def prompt_filled_context(result: dict[str, Any], context_length: int | None) -> bool:
    """Whether the prompt alone took up the whole context window, so was cut.

    Only older Ollama versions get this far: they ignore the request not to
    truncate, and cut an over-long prompt down to exactly the window size.
    """
    if not result.get("ok") or context_length is None:
        return False
    prompt_tokens = result["raw"].get("prompt_eval_count")
    return is_positive_int(prompt_tokens) and prompt_tokens >= context_length


def run_stats(result: dict[str, Any]) -> dict[str, Any]:
    """The figures worth keeping from one run, for metadata.json and the summary."""
    stats: dict[str, Any] = {
        "ok": result["ok"],
        "elapsed_seconds": round(result["elapsed_seconds"], 3),
    }
    if result["ok"]:
        raw = result["raw"]
        stats.update({key: raw[key] for key in RUN_STAT_KEYS if key in raw})
    else:
        stats["error"] = result["error"]
    return stats


def write_summary_file(path: Path, metadata: dict[str, Any]) -> None:
    """Write one table row per model file in the folder, for comparing them."""
    headers = [
        "Output",
        "Model",
        "Thinking",
        "System prompt",
        "Options",
        "Runs",
        "Output tokens",
        "Tokens/s",
        "Time (s)",
        "Load (s)",
        "Prompt tokens",
        "Context",
        "Digest",
        "Started",
    ]
    rows: list[list[str]] = []
    notes: list[str] = []
    for entry in metadata.get("runs", []):
        if isinstance(entry, dict):
            row, row_notes = summary_row(entry)
            rows.append(row)
            notes.extend(row_notes)

    lines = [
        "# Summary",
        "",
        f"Prompt: {metadata.get('prompt_preview', '')}",
        f"Prompt hash: `{metadata.get('prompt_hash', '')}`",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + " --- |" * len(headers),
    ]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    lines.extend(
        [
            "",
            "Output tokens, tokens/s and time are averages over the runs that succeeded, and",
            "count thinking as well as the answer. Load is the longest model load time, which",
            "is usually the first run. Prompt tokens against Context shows how full the",
            f"model's context window was. {MISSING} means the figure was not recorded. A system",
            "prompt is shown by its opening words; each model file has it in full. \"model's",
            "own\" is the one built into the model, used when none was given.",
            "",
        ]
    )
    for note in notes:
        lines.extend([note, ""])
    write_text_atomic(path, "\n".join(lines))


def summary_row(entry: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return the table cells for one metadata entry, plus any warnings about it."""
    files = entry.get("files") if isinstance(entry.get("files"), dict) else {}
    output_name = files.get("model_output") or ""
    results = entry.get("results") if isinstance(entry.get("results"), list) else []
    succeeded = [item for item in results if isinstance(item, dict) and item.get("ok")]

    def values(key: str) -> list[float]:
        found = [item.get(key) for item in succeeded]
        return [v for v in found if isinstance(v, (int, float)) and not isinstance(v, bool)]

    def average(key: str, digits: int = 0) -> str:
        found = values(key)
        return f"{sum(found) / len(found):,.{digits}f}" if found else MISSING

    timed = [
        item
        for item in succeeded
        if is_positive_int(item.get("eval_count")) and is_positive_int(item.get("eval_duration"))
    ]
    if timed:
        tokens = sum(item["eval_count"] for item in timed)
        seconds = sum(item["eval_duration"] for item in timed) / 1e9
        tokens_per_second = f"{tokens / seconds:,.1f}"
    else:
        tokens_per_second = MISSING

    completed = entry.get("runs_completed")
    requested = entry.get("runs_requested")
    runs = str(completed) if completed is not None else MISSING
    if is_positive_int(requested) and isinstance(completed, int) and completed < requested:
        runs = f"{completed} of {requested}"
    failed = len(results) - len(succeeded)
    if failed:
        runs += f" ({failed} failed)"

    options = entry.get("options") if isinstance(entry.get("options"), dict) else {}
    option_text = ", ".join(f"{key}={json.dumps(options[key])}" for key in sorted(options))

    system = entry.get("system")
    model_system = entry.get("model_system")
    if system:
        system_label = prompt_preview(system, SUMMARY_SYSTEM_PREVIEW_CHARS)
    elif model_system:
        system_label = "model's own: " + prompt_preview(model_system, SUMMARY_SYSTEM_PREVIEW_CHARS)
    elif model_system == "":
        system_label = "none"
    else:
        system_label = "model default"

    loads = values("load_duration")
    prompt_tokens = values("prompt_eval_count")
    context_length = entry.get("context_length")
    digest = str(entry.get("digest") or "")
    digest = digest[len("sha256:") :] if digest.startswith("sha256:") else digest
    started = str(entry.get("started_at") or "")[:16].replace("T", " ")

    row = [
        f"[{output_name}]({output_name})" if output_name else MISSING,
        str(entry.get("model") or MISSING),
        think_label(entry["think"]) if "think" in entry else MISSING,
        system_label,
        option_text or "defaults",
        runs,
        average("eval_count"),
        tokens_per_second,
        average("elapsed_seconds", 1),
        f"{max(loads) / 1e9:,.1f}" if loads else MISSING,
        f"{max(prompt_tokens):,.0f}" if prompt_tokens else MISSING,
        f"{context_length:,}" if is_positive_int(context_length) else MISSING,
        f"`{digest[:12]}`" if digest else MISSING,
        started or MISSING,
    ]

    notes: list[str] = []
    more_room = "Give the model more room with `--option num_ctx=<a larger number>`."
    size = f" ({context_length:,} tokens)" if is_positive_int(context_length) else ""
    if any(isinstance(item, dict) and is_context_error(item) for item in results):
        notes.append(
            f"**Input did not fit in `{output_name}`:** the prompt and its files were too "
            f"large for the model's context window{size}, so the run was stopped instead of "
            f"being answered from part of the input. {more_room}"
        )
    elif prompt_tokens and is_positive_int(context_length) and max(prompt_tokens) >= context_length:
        notes.append(
            f"**Input probably cut short in `{output_name}`:** the prompt filled the whole "
            f"context window{size}. {more_room}"
        )
    cut_short = sum(1 for item in succeeded if stopped_at_length(item))
    if cut_short:
        notes.append(
            f"**Answer cut short in `{output_name}`:** {cut_short} of {len(succeeded)} "
            "answers stopped at a length limit, so they are probably incomplete. Either the "
            "context window filled up or `num_predict` was reached."
        )
    return row, notes


def resolve_models(requested_models: list[str] | None, installed: list[str]) -> list[str] | None:
    if not requested_models:
        return choose_models(installed)

    selected: list[str] = []
    missing: list[str] = []
    for name in requested_models:
        if name in installed:
            selected.append(name)
        elif name.strip().lower() == ALL_MODELS:
            selected.extend(generating_models(installed))
        else:
            missing.append(name)

    if missing:
        for name in missing:
            print(f"Requested model is not installed locally: {name}", file=sys.stderr)
        print("Available local models:", file=sys.stderr)
        for model in installed:
            print(f"  - {model}", file=sys.stderr)
        return None

    return unique_in_order(selected)


def unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def generating_models(models: list[str]) -> list[str]:
    """Drop models Ollama says can't generate text, such as embedding models."""
    return [model for model in models if model_supports(model, "completion") is not False]


def models_accepting_images(models: list[str]) -> list[str]:
    accepted: list[str] = []
    for model in models:
        if model_supports(model, "vision") is False:
            print(f"Skipping {model}: it does not accept images.", file=sys.stderr)
        else:
            accepted.append(model)
    return accepted


def resolve_attachments(
    requested_files: list[Path] | None, ask: bool
) -> list[dict[str, Any]] | None:
    """Load the files to send with the prompt; ``None`` if a requested one is unusable."""
    if requested_files:
        attachments: list[dict[str, Any]] = []
        for file_path in requested_files:
            try:
                attachments.append(load_attachment(file_path, attachments))
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return None
        return attachments

    return ask_attachments() if ask else []


def ask_attachments() -> list[dict[str, Any]]:
    print()
    print("Add files to the prompt? Images and text files (code, Markdown, CSV, ...) work.")
    attachments: list[dict[str, Any]] = []
    question = "Path to a file, or press Enter to continue: "
    while True:
        try:
            answer = input(question)
        except EOFError:
            return attachments
        file_path = typed_path(answer)
        if file_path is None:
            return attachments

        try:
            attachment = load_attachment(file_path, attachments)
        except ValueError as exc:
            print(exc)
            continue
        attachments.append(attachment)
        print(f"  Added {attachment['original_name']} ({attachment_summary(attachment)})")
        question = "Path to another file, or press Enter to continue: "


def typed_path(answer: str) -> Path | None:
    """Turn a typed, pasted or dragged-in path into a ``Path``; ``None`` if blank."""
    text = answer.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    if not text:
        return None

    file_path = Path(text).expanduser()
    if not file_path.exists() and os.name != "nt" and "\\" in text:
        # Terminals escape spaces and the like when a file is dragged in.
        unescaped = Path(re.sub(r"\\(.)", r"\1", text)).expanduser()
        if unescaped.exists():
            return unescaped
    return file_path


def load_attachment(file_path: Path, earlier: list[dict[str, Any]]) -> dict[str, Any]:
    """Read one file as an image or as text. Raises ``ValueError`` if it is neither."""
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read file {file_path}: {exc}") from exc
    if not data:
        raise ValueError(f"File is empty: {file_path}")

    attachment: dict[str, Any] = {
        "original_name": file_path.name,
        "file": f"{ATTACHMENTS_DIR}/{copy_name(file_path.name, earlier)}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "data": data,
    }

    if is_image(data):
        attachment["kind"] = "image"
        return attachment

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = ""
    if not text or "\x00" in text:
        raise ValueError(
            f"Cannot add {file_path}: it is neither an image (PNG, JPEG, WebP, GIF) "
            "nor UTF-8 text."
        )
    attachment.update(kind="text", text=text)
    return attachment


def copy_name(name: str, earlier: list[dict[str, Any]]) -> str:
    """Name for a file's saved copy: its own, numbered if an earlier file has it."""
    taken = {Path(item["file"]).name.lower() for item in earlier}
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, number = name, 2
    while candidate.lower() in taken:
        candidate = f"{stem}-{number}{suffix}"
        number += 1
    return candidate


def is_image(data: bytes) -> bool:
    if data.startswith(IMAGE_SIGNATURES):
        return True
    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def attachment_summary(attachment: dict[str, Any]) -> str:
    if attachment["kind"] == "image":
        return "image"
    return f"text, {len(attachment['text']):,} characters"


def compose_prompt(prompt: str, text_files: list[dict[str, Any]] | None) -> str:
    """Return the text sent to the model: any text files, then the prompt.

    The prompt goes last because Ollama drops the start of the input when it
    doesn't fit the model's context window, and the question matters most.
    """
    if not text_files:
        return prompt

    parts = [
        f"File: {item['original_name']}\n{markdown_fence_block(item['text'].rstrip())}"
        for item in text_files
    ]
    parts.append(prompt)
    return "\n\n".join(parts)


def copy_attachments(run_dir: Path, attachments: list[dict[str, Any]]) -> None:
    """Keep a copy of the files sent with a prompt in its folder.

    They are stored once, as files, so that a large document or image is not
    repeated inside the Markdown output.
    """
    for item in attachments:
        destination = run_dir / item["file"]
        if not destination.exists():
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(item["data"])


def attachment_records(attachments: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {key: item[key] for key in ("file", "original_name", "sha256")} for item in attachments
    ]


def attachment_link(item: dict[str, Any]) -> str:
    """Markdown link to a file's saved copy."""
    label = Path(item["file"]).name.replace("[", "\\[").replace("]", "\\]")
    return f"[{label}]({urllib.parse.quote(item['file'])})"


def resolve_system(system_text: str | None, system_file: Path | None) -> str | None:
    """Return the system prompt (``""`` for the model's own); ``None`` if unreadable."""
    if system_file is None:
        return (system_text or "").strip()

    try:
        return system_file.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        print(f"Could not read system prompt file {system_file}: {exc}", file=sys.stderr)
        return None


def resolve_prompt(prompt_file: Path | None) -> str | None:
    if prompt_file is None:
        return read_multiline_prompt()

    try:
        return prompt_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"Could not read prompt file {prompt_file}: {exc}", file=sys.stderr)
        return None


def resolve_runs(requested_runs: int | None) -> int | None:
    if requested_runs is None:
        return ask_positive_int("How many times should this prompt be run? ", "--runs")

    if requested_runs > 0:
        return requested_runs

    print("--runs must be a positive whole number.", file=sys.stderr)
    return None


def resolve_temperature(requested_temperature: float | None) -> float | object | None:
    if requested_temperature is None:
        return ask_optional_temperature()

    if (
        math.isfinite(requested_temperature)
        and TEMPERATURE_MIN <= requested_temperature <= TEMPERATURE_MAX
    ):
        return requested_temperature

    print(
        "--temperature must be a finite number in the expected range "
        f"({TEMPERATURE_MIN:.1f} to {TEMPERATURE_MAX:.1f}).",
        file=sys.stderr,
    )
    return INVALID_TEMPERATURE


def resolve_think(
    requested_think: str | None, model: str, ask: bool = True
) -> bool | str | None:
    """Return the ``think`` value to send, or ``None`` to leave it to the model."""
    controls = model_thinking_controls(model)

    if requested_think is not None:
        think = parse_think_value(requested_think)
        if think is not None and controls is None and model_supports(model, "thinking") is False:
            print(f"{model} does not support thinking; --think is not sent to it.")
            return None
        return think

    if controls is None or not ask:
        return None
    return ask_optional_think(model, controls)


def parse_think_value(text: str) -> bool | str | None:
    value = text.strip()
    lowered = value.lower()
    if lowered in THINK_DEFAULT_WORDS:
        return None
    if lowered in THINK_ON_WORDS:
        return True
    if lowered in THINK_OFF_WORDS:
        return False
    return value


def think_label(value: bool | str | None) -> str:
    if value is None:
        return "model default"
    if value is True:
        return "on"
    if value is False:
        return "off"
    return str(value)


MODEL_INFO_CACHE: dict[str, dict[str, Any]] = {}


def model_info(model: str) -> dict[str, Any]:
    """Return a model's ``/api/show`` details, or ``{}`` if they can't be fetched."""
    if model not in MODEL_INFO_CACHE:
        try:
            MODEL_INFO_CACHE[model] = ollama_post_json("/api/show", {"model": model})
        except RuntimeError:
            MODEL_INFO_CACHE[model] = {}
    return MODEL_INFO_CACHE[model]


def model_system_prompt(model: str) -> str | None:
    """The system prompt built into a model: ``""`` if none, ``None`` if unknown."""
    info = model_info(model)
    if not info:
        return None
    system = info.get("system")
    return system.strip() if isinstance(system, str) else ""


def model_supports(model: str, capability: str) -> bool | None:
    """Whether Ollama lists a capability for a model; ``None`` if it doesn't say."""
    capabilities = model_info(model).get("capabilities")
    if not isinstance(capabilities, list):
        return None
    return capability in capabilities


def model_thinking_controls(model: str) -> dict[str, Any] | None:
    """Return the thinking controls Ollama reports for a model; ``None`` if none.

    Newer Ollama versions report a ``thinking`` object from ``/api/show`` with the
    supported values and the default. Older ones only list ``thinking`` among the
    model's capabilities, in which case the supported values are unknown.
    """
    thinking = model_info(model).get("thinking")
    if isinstance(thinking, dict):
        values = thinking.get("values")
        if isinstance(values, list):
            values = [value for value in values if isinstance(value, (bool, str))]
            if not values or values == [False]:
                return None
            return {"values": values, "default": thinking.get("default")}

    if model_supports(model, "thinking"):
        return {"values": None, "default": None}
    return None


def ask_optional_think(model: str, controls: dict[str, Any]) -> bool | str | None:
    values = controls.get("values")
    default = controls.get("default")
    if values:
        choices = ", ".join(think_label(value) for value in values)
    else:
        choices = "on, off, or a level such as low, medium, high"
    default_note = f" ({think_label(default)})" if isinstance(default, (bool, str)) else ""
    question = (
        f"Thinking for {model} ({choices}), "
        f"or press Enter for the model default{default_note}: "
    )

    while True:
        answer = parse_think_value(ask(question, "--think"))
        if answer is None or not values:
            return answer
        if isinstance(answer, bool):
            if answer in values:
                return answer
        else:
            for value in values:
                if isinstance(value, str) and value.lower() == answer.lower():
                    return value
        print(f"{model} supports: {choices}. Or press Enter for the model default.")


def resolve_stream(requested_stream: bool | None) -> bool:
    if requested_stream is None:
        return ask_yes_no(
            "Stream responses from /api/generate?", "--stream or --no-stream", default=False
        )
    return requested_stream


def read_multiline_prompt() -> str:
    print("Enter the prompt. Put /done on its own line when finished.")
    print()
    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip() == "/done":
                return "\n".join(lines).strip()
            lines.append(line)
    except KeyboardInterrupt:
        print()
        return ""
    except EOFError:
        return "\n".join(lines).strip()


def list_ollama_models() -> dict[str, str]:
    """Return the local models as ``{name: digest}``, sorted by name."""
    payload = ollama_get_json("/api/tags")
    digests: dict[str, str] = {}
    for item in payload.get("models", []):
        name = item.get("name")
        if name:
            digest = item.get("digest")
            digests[name] = digest if isinstance(digest, str) else ""
    return {name: digests[name] for name in sorted(digests, key=str.lower)}


def choose_models(models: list[str]) -> list[str]:
    print()
    print("Available local models:")
    for index, model in enumerate(models, start=1):
        print(f"{index}. {model}")
    print()

    question = (
        "Which model(s) should be used? "
        f"Enter one or more numbers (e.g. 1,3), or '{ALL_MODELS}': "
    )
    while True:
        answer = ask(question, "--model").strip().lower()
        if answer == ALL_MODELS:
            return generating_models(models)

        parts = [part for part in re.split(r"[\s,]+", answer) if part]
        if parts and all(part.isdigit() and 1 <= int(part) <= len(models) for part in parts):
            return unique_in_order([models[int(part) - 1] for part in parts])

        print(f"Enter model numbers from the list, or '{ALL_MODELS}'.")


def ask(question: str, flag: str) -> str:
    """Ask a question, naming the flag that answers it if no input is available."""
    try:
        return input(question)
    except EOFError:
        raise InputEnded(flag) from None


def ask_positive_int(question: str, flag: str) -> int:
    while True:
        answer = ask(question, flag).strip()
        if answer.isdigit() and int(answer) > 0:
            return int(answer)
        print("Enter a positive whole number.")


def ask_optional_float(question: str, flag: str) -> float | None:
    while True:
        answer = ask(question, flag).strip()
        if not answer:
            return None
        try:
            value = float(answer)
        except ValueError:
            print("Enter a number, or press Enter for the default.")
            continue

        if math.isfinite(value):
            return value

        print("Enter a finite number, or press Enter for the default.")


def ask_yes_no(question: str, flag: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = ask(f"{question} {suffix} ", flag).strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False

        print("Enter y or n.")


def ask_optional_temperature() -> float | None:
    question = (
        f"Temperature to use ({TEMPERATURE_MIN:.1f} to {TEMPERATURE_MAX:.1f}), "
        "or press Enter for the model's default: "
    )

    while True:
        temperature = ask_optional_float(question, "--temperature")
        if temperature is None:
            return None
        if TEMPERATURE_MIN <= temperature <= TEMPERATURE_MAX:
            return temperature

        print(
            "Temperature is outside the expected range "
            f"({TEMPERATURE_MIN:.1f} to {TEMPERATURE_MAX:.1f}). Please try again."
        )


def create_prompt_run_dir(
    prompt: str,
    images: list[dict[str, Any]] | None = None,
    text_files: list[dict[str, Any]] | None = None,
) -> Path:
    slug = prompt_slug(prompt)
    digest = prompt_hash(prompt, images, text_files)[:8]
    run_dir = OUTPUT_ROOT / f"{slug}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prompt_slug(prompt: str, max_words: int = 6) -> str:
    words = re.findall(r"[A-Za-z0-9]+", prompt.lower())
    if not words:
        return "prompt"
    return "-".join(words[:max_words])[:80].strip("-") or "prompt"


def prompt_preview(prompt: str, max_chars: int = 160) -> str:
    compact = re.sub(r"\s+", " ", prompt).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def prompt_hash(
    prompt: str,
    images: list[dict[str, Any]] | None = None,
    text_files: list[dict[str, Any]] | None = None,
) -> str:
    """Hash everything sent to the model: the text, then any images in order.

    A prompt with no files hashes to the plain SHA-256 of its text, so folders
    made before files were supported keep their names.
    """
    text = compose_prompt(prompt, text_files)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not images:
        return text_hash
    parts = [text_hash] + [image["sha256"] for image in images]
    return hashlib.sha256("\n".join(parts).encode("ascii")).hexdigest()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    return cleaned.strip("-") or "model"


def unique_model_output_path(run_dir: Path, model: str) -> Path:
    base_name = safe_name(model)
    path = run_dir / f"{base_name}.md"
    if not path.exists():
        return path

    suffix = 2
    while True:
        path = run_dir / f"{base_name}-{suffix}.md"
        if not path.exists():
            return path
        suffix += 1


def generate_once(
    model: str,
    prompt: str,
    options: dict[str, Any],
    stream: bool = False,
    think: bool | str | None = None,
    images: list[str] | None = None,
    system: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    generated_at = now_local()

    request_payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        # Have Ollama refuse input that is too long for the context window. By
        # default it silently drops part of the prompt and answers from the rest.
        "truncate": False,
    }
    if options:
        request_payload["options"] = options
    if think is not None:
        request_payload["think"] = think
    if images:
        request_payload["images"] = images
    if system:
        request_payload["system"] = system

    try:
        if stream:
            response_payload = ollama_post_stream_json(
                "/api/generate", request_payload, timeout=timeout
            )
            print()
        else:
            response_payload = ollama_post_json("/api/generate", request_payload, timeout=timeout)
        elapsed = time.monotonic() - started
        return {
            "ok": True,
            "generated_at": generated_at.isoformat(),
            "elapsed_seconds": elapsed,
            "response": response_payload.get("response") or "",
            "thinking": response_payload.get("thinking") or "",
            "raw": response_payload,
        }
    except RuntimeError as exc:
        if stream:
            print()
        elapsed = time.monotonic() - started
        return {
            "ok": False,
            "generated_at": generated_at.isoformat(),
            "elapsed_seconds": elapsed,
            "error": str(exc),
        }


def unload_model(model: str) -> None:
    """Release this model after its runs without losing results if cleanup fails."""
    try:
        response = ollama_post_json(
            "/api/generate",
            {"model": model, "keep_alive": 0, "stream": False},
            timeout=QUICK_TIMEOUT_SECONDS,
        )
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
    except RuntimeError as exc:
        print(f"Warning: could not unload {model}: {exc}", file=sys.stderr)


def ollama_get_json(path: str) -> dict[str, Any]:
    request = urllib.request.Request(f"{OLLAMA_HOST}{path}", method="GET")
    return open_json_request(request, QUICK_TIMEOUT_SECONDS)


def ollama_post_json(
    path: str,
    payload: dict[str, Any],
    timeout: float | None = QUICK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """POST to Ollama and return its JSON reply. ``timeout=None`` waits indefinitely."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_HOST}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    return open_json_request(request, timeout)


def ollama_post_stream_json(
    path: str,
    payload: dict[str, Any],
    response_output: TextIO | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_HOST}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    if response_output is None:
        response_output = sys.stdout

    dim, reset = (ANSI_DIM, ANSI_RESET) if is_terminal(response_output) else ("", "")
    response_buffer = io.StringIO()
    thinking_buffer = io.StringIO()
    showing_thinking = False
    final_chunk: dict[str, Any] | None = None
    saw_chunk = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for line_number, raw_line in enumerate(response, start=1):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid streaming JSON response on line {line_number}: {line[:200]}"
                    ) from exc

                if not isinstance(chunk, dict):
                    raise RuntimeError(f"streaming JSON line {line_number} was not an object")
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"]))

                saw_chunk = True
                # Thinking models send their reasoning in ``thinking`` chunks
                # ahead of the ``response`` chunks that carry the answer.
                thinking_text = chunk.get("thinking")
                if isinstance(thinking_text, str) and thinking_text:
                    if not showing_thinking:
                        response_output.write(f"{dim}Thinking:{reset}\n")
                        showing_thinking = True
                    thinking_buffer.write(thinking_text)
                    response_output.write(f"{dim}{thinking_text}{reset}")
                    response_output.flush()

                response_text = chunk.get("response")
                if isinstance(response_text, str) and response_text:
                    if showing_thinking:
                        response_output.write(f"\n\n{dim}Response:{reset}\n")
                        showing_thinking = False
                    response_buffer.write(response_text)
                    response_output.write(response_text)
                    response_output.flush()

                final_chunk = chunk
                if chunk.get("done"):
                    break
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(describe_request_error(exc, timeout)) from exc

    if not saw_chunk or final_chunk is None:
        raise RuntimeError("empty streaming response")
    if not final_chunk.get("done"):
        raise RuntimeError("the response stream ended before Ollama finished")

    stream_payload = dict(final_chunk)
    stream_payload["response"] = response_buffer.getvalue()
    if thinking_buffer.tell():
        stream_payload["thinking"] = thinking_buffer.getvalue()
    return stream_payload


def is_terminal(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def describe_request_error(exc: BaseException, timeout: float | None) -> str:
    """Put a failed request to Ollama into words."""
    if isinstance(exc, urllib.error.HTTPError):
        detail = exc.read().decode("utf-8", errors="replace").strip()
        try:
            # Ollama reports problems as {"error": "..."}; show just the message.
            message = json.loads(detail)["error"]
        except (ValueError, KeyError, TypeError):
            message = detail
        return f"HTTP {exc.code}: {message}"

    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    # socket.timeout only became an alias of TimeoutError in Python 3.10.
    if isinstance(reason, (TimeoutError, socket.timeout)):
        waited = f" within {timeout:g} seconds" if timeout else ""
        return f"no response from Ollama{waited}"
    if isinstance(exc, urllib.error.URLError):
        return str(reason)
    return f"the connection to Ollama failed: {exc!r}"


def open_json_request(
    request: urllib.request.Request, timeout: float | None
) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(describe_request_error(exc, timeout)) from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON response: {body[:200]}") from exc


def write_prompt_file(
    path: Path,
    prompt: str,
    started_at: dt.datetime,
    images: list[dict[str, Any]] | None = None,
    text_files: list[dict[str, Any]] | None = None,
) -> None:
    lines = [
        "# Prompt",
        "",
        f"Date: {started_at.isoformat()}",
        f"Prompt hash: `{prompt_hash(prompt, images, text_files)}`",
        "",
        markdown_fence_block(prompt, "text"),
        "",
    ]
    if text_files:
        lines.extend(
            [
                "## Text files",
                "",
                "Sent as part of the prompt text, ahead of the prompt above and in this order,",
                "each as a line reading `File: <name>` followed by its contents in a code block.",
                f"Copies are in `{ATTACHMENTS_DIR}/`.",
                "",
            ]
        )
        for item in text_files:
            copy = Path(item["file"]).name
            sent_as = "" if copy == item["original_name"] else f"sent as `{item['original_name']}`, "
            lines.append(
                f"- {attachment_link(item)} ({sent_as}{len(item['text']):,} characters, "
                f"SHA-256 `{item['sha256']}`)"
            )
        lines.append("")
    if images:
        lines.extend(["## Images", "", f"Sent as images. Copies are in `{ATTACHMENTS_DIR}/`.", ""])
        for image in images:
            lines.extend(
                [
                    f"- {attachment_link(image)} (SHA-256 `{image['sha256']}`)",
                    "",
                    f"  !{attachment_link(image)}",
                    "",
                ]
            )
    write_text_atomic(path, "\n".join(lines))


def write_model_output_file(
    path: Path,
    model: str,
    prompt: str,
    results: list[dict[str, Any]],
    options: dict[str, Any],
    stream: bool,
    started_at: dt.datetime,
    think: bool | str | None = None,
    runs_requested: int | None = None,
    digest: str = "",
    images: list[dict[str, Any]] | None = None,
    text_files: list[dict[str, Any]] | None = None,
    context_length: int | None = None,
    system: str | None = None,
    model_system: str | None = None,
) -> None:
    runs = str(len(results))
    if runs_requested is not None and len(results) < runs_requested:
        runs = f"{len(results)} of {runs_requested}"

    lines = ["# Ollama Model Test", "", f"Model: `{model}`"]
    if digest:
        lines.append(f"Model digest: `{digest}`")
    lines.extend(
        [
            f"Date: {started_at.isoformat()}",
            f"Prompt hash: `{prompt_hash(prompt, images, text_files)}`",
            f"Runs: {runs}",
            f"Options: `{json.dumps(options, sort_keys=True)}`",
            f"Thinking: `{think_label(think)}`",
            f"Streaming: `{stream}`",
        ]
    )
    if context_length:
        lines.append(f"Context window: {context_length:,} tokens")
    if text_files:
        links = ", ".join(attachment_link(item) for item in text_files)
        lines.append(f"Text files: {links} (sent ahead of the prompt)")
    if images:
        lines.append("Images: " + ", ".join(attachment_link(image) for image in images))
    lines.append("")
    if system:
        lines.extend(["## System prompt", "", markdown_fence_block(system, "text"), ""])
    elif model_system:
        lines.extend(
            [
                "## System prompt (the model's own)",
                "",
                markdown_fence_block(model_system, "text"),
                "",
            ]
        )
    lines.extend(["## Prompt", "", markdown_fence_block(prompt, "text"), ""])
    for image in images or []:
        lines.extend([f"!{attachment_link(image)}", ""])

    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"## Run {index}",
                "",
                f"Generated at: {result['generated_at']}",
                f"Elapsed seconds: {result['elapsed_seconds']:.2f}",
                f"Status: {'ok' if result['ok'] else 'error'}",
                "",
            ]
        )
        if result["ok"]:
            metadata_json = json.dumps(saved_metadata(result["raw"]), indent=2, sort_keys=True)
            thinking = result.get("thinking", "").strip()
            if thinking:
                lines.extend(["### Thinking", "", thinking, "", "### Response", ""])
            lines.extend(
                [
                    result["response"].rstrip(),
                    "",
                    "### Ollama metadata",
                    "",
                    markdown_fence_block(metadata_json, "json"),
                    "",
                ]
            )
        else:
            lines.extend(["```text", result["error"], "```", ""])

    write_text_atomic(path, "\n".join(lines))


def write_text_atomic(path: Path, text: str) -> None:
    """Write via a temporary file so a crash mid-write can't truncate ``path``."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def markdown_fence_block(value: str, language: str = "") -> str:
    longest_backtick_run = max((len(match.group(0)) for match in re.finditer(r"`+", value)), default=0)
    fence = "`" * max(3, longest_backtick_run + 1)
    suffix = language if language else ""
    return f"{fence}{suffix}\n{value}\n{fence}"


def saved_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if key not in SAVED_METADATA_OMIT_KEYS}


def save_metadata_run(
    path: Path,
    run_payload: dict[str, Any],
    prompt: str,
    prompt_file_name: str,
    images: list[dict[str, Any]] | None = None,
    text_files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add a run entry to ``metadata.json``, or update it if it is already there.

    Entries are matched on their model output file, which is unique per model per
    invocation, so saving after every run keeps refreshing the same entry.
    """
    if path.exists():
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    else:
        metadata = {}

    runs = metadata.get("runs")
    if not isinstance(runs, list):
        previous_run = legacy_metadata_as_run(metadata)
        runs = [previous_run] if previous_run else []

    output_name = run_payload["files"]["model_output"]
    for index, existing in enumerate(runs):
        files = existing.get("files") if isinstance(existing, dict) else None
        if isinstance(files, dict) and files.get("model_output") == output_name:
            runs[index] = run_payload
            break
    else:
        runs.append(run_payload)

    metadata = {
        "prompt_hash": prompt_hash(prompt, images, text_files),
        "prompt_preview": prompt_preview(prompt),
        "files": {
            "prompt": prompt_file_name,
            "summary": SUMMARY_FILE_NAME,
        },
        "runs": runs,
    }
    if images:
        metadata["images"] = attachment_records(images)
    if text_files:
        metadata["text_files"] = attachment_records(text_files)

    write_text_atomic(path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def legacy_metadata_as_run(metadata: dict[str, Any]) -> dict[str, Any] | None:
    required_keys = [
        "model",
        "runs_requested",
        "runs_completed",
        "started_at",
        "finished_at",
        "options",
    ]
    if not all(key in metadata for key in required_keys):
        return None

    files = metadata.get("files")
    model_output = files.get("model_output") if isinstance(files, dict) else None
    run_payload = {
        "model": metadata["model"],
        "runs_requested": metadata["runs_requested"],
        "runs_completed": metadata["runs_completed"],
        "started_at": metadata["started_at"],
        "finished_at": metadata["finished_at"],
        "options": metadata["options"],
        "files": {},
    }
    if model_output:
        run_payload["files"]["model_output"] = model_output
    return run_payload


def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()


if __name__ == "__main__":
    raise SystemExit(main())
