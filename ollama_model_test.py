#!/usr/bin/env python3
"""Run the same prompt against a local Ollama model and save the outputs."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


OLLAMA_HOST = "http://localhost:11434"
OUTPUT_ROOT = Path("ollama-runs")


def main() -> int:
    print("Ollama model tester")
    print("===================")
    print()

    try:
        models = list_ollama_models()
    except RuntimeError as exc:
        print(f"Could not list Ollama models: {exc}", file=sys.stderr)
        print("Make sure Ollama is running, then try again.", file=sys.stderr)
        return 1

    if not models:
        print("No local Ollama models found. Install one with `ollama pull <model>`.")
        return 1

    model = choose_model(models)

    prompt = read_multiline_prompt()
    if not prompt.strip():
        print("No prompt entered. Exiting.")
        return 1

    runs = ask_positive_int("How many times should this prompt be run? ")

    temperature = ask_optional_float(
        "Temperature to use, or press Enter for Ollama default: "
    )
    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature

    run_dir = create_prompt_run_dir(prompt)
    prompt_path = run_dir / "prompt.md"
    metadata_path = run_dir / "metadata.json"
    output_path = unique_model_output_path(run_dir, model)

    started_at = now_local()
    if not prompt_path.exists():
        write_prompt_file(prompt_path, prompt, started_at)

    print()
    print(f"Saving results to: {run_dir}")
    print()

    results: list[dict[str, Any]] = []
    for index in range(1, runs + 1):
        print(f"Run {index}/{runs} using {model}...")
        result = generate_once(model, prompt, options)
        results.append(result)
        status = "ok" if result["ok"] else "error"
        elapsed = result["elapsed_seconds"]
        print(f"  {status} in {elapsed:.2f}s")

    finished_at = now_local()
    write_model_output_file(output_path, model, prompt, results, options, started_at)
    append_metadata_run(
        metadata_path,
        {
            "model": model,
            "runs_requested": runs,
            "runs_completed": len(results),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "options": options,
            "files": {
                "model_output": output_path.name,
            },
        },
        prompt,
        prompt_path.name,
    )

    print()
    print(f"Done. Wrote {output_path}")
    return 0


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


def list_ollama_models() -> list[str]:
    payload = ollama_get_json("/api/tags")
    models = payload.get("models", [])
    names = [item.get("name") for item in models if item.get("name")]
    return sorted(names, key=str.lower)


def choose_model(models: list[str]) -> str:
    print()
    print("Available local models:")
    for index, model in enumerate(models, start=1):
        print(f"{index}. {model}")
    print()

    while True:
        answer = input("Which model should be used? Enter a number: ").strip()
        if not answer.isdigit():
            print("Enter a model number from the list.")
            continue

        choice = int(answer)
        if 1 <= choice <= len(models):
            return models[choice - 1]

        print("Enter a model number from the list.")


def ask_positive_int(question: str) -> int:
    while True:
        answer = input(question).strip()
        if answer.isdigit() and int(answer) > 0:
            return int(answer)
        print("Enter a positive whole number.")


def ask_optional_float(question: str) -> float | None:
    while True:
        answer = input(question).strip()
        if not answer:
            return None
        try:
            return float(answer)
        except ValueError:
            print("Enter a number, or press Enter for the default.")


def create_prompt_run_dir(prompt: str) -> Path:
    slug = prompt_slug(prompt)
    digest = prompt_hash(prompt)[:8]
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


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


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


def generate_once(model: str, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    generated_at = now_local()

    request_payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        request_payload["options"] = options

    try:
        response_payload = ollama_post_json("/api/generate", request_payload)
        elapsed = time.monotonic() - started
        return {
            "ok": True,
            "generated_at": generated_at.isoformat(),
            "elapsed_seconds": elapsed,
            "response": response_payload.get("response", ""),
            "raw": response_payload,
        }
    except RuntimeError as exc:
        elapsed = time.monotonic() - started
        return {
            "ok": False,
            "generated_at": generated_at.isoformat(),
            "elapsed_seconds": elapsed,
            "error": str(exc),
        }


def ollama_get_json(path: str) -> dict[str, Any]:
    request = urllib.request.Request(f"{OLLAMA_HOST}{path}", method="GET")
    return open_json_request(request)


def ollama_post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_HOST}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    return open_json_request(request)


def open_json_request(request: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc
    except TimeoutError as exc:
        raise RuntimeError("request timed out") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON response: {body[:200]}") from exc


def write_prompt_file(path: Path, prompt: str, started_at: dt.datetime) -> None:
    path.write_text(
        "\n".join(
            [
                "# Prompt",
                "",
                f"Date: {started_at.isoformat()}",
                f"Prompt hash: `{prompt_hash(prompt)}`",
                "",
                markdown_fence_block(prompt, "text"),
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_model_output_file(
    path: Path,
    model: str,
    prompt: str,
    results: list[dict[str, Any]],
    options: dict[str, Any],
    started_at: dt.datetime,
) -> None:
    lines = [
        "# Ollama Model Test",
        "",
        f"Model: `{model}`",
        f"Date: {started_at.isoformat()}",
        f"Prompt hash: `{prompt_hash(prompt)}`",
        f"Runs: {len(results)}",
        f"Options: `{json.dumps(options, sort_keys=True)}`",
        "",
        "## Prompt",
        "",
        markdown_fence_block(prompt, "text"),
        "",
    ]

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
            lines.extend([result["response"].rstrip(), ""])
        else:
            lines.extend(["```text", result["error"], "```", ""])

    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_fence_block(value: str, language: str = "") -> str:
    longest_backtick_run = max((len(match.group(0)) for match in re.finditer(r"`+", value)), default=0)
    fence = "`" * max(3, longest_backtick_run + 1)
    suffix = language if language else ""
    return f"{fence}{suffix}\n{value}\n{fence}"


def append_metadata_run(
    path: Path,
    run_payload: dict[str, Any],
    prompt: str,
    prompt_file_name: str,
) -> None:
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

    metadata = {
        "prompt_hash": prompt_hash(prompt),
        "prompt_preview": prompt_preview(prompt),
        "files": {
            "prompt": prompt_file_name,
        },
        "runs": runs,
    }
    metadata["runs"].append(run_payload)

    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
