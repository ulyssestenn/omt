"""Run with: python -m unittest -v test_model_unloading"""

import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import ollama_model_test as omt


class ModelUnloadingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prompt = self.root / "prompt.txt"
        self.prompt.write_text("Say hello.", encoding="utf-8")
        self.events = []
        self.unload_requests = []
        self.generation_count = 0
        self.interrupt_at = None
        self.generation_error = None
        self.unload_error = None
        self.stderr = io.StringIO()
        for target, value in (
            ("OUTPUT_ROOT", self.root / "runs"),
            ("MODEL_INFO_CACHE", {}),
        ):
            patcher = patch.object(omt, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(omt.urllib.request, "urlopen", side_effect=self.urlopen)
        patcher.start()
        self.addCleanup(patcher.stop)

    def urlopen(self, request, timeout=None):
        path = request.full_url.rsplit("/", 1)[-1]
        payload = json.loads(request.data) if request.data else {}
        if path == "tags":
            response = {"models": [{"name": "first:1"}, {"name": "second:1"}]}
        elif path == "show":
            response = {"capabilities": ["completion"], "system": ""}
        elif path == "ps":
            response = {"models": []}
        elif path == "generate":
            model = payload["model"]
            if payload.get("keep_alive") == 0:
                self.events.append(("unload", model))
                self.unload_requests.append((payload, timeout))
                if self.unload_error == "connection":
                    raise urllib.error.URLError("server unavailable")
                response = (
                    {"error": "cannot unload"}
                    if self.unload_error == "json"
                    else {"done": True, "done_reason": "unload", "response": ""}
                )
            else:
                self.events.append(("generate", model))
                self.generation_count += 1
                if self.generation_count == self.interrupt_at:
                    raise KeyboardInterrupt
                if model == "first:1" and self.generation_error:
                    raise urllib.error.HTTPError(
                        request.full_url, 400, "Bad Request", {},
                        io.BytesIO(json.dumps({"error": self.generation_error}).encode()),
                    )
                response = {"done": True, "response": "Hello!", "eval_count": 1}
        else:
            self.fail("Unexpected endpoint: " + request.full_url)
        # One JSON line is valid for both streaming and non-streaming responses.
        return io.BytesIO((json.dumps(response) + "\n").encode("utf-8"))

    def run_cli(self, stream=False):
        with redirect_stdout(io.StringIO()), redirect_stderr(self.stderr):
            return omt.main([
                "--model", "first:1", "--model", "second:1",
                "--prompt-file", str(self.prompt), "--runs", "2",
                "--temperature", "0", "--think", "default",
                "--stream" if stream else "--no-stream",
            ])

    def saved_runs(self):
        paths = list((self.root / "runs").glob("*/metadata.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text(encoding="utf-8"))["runs"]

    def test_unloads_after_repeated_runs_and_before_next_model(self):
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.events, [
            ("generate", "first:1"), ("generate", "first:1"), ("unload", "first:1"),
            ("generate", "second:1"), ("generate", "second:1"), ("unload", "second:1"),
        ])
        self.assertEqual(self.unload_requests, [
            ({"model": model, "keep_alive": 0, "stream": False}, omt.QUICK_TIMEOUT_SECONDS)
            for model in ("first:1", "second:1")
        ])
        runs = self.saved_runs()
        self.assertEqual([run["runs_completed"] for run in runs], [2, 2])
        self.assertTrue(all(len(run["results"]) == 2 for run in runs))

    def test_streaming_also_unloads_each_model_only_after_its_runs(self):
        self.assertEqual(self.run_cli(stream=True), 0)
        self.assertEqual(self.events, [
            ("generate", "first:1"), ("generate", "first:1"), ("unload", "first:1"),
            ("generate", "second:1"), ("generate", "second:1"), ("unload", "second:1"),
        ])
        self.assertEqual([run["runs_completed"] for run in self.saved_runs()], [2, 2])

    def test_context_error_unloads_before_continuing_to_next_model(self):
        self.generation_error = "input exceeds the available context window"
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.events[:3], [
            ("generate", "first:1"), ("unload", "first:1"), ("generate", "second:1"),
        ])
        runs = self.saved_runs()
        self.assertEqual([run["runs_completed"] for run in runs], [1, 2])
        self.assertFalse(runs[0]["results"][0]["ok"])

    def test_generation_errors_still_unload_after_attempted_runs(self):
        self.generation_error = "generation failed"
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.events[2:4], [("unload", "first:1"), ("generate", "second:1")])
        self.assertTrue(all(not item["ok"] for item in self.saved_runs()[0]["results"]))

    def test_interrupt_unloads_current_model_and_preserves_completed_run(self):
        self.interrupt_at = 2
        self.assertEqual(self.run_cli(), omt.EXIT_INTERRUPTED)
        self.assertEqual(self.events, [
            ("generate", "first:1"), ("generate", "first:1"), ("unload", "first:1"),
        ])
        self.assertEqual(self.saved_runs()[0]["runs_completed"], 1)

    def test_interrupt_on_first_run_also_unloads(self):
        self.interrupt_at = 1
        self.assertEqual(self.run_cli(), omt.EXIT_INTERRUPTED)
        self.assertEqual(self.events, [("generate", "first:1"), ("unload", "first:1")])

    def test_save_failure_unloads_without_masking_original_error(self):
        with patch.object(omt, "write_model_output_file", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_cli()
        self.assertEqual(self.events, [("generate", "first:1"), ("unload", "first:1")])

    def test_unload_connection_error_warns_and_preserves_results(self):
        self.unload_error = "connection"
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual([run["runs_completed"] for run in self.saved_runs()], [2, 2])
        self.assertIn("Warning: could not unload first:1: server unavailable", self.stderr.getvalue())

    def test_unload_json_error_warns_and_continues(self):
        self.unload_error = "json"
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.events[-1], ("unload", "second:1"))
        self.assertIn("Warning: could not unload first:1: cannot unload", self.stderr.getvalue())

    def test_cleanup_failure_does_not_mask_interrupt(self):
        self.interrupt_at = 2
        self.unload_error = "connection"
        self.assertEqual(self.run_cli(), omt.EXIT_INTERRUPTED)
        self.assertEqual(self.saved_runs()[0]["runs_completed"], 1)
        self.assertIn("Warning: could not unload first:1", self.stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
