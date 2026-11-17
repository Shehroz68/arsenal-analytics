"""Exercise orchestration without downloading data or training models."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runner", ROOT / "run_all.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    @patch.object(runner.subprocess, "run")
    def test_all_steps_use_repo_root(self, run):
        runner.main([])
        self.assertEqual(run.call_count, len(runner.STEPS))
        for call, (_, script) in zip(run.call_args_list, runner.STEPS):
            self.assertEqual(call.args[0][1], str(ROOT / script))
            self.assertEqual(call.kwargs, {"cwd": ROOT, "check": True})

    @patch.object(runner.subprocess, "run")
    def test_skip_download(self, run):
        runner.main(["--skip-download"])
        scripts = [call.args[0][1] for call in run.call_args_list]
        self.assertEqual(len(scripts), len(runner.STEPS) - 1)
        self.assertNotIn(str(ROOT / "src/download.py"), scripts)

    @patch.object(runner.subprocess, "run")
    def test_failure_stops_pipeline(self, run):
        run.side_effect = subprocess.CalledProcessError(1, "step")
        with self.assertRaises(subprocess.CalledProcessError):
            runner.main([])
        self.assertEqual(run.call_count, 1)

    @patch.object(runner.subprocess, "run")
    def test_unknown_argument_rejected(self, run):
        with self.assertRaises(SystemExit) as error:
            runner.main(["--skip-dowload"])
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()
