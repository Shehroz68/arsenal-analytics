"""Download failures must stop the pipeline before stale data is used."""
from pathlib import Path
import runpy
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "src/download.py"


class DownloadTests(unittest.TestCase):
    def run_download(self, response=None, error=None):
        fake = types.SimpleNamespace(RequestException=RuntimeError)
        from unittest.mock import Mock
        fake.get = Mock(return_value=response, side_effect=error)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("sys.modules", {"requests": fake}):
                with patch("pathlib.Path", wraps=Path) as paths:
                    paths.return_value = Path(directory)
                    runpy.run_path(str(SCRIPT), run_name="__main__")
            self.assertEqual(len(list(Path(directory).glob("*.csv"))), 26)

    def test_success(self):
        self.run_download(types.SimpleNamespace(ok=True, content=b"a" * 1001))

    def test_http_failure(self):
        with self.assertRaises(SystemExit) as error:
            self.run_download(types.SimpleNamespace(ok=False, status_code=503))
        self.assertEqual(error.exception.code, 1)

    def test_short_response(self):
        with self.assertRaises(SystemExit):
            self.run_download(types.SimpleNamespace(ok=True, content=b"short", status_code=200))

    def test_network_failure(self):
        with self.assertRaises(SystemExit):
            self.run_download(error=RuntimeError("offline"))
