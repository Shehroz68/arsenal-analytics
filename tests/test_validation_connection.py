"""Validation must never create an empty database or alter source data."""
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch
import types

SCRIPT = Path(__file__).resolve().parents[1] / "src/validate.py"
spec = importlib.util.spec_from_file_location("validation", SCRIPT)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class ConnectionTests(unittest.TestCase):
    def test_missing_database_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.db"
            with self.assertRaises(sqlite3.OperationalError):
                validation.connect_existing(path)
            self.assertFalse(path.exists())

    def test_existing_database_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data #1.db"
            with sqlite3.connect(path) as con:
                con.execute("CREATE TABLE matches (result TEXT)")
                con.execute("INSERT INTO matches VALUES ('H')")
            con = validation.connect_existing(path)
            try:
                self.assertEqual(con.execute("SELECT result FROM matches").fetchone(), ("H",))
                with self.assertRaises(sqlite3.OperationalError):
                    con.execute("DELETE FROM matches")
            finally:
                con.close()


class ValidationRunnerTests(unittest.TestCase):
    def test_connection_closed_on_success_and_query_failure(self):
        for failure in (None, RuntimeError("query failed")):
            con = Mock()
            read_sql = Mock(side_effect=failure)
            pandas = types.SimpleNamespace(read_sql=read_sql)
            with self.subTest(failure=failure):
                with patch.dict("sys.modules", {"pandas": pandas}):
                    with patch.object(validation, "connect_existing", return_value=con) as connect:
                        if failure:
                            with self.assertRaises(RuntimeError):
                                validation.main()
                        else:
                            validation.main()
                        connect.assert_called_once_with(SCRIPT.parents[1] / "data/processed/epl.db")
                con.close.assert_called_once_with()
