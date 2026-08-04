import os
import tempfile

import pytest

from fmlib.utils.import_context import import_context


def test_import_context():
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ImportError):
            from test_import import TEST_VAR  # pylint: disable=import-error
        file_path: str = os.path.join(temp_dir, "test_import.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("TEST_VAR: int = 42")
        with import_context(temp_dir):
            from test_import import TEST_VAR  # pylint: disable=import-error

            assert TEST_VAR == 42
            del TEST_VAR
