from typing import Any, Dict, List

from fmlib.constants.metadata import SEQUENTIAL_FLAG

from ..metadata import (
    list_not_sequential,
    list_sequential,
)

TEST_METADATA: Dict[str, Dict[str, Any]] = {
    "ft_0": {},
    "seq_0": {SEQUENTIAL_FLAG: True},
}


def test_sequential():
    result: List[str] = list_sequential(TEST_METADATA)
    assert result == ["seq_0"]


def test_not_sequential():
    result: List[str] = list_not_sequential(TEST_METADATA)
    assert result == ["ft_0"]
