from typing import Any, Callable, Dict, List, Union

from fmlib.constants.metadata import (
    DEFAULT_PADDING,
    PADDING_FLAG,
    SEQUENCE_LENGTH_FLAG,
    SEQUENTIAL_FLAG,
)

FieldType = Union[bool, int, float, str]
ColumnMetadata = Dict[str, FieldType]
Metadata = Dict[str, ColumnMetadata]

ColumnCheck = Callable[[ColumnMetadata], bool]
CheckColumn = Callable[[ColumnCheck], bool]
Listing = Callable[[Metadata], List[str]]


def make_column_check(flag: str) -> ColumnCheck:
    def function(column_metadata: ColumnMetadata) -> bool:
        if flag in column_metadata:
            value: Any = column_metadata[flag]
            assert isinstance(value, bool)
            return value
        return False

    return function


def make_not_check(check: ColumnCheck) -> ColumnCheck:
    def function(column_metadata: ColumnCheck) -> bool:
        return not check(column_metadata)

    return function


def all_column_checks(*checks: ColumnCheck) -> ColumnCheck:
    def function(column_metadata: ColumnMetadata) -> bool:
        def perform_check(check):
            return check(column_metadata)

        return all(map(perform_check, checks))

    return function


def any_column_checks(*checks: ColumnCheck) -> ColumnCheck:
    def function(column_metadata: ColumnMetadata) -> bool:
        def perform_check(check):
            return check(column_metadata)

        return any(map(perform_check, checks))

    return function


is_sequential: ColumnCheck = all_column_checks(make_column_check(SEQUENTIAL_FLAG))
is_not_sequential: ColumnCheck = all_column_checks(make_not_check(is_sequential))


def make_listing(check: ColumnCheck) -> Listing:
    def function(metadata: Metadata) -> List[str]:
        result: List[str] = []
        for col_name, col_meta in metadata.items():
            if check(col_meta):
                result.append(col_name)
        return sorted(result)

    return function


list_sequential: Listing = make_listing(is_sequential)
list_not_sequential: Listing = make_listing(is_not_sequential)


def get_padding(metadata: Metadata, column_name: str) -> Any:
    if column_name not in metadata:
        msg: str = f"Column {column_name} not found in metadata."
        raise KeyError(msg)
    return metadata[column_name].get(PADDING_FLAG, DEFAULT_PADDING)


def get_sequence_length(metadata: Metadata, column_name: str) -> int:
    if column_name not in metadata:
        msg: str = f"Column {column_name} not found in metadata."
        raise KeyError(msg)
    if not is_sequential(metadata[column_name]):
        msg: str = f"Column {column_name} is not sequential."
        raise ValueError(msg)
    if SEQUENCE_LENGTH_FLAG not in metadata[column_name]:
        msg: str = f"Column {column_name} is sequential but does not have a sequence length."
        raise KeyError(msg)
    result: Any = metadata[column_name][SEQUENCE_LENGTH_FLAG]
    if not isinstance(result, int):
        msg: str = f"Sequence length for column {column_name} is not an integer."
        raise TypeError(msg)
    if result < 1:
        msg: str = f"Sequence length for column {column_name} is not a positive integer."
        raise ValueError(msg)
    return result
