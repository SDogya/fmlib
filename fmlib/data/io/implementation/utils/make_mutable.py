import numpy as np

WRITEABLE_FLAG: str = "WRITEABLE"


def copy_if_immutable(array: np.array) -> np.array:
    if not array.flags[WRITEABLE_FLAG]:
        result = array.copy()
        assert result.flags[WRITEABLE_FLAG]
        return result
    return array
