from typing import Union, Any, Optional
from omegaconf import DictConfig


def metrics_dict(train_metrics, valid_metrics):
    train_prefix, valid_prefix = "train_", "valid_" # мб перенести префиксы константы
    train_metrics = {train_prefix + key: value for key, value in train_metrics.items()}
    valid_metrics = {valid_prefix + key: value for key, value in valid_metrics.items()}

    return {**train_metrics, **valid_metrics}

def flatten_dict(
    params_dict: Union[dict[str, Any], DictConfig],
    parent_key: str = "",
    sep: str = "_",
    ignore_keys: Optional[Union[set[str], list]] = None,
) -> dict[str, Any]:
    """Recursively flatten a nested dictionary or DictConfig into a single-level dictionary.

    This function handles both regular dictionaries and OmegaConf DictConfig objects,
    creating keys by concatenating nested keys with the specified separator.

    Args:
        params_dict (Union[dict[str, Any], DictConfig]):
            Input dictionary or DictConfig to flatten. Can be arbitrarily nested.
        parent_key (str): Base key string used for recursion (leave empty for top-level call).
        sep (str): Separator to use between concatenated keys. Defaults to '_'.
        ignore_keys (Optional[Union[set[str], list]]):
            Keys to exclude from the output. Can be a set or list.
            Nested keys should be specified in their flattened form.

    Returns:
        A flattened dictionary where:
        - Keys are paths through the original nested structure joined by `sep`
        - Values are the leaf nodes from the original structure

    Raises:
        TypeError: If input is not a dictionary or DictConfig.
        ValueError: If separator appears in original keys (would cause ambiguity).
    """
    if ignore_keys is None:
        ignore_keys = set()
    else:
        ignore_keys = set(ignore_keys)
    items = {}
    for k, v in params_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if k in ignore_keys:
            continue

        if isinstance(v, DictConfig):
            items.update(flatten_dict(v, new_key, sep=sep, ignore_keys=ignore_keys))
        else:
            items[new_key] = v

    return items
