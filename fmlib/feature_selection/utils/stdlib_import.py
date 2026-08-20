"""Helpers to import stdlib modules shadowed by this package directory.

Pytest invoked from ``fmlib/feature_selection`` puts that directory on
``sys.path``. The local ``statistics`` package then wins over stdlib
``statistics``, and third-party code such as seaborn (via BorutaShap)
fails with ``cannot import name 'NormalDist'``.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _is_package_root(item: str) -> bool:
    try:
        return Path(item).resolve() == _PACKAGE_ROOT
    except OSError:
        return False


def _is_local_shadow(module: object, name: str) -> bool:
    file_name = (getattr(module, "__file__", "") or "").replace("\\", "/")
    marker = f"feature_selection/{name}"
    return marker in file_name


@contextmanager
def stdlib_module(name: str) -> Iterator[None]:
    """Bind ``sys.modules[name]`` to the stdlib module for the duration."""
    saved = {
        key: sys.modules[key]
        for key in list(sys.modules)
        if key == name or key.startswith(f"{name}.")
    }
    local = saved.get(name)
    if local is not None and not _is_local_shadow(local, name):
        yield
        return

    kept_path = [item for item in sys.path if not _is_package_root(item)]
    for key in saved:
        sys.modules.pop(key, None)
    original_path = sys.path[:]
    sys.path[:] = kept_path
    try:
        importlib.import_module(name)
        yield
    finally:
        sys.path[:] = original_path
        if saved:
            sys.modules.update(saved)
        else:
            current = sys.modules.get(name)
            if current is not None and not _is_local_shadow(current, name):
                sys.modules.pop(name, None)
                for key in list(sys.modules):
                    if not key.startswith(f"{name}."):
                        continue
                    nested = sys.modules[key]
                    if not _is_local_shadow(nested, name):
                        sys.modules.pop(key, None)
