import os
import sys
import warnings
from typing import List, Optional

VENV_MARKER: str = "PYTHONPATH"
VERSION: str = f"{sys.version_info.major}.{sys.version_info.minor}"
LIB32_POSTFIX: List[str] = ["lib", f"python{VERSION}", "site-packages"]
LIB64_POSTFIX: List[str] = ["lib64", f"python{VERSION}", "site-packages"]


def venv_bootstrap(marker: str = VENV_MARKER) -> None:
    venv_path: Optional[str] = os.getenv(marker)
    if venv_path is not None:
        msg: str = f"Detected environemnt in: {venv_path=}."
        warnings.warn(msg, stacklevel=2)
        lib32_path = os.path.join(venv_path, *LIB32_POSTFIX)
        lib64_path = os.path.join(venv_path, *LIB64_POSTFIX)
        sys.path = [lib64_path, lib32_path, *(sys.path)]

        os.environ["HYDRA_FULL_ERROR"] = "1"
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT"] = "100_000_000"
        os.environ["NCCL_TIMEOUT"] = "36_000_000"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["NCCL_DEBUG"] = "INFO"
