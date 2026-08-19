import atexit
import signal
from typing import Any

import torch
from accelerate import Accelerator


def register_cleanup(accelerator: Accelerator) -> None:
    def _cleanup() -> None:
        try:
            accelerator.end_training()
        finally:
            pass

        try:
            accelerator.free_memory()
        finally:
            pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            finally:
                pass

    atexit.register(_cleanup)

    def _handler(signum: Any, frame: Any) -> None:
        accelerator.set_trigger()

    signal.signal(signal.SIGINT, _handler)


def check_exit(accelerator: Accelerator) -> None:
    msg: str = "Training process was forcefully stopped."
    if accelerator.check_trigger():
        raise KeyboardInterrupt(msg)
