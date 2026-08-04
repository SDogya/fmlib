from typing import Callable, Dict, Union

import torch

Batch = Dict[str, torch.Tensor]

GeneralValue = Union[torch.Tensor, "GeneralBatch"]
GeneralBatch = Dict[str, GeneralValue]
GeneralCollateFn = Callable[[GeneralBatch], GeneralBatch]
