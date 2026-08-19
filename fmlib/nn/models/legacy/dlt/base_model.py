import glob
import json
import os
from typing import List, Optional, Union

import torch
import torch.nn as nn


class BaseModelClass(nn.Module):
    """
    Base model class with common utilities for initialization, freezing, loading weights, and summarization.
    """

    def __init__(self):
        """
        Initialize the base model.
        """
        super().__init__()

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights of common layers: Linear, Embedding, LayerNorm.

        :param module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _freeze_body(self, ignore_layers: Optional[List[str]] = None):
        """
        Freeze all model parameters except those whose name contains substrings from `ignore_layers`.

        :param ignore_layers: List of substrings. Parameters whose names contain any of these will remain trainable.
        """
        if ignore_layers is None:
            ignore_layers = []
        for name, param in self.named_parameters():
            param.requires_grad = False
            for part in ignore_layers:
                if name.find(part) >= 0:
                    param.requires_grad = True

    @classmethod
    def load_from_pretrained(
        cls,
        path: str,
        tgt_cols: Optional[Union[int, List[str]]] = None,
        embed_mode: bool = True,
        finetune: bool = True,
    ) -> "BaseModelClass":
        """
        Load model from a pretrained directory.

        :param path: Path to the directory containing `.json` config and `.pt` weights.
        :param tgt_cols: Optional override for output layer configuration.
        :param embed_mode: If True, removes 'module.' and 'model.' prefixes from keys in state_dict.
        :return: Instantiated model with loaded weights.
        :finetune: If True, load weights, if False - only config without weights
        """
        config_model = glob.glob(os.path.join(path, "*.json"))
        assert len(config_model) == 1, f"{cls.__name__}: fail - {len(config_model)} config files found"

        with open(config_model[0], "r") as f:
            config = json.load(f)

        if tgt_cols:
            config["output_layers"] = tgt_cols

        model = cls(**config)
        if not finetune:
            return model

        weights_path = glob.glob(os.path.join(path, "*.pt"))
        assert len(weights_path) == 1, f"{cls.__name__}: fail - {len(weights_path)} weights files found"

        state_dict = torch.load(weights_path[0], map_location="cpu")

        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict,
            strict=False,
        )
        if missing_keys:
            pass
        if unexpected_keys:
            pass

        return model

    def _load_head_weights(self, path_weights: str):
        """
        Load weights for the model head (e.g., classifier layer).

        :param path_weights: Path to `.pt` file containing head weights.
        """
        _, un = self.load_state_dict(torch.load(path_weights), strict=False)
        if un:
            pass

    def _get_trainable_params(self) -> List[torch.nn.Parameter]:
        """
        Get all trainable parameters.

        :return: List of parameters with `requires_grad=True`.
        """
        train_params = []
        for x in self.parameters():
            if x.requires_grad:
                train_params.append(x)

        return train_params

    def _summary(self):
        """
        Print a formatted summary of the model structure and parameter counts.
        """

        total_params = 0
        trainable_params = 0
        non_trainable_params = 0

        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name)
            if isinstance(attr, nn.Module):
                num_params = sum(p.numel() for p in attr.parameters())
                num_trainable = sum(p.numel() for p in attr.parameters() if p.requires_grad)
                num_non_trainable = num_params - num_trainable

                total_params += num_params
                trainable_params += num_trainable
                non_trainable_params += num_non_trainable
