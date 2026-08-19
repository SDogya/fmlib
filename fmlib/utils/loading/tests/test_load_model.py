import os
import tempfile
from typing import Dict, List, Optional, Self, Tuple

import pytest
import safetensors.torch as st
import torch

from fmlib.data.utils.random.random_context import DEFAULT_SEED, SeededRandomContext
from fmlib.utils.loading.load_model import get_mapping, instantiate_model, load_model

CUSTOM_MODULE_CODE: str = """
import torch

class CustomTestModule{postfix}(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = torch.nn.Linear(size, size)

    def forward(self, x):
        return self.linear(x)
"""

CUSTOM_CONFIG_CODE: str = """
model:
    _target_: custom_model{postfix}.CustomTestModule{postfix}
    size: 32
weights_path: "custom_weights{postfix}.{format}"
"""

HYBRID_CONFIG_CODE: str = """
model:
    _target_: fmlib.utils.loading.tests.test_load_model.HybridTestModule
    impl:
        _target_: custom_model{postfix}.CustomTestModule{postfix}
        size: 32
weights_path: "hybrid_weights{postfix}.{format}"
"""

SEEDS: List[int] = [1, 42, DEFAULT_SEED]
FORMATS: List[str] = ["pt", "safetensors"]
POSTFIXES: List[str] = [f"P{i}" for i in range(3)]


class HybridTestModule(torch.nn.Module):
    def __init__(self: Self, impl: torch.nn.Module):
        super().__init__()

        self.impl: torch.nn.Module = impl

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)


def save_model(model: torch.nn.Module, path: str, mode: str):
    assert mode in FORMATS

    if mode == "pt":
        with open(path, "wb") as f:
            torch.save(model.state_dict(), f)

    if mode == "safetensors":
        st.save_file(model.state_dict(), path)


@pytest.mark.parametrize("postfix", POSTFIXES)
def test_custom_instantiate(postfix: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_name: str = f"custom_model{postfix}.py"
        model_path: str = os.path.join(tmpdir, custom_name)
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(CUSTOM_MODULE_CODE.format(postfix=postfix))
        config_path: str = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                CUSTOM_CONFIG_CODE.format(
                    postfix=postfix,
                    format="pt",
                ),
            )

        with SeededRandomContext(DEFAULT_SEED):
            first: torch.nn.Module = instantiate_model(tmpdir)
        assert isinstance(first, torch.nn.Module)
        assert isinstance(first.linear, torch.nn.Linear)
        assert first.__class__.__name__ == f"CustomTestModule{postfix}"

        with SeededRandomContext(DEFAULT_SEED):
            second: torch.nn.Module = instantiate_model(tmpdir)
        assert id(first) != id(second)
        assert id(first.linear.bias) != id(second.linear.bias)
        assert id(first.linear.weight) != id(second.linear.weight)
        assert second.__class__.__name__ == f"CustomTestModule{postfix}"
        assert torch.allclose(first.linear.bias, second.linear.bias)
        assert torch.allclose(first.linear.weight, second.linear.weight)


@pytest.mark.parametrize("postfix", POSTFIXES)
def test_hybrid_instantiate(postfix: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_name: str = f"custom_model{postfix}.py"
        model_path: str = os.path.join(tmpdir, custom_name)
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(CUSTOM_MODULE_CODE.format(postfix=postfix))
        config_path: str = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                HYBRID_CONFIG_CODE.format(
                    postfix=postfix,
                    format="pt",
                ),
            )

        with SeededRandomContext(DEFAULT_SEED):
            first: torch.nn.Module = instantiate_model(tmpdir)
        assert isinstance(first, torch.nn.Module)
        assert isinstance(first.impl.linear, torch.nn.Linear)
        assert first.__class__.__name__ == "HybridTestModule"
        assert first.impl.__class__.__name__ == f"CustomTestModule{postfix}"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", FORMATS)
@pytest.mark.parametrize("postfix", POSTFIXES)
def test_custom_load(seed: int, mode: str, postfix: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_name: str = f"custom_model{postfix}.py"
        model_path: str = os.path.join(tmpdir, custom_name)
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(CUSTOM_MODULE_CODE.format(postfix=postfix))
        config_path: str = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                CUSTOM_CONFIG_CODE.format(
                    postfix=postfix,
                    format=mode,
                ),
            )
        with SeededRandomContext(seed):
            model: torch.nn.Module = instantiate_model(tmpdir)
            weights_name: str = f"custom_weights{postfix}.{mode}"
            weights_path: str = os.path.join(tmpdir, weights_name)
            save_model(model, weights_path, mode)

        loaded: torch.nn.Module
        if mode == "safetensors":
            loaded = load_model(tmpdir)
        else:
            with pytest.warns(UserWarning):
                loaded = load_model(tmpdir)

        assert isinstance(loaded, torch.nn.Module)
        assert loaded.__class__.__name__ == f"CustomTestModule{postfix}"
        assert torch.allclose(model.linear.bias, loaded.linear.bias)
        assert torch.allclose(model.linear.weight, loaded.linear.weight)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", FORMATS)
@pytest.mark.parametrize("postfix", POSTFIXES)
def test_hybrid_load(seed: int, mode: str, postfix: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_name: str = f"custom_model{postfix}.py"
        model_path: str = os.path.join(tmpdir, custom_name)
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(CUSTOM_MODULE_CODE.format(postfix=postfix))
        config_path: str = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                HYBRID_CONFIG_CODE.format(
                    postfix=postfix,
                    format=mode,
                ),
            )
        with SeededRandomContext(seed):
            model: torch.nn.Module = instantiate_model(tmpdir)
            weights_name: str = f"hybrid_weights{postfix}.{mode}"
            weights_path: str = os.path.join(tmpdir, weights_name)
            save_model(model, weights_path, mode)

        loaded: torch.nn.Module
        if mode == "safetensors":
            loaded = load_model(tmpdir)
        else:
            with pytest.warns(UserWarning):
                loaded = load_model(tmpdir)

        assert isinstance(loaded, torch.nn.Module)
        assert isinstance(loaded, HybridTestModule)
        assert torch.allclose(model.impl.linear.bias, loaded.impl.linear.bias)
        assert torch.allclose(model.impl.linear.weight, loaded.impl.linear.weight)


TEST_MODEL_KEYS: List[Tuple[str, Optional[str]]] = [
    ("target.weight", "weight"),
    ("model.target.weight", "weight"),
    ("model.sub_model.target.weight", "weight"),
    ("model.target.sub_model.weight", "sub_model.weight"),
]


def test_get_mapping():
    pattern: str = r"^.*?target\.(.*)$"
    keys: List[str] = sorted([k for k, _ in TEST_MODEL_KEYS])
    result: List[Tuple[str, str]] = get_mapping(keys, pattern)

    gtr_dict: Dict[str, Optional[str]] = dict(TEST_MODEL_KEYS)
    result_dict: Dict[str, str] = dict(result)

    for key, rename in gtr_dict.items():
        assert rename == result_dict.get(key)
