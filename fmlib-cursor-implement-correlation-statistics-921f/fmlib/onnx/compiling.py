import abc
import warnings
from typing import Any, Dict, List, Mapping, Optional, Self, Tuple, cast

import torch
import torch.onnx

from fmlib.utils.mode_context import mode_context
from fmlib.utils.named_adapters import InputAdapter, OutputAdapter

TensorAxis = Mapping[int, str]
DynamicAxes = Mapping[str, TensorAxis]


def _get_onnx_opset() -> int:
    return 20


def _make_dynamic_shapes(names: List[str]) -> DynamicAxes:
    return {n: {0: "batch_size"} for n in sorted(names)}


def _get_input_names(model: Any, input_names: Optional[List[str]] = None) -> List[str]:
    if input_names is None:
        assert hasattr(model, "get_input_names")
        input_names = model.get_input_names()
    return list(input_names)


def _get_output_names(model: Any, output_names: Optional[List[str]] = None) -> List[str]:
    if output_names is None:
        assert hasattr(model, "get_output_names")
        output_names = model.get_output_names()
    return list(output_names)


def _get_dynamic_shapes(model: Any, names: List[str], dynamic_shapes: Optional[DynamicAxes] = None) -> List[str]:
    if dynamic_shapes is not None:
        raw: List[str] = dynamic_shapes
    elif hasattr(model, "get_dynamic_shapes"):
        raw: List[str] = model.get_dynamic_shapes()
    else:
        warnings.warn("Default dynamic axes are used.", stacklevel=2)
        raw: List[str] = _make_dynamic_shapes(names)
    triggered: bool = False
    result: DynamicAxes = {}
    for name in names:
        if name in raw:
            result[name] = raw[name]
        else:
            warnings.warn(f'Warning: "{name}" is not in dynamic shapes.', stacklevel=2)
            triggered = True
    if triggered:
        warnings.warn(f"Fixed list of dynamic shapes: {result}.", stacklevel=2)
    return result


def compile_model(
    model: torch.nn.Module,
    args: Tuple[Any, ...] = (),
    kwargs: Dict[str, Any] | None = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_shapes: Optional[DynamicAxes] = None,
    onnx_export_kwargs: Dict[str, Any] | None = None,
) -> torch.onnx.ONNXProgram:
    if onnx_export_kwargs is None:
        onnx_export_kwargs = {}
    if kwargs is None:
        kwargs = {}
    input_names: List[str] = _get_input_names(model, input_names)
    output_names: List[str] = _get_output_names(model, output_names)
    dynamic_shapes: List[str] = _get_dynamic_shapes(model, input_names, dynamic_shapes)
    with mode_context(model, training=False), torch.no_grad():
        final_kwargs: Dict[str, Any] = dict(
            model=model,
            args=args,
            kwargs=kwargs,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=_get_onnx_opset(),
            **onnx_export_kwargs,
        )

        result: torch.onnx.ONNXProgram = torch.onnx.export(
            dynamo=True,
            verify=True,
            optimize=False,
            do_constant_folding=True,
            **final_kwargs,
        )

    result.optimize()

    return result


class BaseModel(abc.ABC):
    def __init__(
        self: Self,
        input_names: List[str],
        output_names: List[str],
    ) -> None:
        self.input_adapter = InputAdapter(input_names)
        self.output_adapter = OutputAdapter(output_names)

    @abc.abstractmethod
    def get_model(self: Self) -> Optional[torch.nn.Module]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_compiled(self: Self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.onnx.ONNXProgram:
        raise NotImplementedError()

    def convert_outputs(self: Self, *args: torch.Tensor) -> Dict[str, torch.Tensor]:
        return dict(zip(self.output_adapter.get_output_names(), list(args), strict=False))

    def __call__(self: Self, **kwargs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        model: torch.onnx.ONNXProgram = self.get_compiled(**kwargs)
        raw: Tuple[torch.Tensor, ...] = model(**kwargs)
        return self.convert_outputs(*raw)


class CompiledModel(BaseModel):
    def __init__(
        self: Self,
        compiled: torch.onnx.ONNXProgram,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> None:
        self.compiled: torch.onnx.ONNXProgram = compiled

        if input_names is None:
            inputs: List[Any] = compiled.model.graph.inputs
            input_names = [inp.name for inp in inputs]
        input_names: List[str] = cast(List[str], input_names)

        if output_names is None:
            outputs: List[Any] = compiled.model.graph.outputs
            output_names = [out.name for out in outputs]
        output_names: List[str] = cast(List[str], output_names)

        super().__init__(
            input_names=input_names,
            output_names=output_names,
        )

    def get_model(self: Self) -> None:
        return None

    def get_compiled(self: Self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.onnx.ONNXProgram:
        return self.compiled


class LazyCompiledModel(BaseModel):
    def __init__(
        self: Self,
        model: torch.nn.Module,
        compiled: Optional[torch.onnx.ONNXProgram] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_shapes: Optional[DynamicAxes] = None,
        onnx_export_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        if onnx_export_kwargs is None:
            onnx_export_kwargs = {}
        self.model: torch.nn.Module = model
        self.compiled: Optional[torch.onnx.ONNXProgram] = compiled

        self.dynamic_shapes: Optional[DynamicAxes] = dynamic_shapes
        self.onnx_export_kwargs: Dict[str, Any] = onnx_export_kwargs

        super().__init__(
            input_names=_get_input_names(model, input_names),
            output_names=_get_output_names(model, output_names),
        )

    def get_model(self: Self) -> torch.nn.Module:
        return self.model

    def get_compiled(self: Self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.onnx.ONNXProgram:
        if self.compiled is not None:
            return self.compiled

        result: torch.onnx.ONNXProgram = compile_model(
            model=self.model,
            args=args,
            kwargs=kwargs,
            dynamic_shapes=self.dynamic_shapes,
            input_names=self.input_adapter.get_input_names(),
            output_names=self.output_adapter.get_output_names(),
            onnx_export_kwargs=self.onnx_export_kwargs,
        )

        self.compiled = result
        return self.compiled
