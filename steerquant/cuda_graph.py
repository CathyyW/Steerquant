"""Lazy fixed-shape CUDA Graph wrapper for DiT inference.

The wrapper keeps non-tensor arguments static and copies new tensor values into
the graph input surface on replay. WAM gets one graph per denoise-call schedule
index so each captured kernel retains the correct calibrated scale pointers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten

from cosmos_policy._src.imaginaire.utils.graph import make_graphed_callables_forward

from .wam_runtime import set_wam_timestep


def _is_tensor_tree(value: Any) -> bool:
    leaves, _ = tree_flatten(value)
    return bool(leaves) and all(isinstance(leaf, torch.Tensor) for leaf in leaves)


def _tree_signature(value: Any) -> tuple[Any, ...]:
    leaves, spec = tree_flatten(value)
    tensors = []
    for tensor in leaves:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("CUDA Graph dynamic inputs must be tensors or tensor pytrees.")
        tensors.append(
            (
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.dtype,
                tensor.device,
                bool(tensor.requires_grad),
            )
        )
    return (str(spec), *tensors)


class _TensorCallAdapter(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        positional_layout: tuple[tuple[bool, Any], ...],
        tensor_kwarg_names: tuple[str, ...],
        constant_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.module = module
        self.positional_layout = positional_layout
        self.tensor_kwarg_names = tensor_kwarg_names
        self.constant_kwargs = constant_kwargs

    def forward(self, *tensor_args: Any, **tensor_kwargs: Any) -> Any:
        restored_args = []
        tensor_index = 0
        for is_tensor, value in self.positional_layout:
            if is_tensor:
                restored_args.append(tensor_args[tensor_index])
                tensor_index += 1
            else:
                restored_args.append(value)
        restored_kwargs = dict(self.constant_kwargs)
        restored_kwargs.update({name: tensor_kwargs[name] for name in self.tensor_kwarg_names})
        return self.module(*restored_args, **restored_kwargs)


@dataclass
class _GraphRecord:
    adapter: _TensorCallAdapter
    callable: Any
    tensor_arg_indices: tuple[int, ...]
    tensor_kwarg_names: tuple[str, ...]


class CUDAGraphInferenceWrapper(nn.Module):
    """Capture each fixed DiT input signature lazily and replay it safely."""

    def __init__(self, module: nn.Module, *, warmup: int = 3) -> None:
        super().__init__()
        if warmup < 1:
            raise ValueError("CUDA Graph warmup must be positive.")
        self.module = module
        self.train(module.training)
        self.warmup = int(warmup)
        self._graphs: dict[tuple[Any, ...], _GraphRecord] = {}
        self._wam_tracker = getattr(module, "_wam_network_call_tracker", None)
        if self._wam_tracker is not None:
            # Python forward hooks do not execute during graph replay. Move the
            # schedule counter to this wrapper and capture one graph per index.
            self._wam_tracker.remove()

    @property
    def captured_graph_count(self) -> int:
        return len(self._graphs)

    def __getattr__(self, name: str) -> Any:
        """Preserve the wrapped DiT's public control surface.

        The parent model calls helpers such as ``disable_context_parallel``
        and reads ``is_context_parallel_enabled`` directly from ``net``.
        ``nn.Module`` stores child modules in ``_modules``, so use its own
        attribute resolver before forwarding unknown names to the wrapped DiT.
        """

        try:
            return super().__getattr__(name)
        except AttributeError:
            module = super().__getattr__("module")
            return getattr(module, name)

    def _select_wam_timestep(self) -> int | None:
        tracker = self._wam_tracker
        if tracker is None:
            return None
        timestep = tracker.total_calls % tracker.num_denoise_calls
        set_wam_timestep(self.module, timestep)
        tracker.current_call = timestep
        tracker.total_calls += 1
        return timestep

    def _key(self, args: tuple[Any, ...], kwargs: dict[str, Any], timestep: int | None) -> tuple[Any, ...]:
        key: list[Any] = [self.module.training, timestep]
        for value in args:
            if _is_tensor_tree(value):
                key.append(("tensor_arg", _tree_signature(value)))
            else:
                key.append(("constant_arg", type(value).__qualname__, repr(value)))
        for name, value in kwargs.items():
            if _is_tensor_tree(value):
                key.append(("tensor_kwarg", name, _tree_signature(value)))
            else:
                key.append(("constant_kwarg", name, type(value).__qualname__, repr(value)))
        return tuple(key)

    def _capture(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _GraphRecord:
        positional_layout = []
        tensor_args = []
        tensor_arg_indices = []
        for index, value in enumerate(args):
            is_tensor = _is_tensor_tree(value)
            positional_layout.append((is_tensor, None if is_tensor else value))
            if is_tensor:
                tensor_args.append(value)
                tensor_arg_indices.append(index)

        tensor_kwargs = {}
        constant_kwargs = {}
        for name, value in kwargs.items():
            if _is_tensor_tree(value):
                tensor_kwargs[name] = value
            else:
                constant_kwargs[name] = value

        adapter = _TensorCallAdapter(
            self.module,
            tuple(positional_layout),
            tuple(tensor_kwargs),
            constant_kwargs,
        )
        adapter.train(self.module.training)
        graphed = make_graphed_callables_forward(
            adapter,
            tuple(tensor_args),
            sample_kwargs=tensor_kwargs,
            num_warmup_iters=self.warmup,
        )
        return _GraphRecord(
            adapter=adapter,
            callable=graphed,
            tensor_arg_indices=tuple(tensor_arg_indices),
            tensor_kwarg_names=tuple(tensor_kwargs),
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.training or torch.is_grad_enabled() or not torch.cuda.is_available():
            return self.module(*args, **kwargs)
        tensor_leaves = []
        for value in (*args, *kwargs.values()):
            if _is_tensor_tree(value):
                leaves, _ = tree_flatten(value)
                tensor_leaves.extend(leaves)
        if not tensor_leaves or any(not tensor.is_cuda for tensor in tensor_leaves):
            return self.module(*args, **kwargs)

        timestep = self._select_wam_timestep()
        key = self._key(args, kwargs, timestep)
        record = self._graphs.get(key)
        if record is None:
            record = self._capture(args, kwargs)
            self._graphs[key] = record
        tensor_args = tuple(args[index] for index in record.tensor_arg_indices)
        tensor_kwargs = {name: kwargs[name] for name in record.tensor_kwarg_names}
        return record.callable(*tensor_args, **tensor_kwargs)


__all__ = ["CUDAGraphInferenceWrapper"]
