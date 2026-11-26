"""Reuse one sampling graph for action-map projections.

Preserves final-layer selective checkpointing across repeated backward passes.
"""

from contextlib import contextmanager
import gc
import time

import torch


class ReusableRecomputeContext:
    """Restore selective-checkpoint cache lists and policy counters per backward."""

    def __init__(self, context):
        if not hasattr(context, "storage") or not hasattr(context, "policy_fn"):
            raise RuntimeError("Unsupported selective-checkpoint context for map graph reuse")
        self.context = context
        self.saved_storage = None
        self.saved_counters = None

    def __enter__(self):
        if self.saved_storage is None:
            self.saved_storage = {key: list(value) for key, value in self.context.storage.items()}
            closure = self.context.policy_fn.__closure__ or ()
            self.saved_counters = [
                (cell.cell_contents, dict(cell.cell_contents))
                for cell in closure
                if isinstance(cell.cell_contents, dict)
            ]
        self.context.storage = {key: list(value) for key, value in self.saved_storage.items()}
        for counter, saved in self.saved_counters:
            counter.clear()
            counter.update(saved)
        return self.context.__enter__()

    def __exit__(self, *exc):
        return self.context.__exit__(*exc)


def _load_transformer_engine():
    import transformer_engine
    from transformer_engine.pytorch.ops import fuser
    from transformer_engine.pytorch.ops.basic import rmsnorm

    def major_minor(version):
        return tuple(int(part) for part in str(version).split("+")[0].split(".")[:2])

    if major_minor(torch.__version__) != (2, 7) or major_minor(transformer_engine.__version__) != (2, 2):
        raise RuntimeError(
            "Retained CPU-saved map graphs require the pinned PyTorch 2.7 / Transformer Engine 2.2 stack; "
            "install the pinned dependencies for CPU-saved map graphs"
        )
    function = getattr(fuser, "_OperationFuserAutogradFunction", None)
    if function is None or not hasattr(rmsnorm, "clear_tensor_data"):
        raise RuntimeError("Unsupported Transformer Engine backward internals for map graph reuse")
    return function, rmsnorm


@contextmanager
def retained_backward_state(net, *, enabled):
    """Scope the version-specific TE/checkpoint repair to one retained graph.

    Denoiser checkpoint mode builds fresh inner autograd contexts during each
    reverse pass and does not need these patches. CPU-saved mode traverses the
    same TE contexts repeatedly and needs the compatibility repair. The latter is intended for a single calibration thread/process.
    """
    if not enabled:
        yield
        return
    function, rmsnorm = _load_transformer_engine()
    final = getattr(net, "final_layer", None)
    checkpoint_fn = getattr(final, "checkpoint_fn", None)
    keywords = getattr(checkpoint_fn, "keywords", None)
    if (
        getattr(final, "_checkpoint_wrapped_module", None) is None
        or not isinstance(keywords, dict)
        or not callable(keywords.get("context_fn"))
    ):
        raise RuntimeError("Map graph reuse requires the pinned final-layer selective checkpoint wrapper")
    original_backward = function.backward
    original_clear = rmsnorm.clear_tensor_data
    original_context_fn = keywords["context_fn"]

    def preserving_backward(ctx, *args):
        if not hasattr(ctx, "basic_op_ctxs"):
            raise RuntimeError(
                "Transformer Engine backward context changed; rerun map compatibility validation"
            )
        snapshots = [dict(op.__dict__) for op in ctx.basic_op_ctxs]
        try:
            return original_backward(ctx, *args)
        finally:
            for op, snapshot in zip(ctx.basic_op_ctxs, snapshots):
                op.__dict__.clear()
                op.__dict__.update(snapshot)

    def context_fn():
        forward, recompute = original_context_fn()
        return forward, ReusableRecomputeContext(recompute)

    try:
        function.backward = staticmethod(preserving_backward)
        rmsnorm.clear_tensor_data = lambda *args, **kwargs: None
        keywords["context_fn"] = context_fn
        yield
    finally:
        keywords["context_fn"] = original_context_fn
        rmsnorm.clear_tensor_data = original_clear
        function.backward = staticmethod(original_backward)


def run_action_probes(build_forward, collector, generator, *, on_probe=None):
    """Accumulate Rademacher scores from one retained sampling graph.

    `build_forward` returns standardized actions with an attached autograd graph.
    Reuse changes graph lifetime only. Denoiser recomputation inside backward is
    still expected when the low-memory checkpoint mode is selected.
    """
    if collector.num_action_probes < 1 or not collector.reuse_forward:
        raise ValueError("Action probes require a retained graph and a positive projection count")
    action = None
    sampling_forwards = 0
    try:
        for probe in range(collector.num_action_probes):
            started = time.monotonic()
            if probe == 0:
                collector.begin_pass()
                action = build_forward()
                sampling_forwards += 1
                collector.validate_capture_count()
            signs = torch.randint(0, 2, action.shape, generator=generator, device=action.device).float()
            signs.mul_(2).sub_(1)
            collector.set_probe(probe)
            last_probe = probe + 1 == collector.num_action_probes
            (action * signs).sum().backward(retain_graph=not last_probe)
            collector.validate_backward_count()
            cuda = action.is_cuda
            if last_probe:
                action = None
            del signs
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
            if on_probe is not None:
                on_probe(probe, time.monotonic() - started)
        return sampling_forwards
    except BaseException:
        # Do not retain graph/capture tensors after an interrupted observation.
        collector.begin_pass()
        raise
    finally:
        action = None
