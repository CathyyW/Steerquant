"""Load a calibrated checkpoint and install its inference modules."""

from .backends import get_backend
from .wam_pipeline import WAMCalibrationCheckpoint, WAMNetworkCallTracker, replace_linears_with_wam


def quantize_model_for_inference(model, runtime):
    if not getattr(runtime, "quantization_enabled", False):
        return model
    checkpoint = WAMCalibrationCheckpoint.load(runtime.quantization_calibration_path)
    requested = (runtime.quantization_weight_bits, runtime.quantization_activation_bits)
    if requested != (checkpoint.weight_bits, checkpoint.activation_bits):
        raise ValueError(f"Requested W{requested[0]}A{requested[1]} does not match checkpoint bitwidths")
    target = getattr(model, "net", model)
    replace_linears_with_wam(
        target, checkpoint, backend=get_backend(getattr(runtime, "quantization_backend", "reference"))
    )
    tracker = WAMNetworkCallTracker(target, checkpoint.num_denoise_calls)
    tracker.register()
    target._wam_network_call_tracker = tracker
    if getattr(runtime, "dit_cuda_graph", False):
        from .cuda_graph import CUDAGraphInferenceWrapper

        model.net = CUDAGraphInferenceWrapper(target)
    return model
