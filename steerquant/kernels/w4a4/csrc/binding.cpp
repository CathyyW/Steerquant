#include <torch/extension.h>

#include <optional>

torch::Tensor w4a4_symmetric_static_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scale,
    torch::Tensor row_gains,
    bool input_scale_is_inverse);

torch::Tensor w4a4_symmetric_static_stream_scheduled_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t schedule_index,
    bool input_scale_is_inverse);

torch::Tensor w4a4_symmetric_static_stream_scheduled_preprocess_cuda(
    torch::Tensor x,
    torch::Tensor rotation_signs,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t schedule_index,
    bool input_scale_is_inverse);

torch::Tensor w4a4_symmetric_static_stream_scheduled_adaln_cuda(
    torch::Tensor x,
    torch::Tensor adaln_scale,
    torch::Tensor adaln_shift,
    int64_t modulation_span,
    double epsilon,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t schedule_index,
    bool input_scale_is_inverse);

torch::Tensor w4a4_symmetric_static_stream_scheduled_adaln_preprocess_cuda(
    torch::Tensor x,
    torch::Tensor adaln_scale,
    torch::Tensor adaln_shift,
    int64_t modulation_span,
    double epsilon,
    torch::Tensor rotation_signs,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t schedule_index,
    bool input_scale_is_inverse);

torch::Tensor w4a4_symmetric_static_stream_scheduled_gate_residual_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t schedule_index,
    bool input_scale_is_inverse,
    torch::Tensor residual,
    torch::Tensor gate,
    int64_t gate_span);

torch::Tensor w4a4_symmetric_static_stream_scheduled_gate_residual_preprocess_cuda(
    torch::Tensor x,
    torch::Tensor rotation_signs,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t schedule_index,
    bool input_scale_is_inverse,
    torch::Tensor residual,
    torch::Tensor gate,
    int64_t gate_span);

torch::Tensor normalized_hadamard_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("normalized_hadamard", &normalized_hadamard_cuda);





  module.def("symmetric_static", &w4a4_symmetric_static_cuda, "Packed W4A4 static Linear (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled",
      &w4a4_symmetric_static_stream_scheduled_cuda,
      "Packed W4A4 static Linear with fused WAM stream and timestep mapping (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled_preprocess",
      &w4a4_symmetric_static_stream_scheduled_preprocess_cuda,
      "Packed W4A4 WAM Linear with fused Preprocess and static A4 (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled_adaln",
      &w4a4_symmetric_static_stream_scheduled_adaln_cuda,
      "Packed W4A4 WAM Linear with fused LayerNorm, AdaLN and static A4 (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled_adaln_preprocess",
      &w4a4_symmetric_static_stream_scheduled_adaln_preprocess_cuda,
      "Packed W4A4 WAM Linear with fused LayerNorm, AdaLN, Preprocess and static A4 (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled_gate_residual",
      &w4a4_symmetric_static_stream_scheduled_gate_residual_cuda,
      "Packed W4A4 WAM Linear with fused gate and residual epilogue (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled_gate_residual_preprocess",
      &w4a4_symmetric_static_stream_scheduled_gate_residual_preprocess_cuda,
      "Packed W4A4 WAM Preprocess Linear with fused gate and residual epilogue (CUDA)");
}
