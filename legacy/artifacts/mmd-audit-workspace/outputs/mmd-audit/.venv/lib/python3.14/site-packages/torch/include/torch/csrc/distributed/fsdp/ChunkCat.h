#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/Export.h>

namespace torch::distributed::fsdp {

// Composite kernel of fsdp::chunk_cat_mixed_dtype: calls _chunk_cat.out, first
// casting each input to out's dtype if the input dtypes differ.
TORCH_API void chunk_cat_mixed_dtype(
    at::TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    at::Tensor& out);

} // namespace torch::distributed::fsdp

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
