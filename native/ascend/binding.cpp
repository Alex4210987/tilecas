#include <tiling/platform/platform_ascendc.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <vector>

#include "shtl_ascend_abi.h"

namespace {

std::int32_t DTypeCode(c10::ScalarType dtype) {
    switch (dtype) {
        case c10::ScalarType::Bool:
            return SHTL_ASCEND_BOOL;
        case c10::ScalarType::Char:
            return SHTL_ASCEND_INT8;
        case c10::ScalarType::Int:
            return SHTL_ASCEND_INT32;
        case c10::ScalarType::Long:
            return SHTL_ASCEND_INT64;
        case c10::ScalarType::Half:
            return SHTL_ASCEND_FLOAT16;
        case c10::ScalarType::BFloat16:
            return SHTL_ASCEND_BFLOAT16;
        case c10::ScalarType::Float:
            return SHTL_ASCEND_FLOAT32;
        default:
            TORCH_CHECK(false, "unsupported Ascend Native ABI dtype: ", dtype);
    }
}

ShtlAscendTensorView MakeView(const at::Tensor& tensor, int device) {
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        "Ascend Native ABI accepts only NPU tensors");
    TORCH_CHECK(tensor.get_device() == device, "all ABI tensors must share one NPU device");
    TORCH_CHECK(tensor.dim() <= SHTL_ASCEND_MAX_RANK, "tensor rank exceeds ABI limit");
    ShtlAscendTensorView result{};
    result.data = const_cast<void*>(tensor.data_ptr());
    result.dtype = DTypeCode(tensor.scalar_type());
    result.rank = static_cast<std::int32_t>(tensor.dim());
    for (std::int64_t index = 0; index < tensor.dim(); ++index) {
        result.sizes[index] = tensor.size(index);
        result.strides[index] = tensor.stride(index);
    }
    return result;
}

std::vector<at::Tensor> Launch(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor> outputs) {
    TORCH_CHECK(!inputs.empty(), "Ascend Native ABI requires at least one input");
    TORCH_CHECK(!outputs.empty(), "Ascend Native ABI requires preallocated outputs");
    TORCH_CHECK(
        shtl_ascend_abi_version() == SHTL_ASCEND_ABI_VERSION,
        "candidate Ascend Native ABI version mismatch");
    TORCH_CHECK(
        inputs.front().device().type() == c10::DeviceType::PrivateUse1,
        "first input is not an NPU tensor");
    const int device = inputs.front().get_device();

    std::vector<ShtlAscendTensorView> input_views;
    std::vector<ShtlAscendTensorView> output_views;
    input_views.reserve(inputs.size());
    output_views.reserve(outputs.size());
    for (const at::Tensor& tensor : inputs) {
        input_views.push_back(MakeView(tensor, device));
    }
    for (const at::Tensor& tensor : outputs) {
        output_views.push_back(MakeView(tensor, device));
    }
    const std::uint32_t core_count =
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(core_count > 0, "Ascend platform reported zero AIC cores");
    void* stream = c10_npu::getCurrentNPUStream(device).stream(false);
    const std::int32_t status = shtl_ascend_launch(
        input_views.data(), input_views.size(), output_views.data(),
        output_views.size(), core_count, stream);
    TORCH_CHECK(status == 0, "candidate shtl_ascend_launch failed with status ", status);
    return outputs;
}

}  // namespace

TORCH_LIBRARY(shtl_ascend, module) {
    module.def("launch(Tensor[] inputs, Tensor[] outputs) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(shtl_ascend, PrivateUse1, module) {
    module.impl("launch", &Launch);
}
