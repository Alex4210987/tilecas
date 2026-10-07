#pragma once

#include <cstddef>
#include <cstdint>

#define SHTL_ASCEND_ABI_VERSION 1
#define SHTL_ASCEND_MAX_RANK 8

enum ShtlAscendDType : std::int32_t {
    SHTL_ASCEND_BOOL = 0,
    SHTL_ASCEND_INT8 = 1,
    SHTL_ASCEND_INT32 = 2,
    SHTL_ASCEND_INT64 = 3,
    SHTL_ASCEND_FLOAT16 = 4,
    SHTL_ASCEND_BFLOAT16 = 5,
    SHTL_ASCEND_FLOAT32 = 6,
};

struct ShtlAscendTensorView {
    void* data;
    std::int32_t dtype;
    std::int32_t rank;
    std::int64_t sizes[SHTL_ASCEND_MAX_RANK];
    std::int64_t strides[SHTL_ASCEND_MAX_RANK];
};

// Candidate-owned host symbols. The harness owns tensor validation, allocation,
// stream selection, core-count discovery, correctness, timing, and registration.
extern "C" std::int32_t shtl_ascend_abi_version();
extern "C" std::int32_t shtl_ascend_launch(
    const ShtlAscendTensorView* inputs,
    std::size_t input_count,
    ShtlAscendTensorView* outputs,
    std::size_t output_count,
    std::uint32_t aic_core_count,
    void* stream);
