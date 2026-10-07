# Activate the desired Python environment before sourcing this file.
# Wheel-based SDK installations should set ROCM_PATH explicitly.
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export ROCM_HOME="${ROCM_HOME:-$ROCM_PATH}"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
export PATH="$ROCM_PATH/bin:$PATH"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1201}"
export TILELANG_HIP_SAVE_TEMP_FILES="${TILELANG_HIP_SAVE_TEMP_FILES:-0}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"

# The devel wheel supplies the unversioned profiler library links.
export ROCPROF_LIST_AVAIL_TOOL_LIBRARY="${ROCPROF_LIST_AVAIL_TOOL_LIBRARY:-$ROCM_PATH/lib/rocprofiler-sdk/librocprofv3-list-avail.so}"
