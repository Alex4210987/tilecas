"""Opt-in-by-model platform documentation, not an optimization schedule."""

VERSION = 'deepseek-ascend-platform-v1'


def for_model(model):
    return HINTS if str(model).lower().startswith('deepseek-') else ''


HINTS = """
DeepSeek Ascend platform notes (deepseek-ascend-platform-v1).
These notes describe the installed A2 backend, not an algorithm or tiling choice.
HINTS.md and the experiment's installed optimization skill own the workflow.
Other skills/examples are technical references only: their package-building phases,
model names, time limits and stopping rules do not override this experiment.
Fixed High and Low remain separate fresh AKO conversations without a history handoff.

Memory scopes and compiler budgets:
- Installed src/target/codegen_ascend.cc defines A2/A3 budgets in bytes:
  UB 196352, L0C 131072, L1 524032, L0A/L0B 65536 each. These are compiler
  budgets, not universally available free space; recheck installed code if changed.
- T.alloc_ub -> shared.ub; T.alloc_L1 -> shared.l1;
  T.alloc_L0C -> wmma.accumulator. Sum simultaneously live allocations in each
  scope: product(shape) * bytes_per_element, including alignment, padding,
  pipeline buffers and compiler temporaries. FP32 takes 4 bytes, FP16 2 bytes.
  A tile fitting one scope says nothing about the other scopes or total live usage.

API and dtype details:
- Installed T.serial(start, stop=None, *, annotations=None) has no num_stages.
  T.Pipelined has num_stages; check tilelang/language/pipeline.py and
  tilelang/language/tir/ir.py (or ast/ir.py) before changing loop annotations.
- A T.copy is not an arbitrary dtype conversion. In the installed
  src/tl_templates/ascend/common.h, copy_gm_to_l1 and copy_ub_to_gm use the
  same template T for source and destination. Check the actual generated helper
  signature and supported cast path for each scope/dtype pair. Include all
  conversion and packing work in forward and in the unchanged benchmark.
- Mixed precision must use authored device conversions (for example the supported
  T.tile.cast path), not Q.half(), Q.float(), or dtype-changing Q.to(...).
  Framework casts fail the component contract and cannot be exported to Low.
- Keep host Python values and device TIR expressions distinct. Follow installed
  parser examples for buffer declarations and device scalar expressions; do not
  assume a Python helper returning a buffer is a scalar TIR expression.
- Automatic workspace insertion can change the compiled parameter list. Inspect
  the compiled adapter's params, result_idx/out_idx, workspace_idx and auto_gm_idx.
  Do not assume a negative output index still denotes the intended result tensor;
  check its position, shape and dtype against the generated launch ABI.

Reading failures:
- Memory allocation errors identify a scope and required bytes: compare the full
  live allocation set with that scope's budget before changing unrelated code.
- Auto Cube/Vector synchronization warnings about unequal loop counts can signal
  unmatched producer/consumer events. Check loop nesting, ownership and event
  counts in the generated source; successful compilation is not proof of safety.
- For an MTE write-address-out-of-range error, locate the generated copy/Fixpipe
  operation. Check buffer bounds, element-versus-byte units, strides, dtype,
  tail extents, block/subblock offsets and allocated launch arguments. This is
  an execution fault, not a numerical-tolerance failure; preserve the full error.
- Compiled source is retained in the trajectory's TileLang cache. Use installed
  API definitions to interpret it. Do not seed solutions from previous experiments.

Validation remains unchanged: solution/ModelNew.py and scripts/bench.sh own the
formal result. Temporary probes, an example's 'Test Passed', host wall timing,
or one seed are not a complete-suite pass. Keep N(0,1), FP32 input/output, seeds,
tolerances and device timing unchanged; no framework-compute fallback in solution/.
"""
