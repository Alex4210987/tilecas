"""Independent AKO4ALL CUDA control arm, initialized without a High candidate."""

import fcntl
from pathlib import Path
import shutil
import time

from execution_platform import native_language, configure
from fixed_cascade import copy_source, launch, prepare, validated_final


def run_native_only(api, run_id, task, resume=False):
    if resume:
        raise ValueError("Native-only starts from an empty CUDA solution; resume is not supported")

    configure(api)
    root = api.RUN_ROOT / run_id
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = root / "reference.py"
    if reference.exists() or (root / "sessions").exists() or (root / "solution").exists():
        lock.close()
        raise ValueError("Native-only run already initialized; a fresh run id is required")
    shutil.copyfile(api.resolve_reference(task), reference)

    seed_value = getattr(api, "NATIVE_SEED", None)
    seed = Path(seed_value) if seed_value else None
    if seed is not None:
        if not seed.is_dir() or not any(seed.iterdir()):
            raise ValueError(f"Native continuation seed is missing or empty: {seed}")
        if (seed.parent / "reference.py").read_text() != reference.read_text():
            raise ValueError("Native continuation reference differs from the selected task")

    state = dict(run_id=run_id, task=task, mode="native-only", status="running", stage="Low",
                 backend=native_language(api), device=api.DEVICE, agent_model=api.MODEL, high_only=False,
                 high_iterations=0, low_iterations=0, run_scope="native_only_from_empty_" + native_language(api),
                 started_at=api.now(), started_at_epoch=time.time(), current={}, best={})
    if seed is not None:
        state.update(run_scope="native_only_continuation_from_validated_native",
                     source_run=getattr(api, "NATIVE_SOURCE_RUN", None),
                     initial_metrics=getattr(api, "NATIVE_INITIAL_METRICS", None))
    api.publish(root / "state.json", state)
    try:
        workspace = root / "sessions/2"
        prepare(api, workspace, reference, native_language(api))
        language_label = "CUDA" if native_language(api) == "cuda" else "AscendC"
        if seed is None:
            # prepare() described optimizing a supplied implementation; this arm
            # starts from nothing, which is the whole of what makes it a control.
            with (workspace / "HINTS.md").open("a") as hints:
                hints.write(f"\nThis workspace starts from an empty solution/: create the initial "
                            f"{native_language(api)} implementation yourself, then run the AKO loop on it.\n")
            if list((workspace / "solution").iterdir()):
                raise RuntimeError("Native-only workspace must start with an empty solution")
        else:
            (workspace / "solution").rmdir()
            copy_source(seed, workspace / "solution")
            with (workspace / "HINTS.md").open("a") as hints:
                hints.write("\nThis is a fresh optimizer conversation continuing from the supplied "
                            "validated Native solution. Benchmark it first, preserve it as the fallback, "
                            "and improve it under the same correctness and timing protocol.\n")

        launch(api, root, workspace, "Low", state)

        metrics = validated_final(workspace)
        state.update(status="complete", phase="complete", final=metrics, current=metrics)
    except Exception as error:
        state.update(status="exhausted" if isinstance(error, TimeoutError) else "incomplete",
                     phase="failed", reason=str(error))
    finally:
        api.publish(root / "state.json", state)
        lock.close()
    return 0 if state["status"] == "complete" else 2
