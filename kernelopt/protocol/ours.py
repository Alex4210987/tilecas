"""Launch one Ours conversation with cross-level tools and no optimization deadline."""

import fcntl
import json
from pathlib import Path
import shutil
import time

from execution_platform import configure, native_language
from fixed_cascade import launch, prepare, run_window, latest_private_session, validated_final
from ours_observe import level_of

def install_cross_level_tools(api, workspace):
    """Install export, offline diagnosis and the backend timing parser."""
    import os
    from execution_platform import exporter, is_ascend
    source = Path(exporter(api))
    shutil.copyfile(source, workspace / "scripts/export.py")
    dependencies = ("native_ascend_runtime.py",) if is_ascend(api) else ("module_loader.py", "native_runtime.py")
    for name in dependencies:
        beside = source.with_name(name)
        if beside.is_file():
            shutil.copyfile(beside, workspace / "scripts" / name)
    if os.environ.get("KERNELBENCH_NO_DIAGNOSIS") == "1":
        return
    for name, module in (("diagnose.py", "ours_diagnose.py"), ("attribution.py", "attribution.py")):
        shutil.copyfile(Path(__file__).with_name(module), workspace / "scripts" / name)
    # The per-task parser lives with the backend; diagnose.py reads it under
    # one name so the script itself stays backend-agnostic. Reaching for a
    # controller module instead is how this tool silently became unusable.
    for name in (("ascend_evaluation_stages.py",) if is_ascend(api) else ("cuda_evaluation_stages.py",)):
        source = Path(__file__).with_name(name)
        if source.is_file():
            shutil.copyfile(source, workspace / "scripts/evaluation_stages.py")
            break
    else:
        raise RuntimeError("No evaluation-stage parser to install beside diagnose.py")


def run_ours(api, run_id, task, resume=False):
    configure(api)
    root = api.RUN_ROOT / run_id
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = root / "reference.py"
    workspace = root / "sessions/1"
    resume_session = None
    if resume:
        try:
            state = json.loads((root / "state.json").read_text())
            if (state.get("run_id") != run_id or state.get("mode") != "ours"
                    or state.get("task") != task or state.get("agent_model") != api.MODEL
                    or state.get("device") != api.DEVICE
                    or state.get("status") not in {"retryable", "incomplete"}):
                raise ValueError("Resume requires this same interrupted Ours experiment")
            if api.ACTIVE_WALLTIME_SECONDS is not None and time.time() >= state["started_at_epoch"] + api.ACTIVE_WALLTIME_SECONDS:
                raise TimeoutError("Original Ours run envelope has expired")
            if not reference.is_file() or not (workspace / "solution").is_dir():
                raise ValueError("Ours resume requires its existing workspace and reference")
            resume_session = latest_private_session(workspace)
            recorded = state.get("agent_sessions", {}).get("1") or state.get("agent_session_id")
            if recorded and recorded != resume_session:
                raise ValueError("Recorded Ours conversation does not match its private transcript")
            state.update(status="running", phase="resuming", resumed_at=api.now())
        except Exception:
            lock.close()
            raise
    else:
        if reference.exists() or (root / "sessions").exists() or (root / "solution").exists():
            lock.close()
            raise ValueError("Ours run already initialized; a fresh run id is required")
        shutil.copyfile(api.resolve_reference(task), reference)
        state = dict(run_id=run_id, task=task, mode="ours", status="running", stage="High",
                     backend="tilelang", device=api.DEVICE, agent_model=api.MODEL, high_only=False,
                     high_iterations=0, low_iterations=0, run_scope="single_agent_cross_level",
                     started_at=api.now(), started_at_epoch=time.time(), current={}, best={})
    api.publish(root / "state.json", state)
    try:
        # This arm owns both levels, so its evaluator must follow solution/
        # rather than the language the workspace happened to start in.
        if resume:
            phase = "High" if level_of(workspace / "solution") == "High" else "Low"
            launch(api, root, workspace, phase, state, resume_session=resume_session)
        else:
            prepare(api, workspace, reference, "tilelang", detect_level=True,
                    deadline=None if api.ACTIVE_WALLTIME_SECONDS is None else state["started_at_epoch"] + api.ACTIVE_WALLTIME_SECONDS)
            with (workspace / "HINTS.md").open("a") as hints:
                hints.write(run_window(api, state["started_at_epoch"]))
            install_cross_level_tools(api, workspace)
            if list((workspace / "solution").iterdir()):
                raise RuntimeError("Ours workspace must start with an empty solution")
            launch(api, root, workspace, "High", state)

        metrics = validated_final(workspace)
        level = level_of(workspace / "solution")
        state.update(stage="High" if level == "High" else "Low",
                     backend="tilelang" if level == "High" else native_language(api),
                     final_level=level, final_level_source="solution/ contents at the agent's last edit",
                     status="complete", phase="complete", final=metrics, current=metrics)
    except Exception as error:
        state.update(status="exhausted" if isinstance(error, TimeoutError) else "incomplete",
                     phase="failed", reason=str(error))
    finally:
        api.publish(root / "state.json", state)
        lock.close()
    return 0 if state["status"] == "complete" else 2
