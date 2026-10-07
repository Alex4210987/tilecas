"""Fixed orchestration; each AKO invocation sees only an independent workspace."""
import codecs
import json
import fcntl
import os
import re
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time
import agent_driver
from validate_candidate import LIBRARY_RULE
from execution_platform import is_ascend, native_language, exporter, evaluator, configure, references, library_rule


VIEW = Path("/tmp/workspace")
OPTIMIZATION_HINT = """
Apply these rules in both High and Native. Test one performance hypothesis at a time;
state the bottleneck, the changed mechanism, and the measured result in ITERATIONS.md.
Compilation/correctness failures and their repairs belong to the original hypothesis,
not additional exhausted optimization directions. Only a correct measured candidate
can support a performance conclusion. Keep the best valid source while repairing.
Low occupancy alone does not prove a bottleneck; high throughput alone does not prove
an optimum. Use existing per-kernel timings to locate dominant cost, then collect only
the counters needed to distinguish the next concrete hypothesis. Do not profile every edit.
Before ending a stage/run, briefly identify the dominant remaining cost, which distinct
hypotheses were actually tested, and the best untested next step. A failed implementation
does not refute its optimization idea, and a regression refutes only the tested variant.
Prefer a bounded discriminating edit/profile when a substantial bottleneck is unresolved;
stop when remaining concrete opportunities lack justified value.
No mandatory iteration quota or hardware-floor proof; do not invent available speedup.
Use the existing log and source directly; no extra planning files, agents or approval steps.
Do not calculate hashes/checksums or modify evaluation scripts; report infrastructure issues.
"""
REFERENCES = ("/data1/xiaoxinyu/undergraduate_paper/tilelang (including docs); "
              "/data1/xiaoxinyu/tilelang-cuda-skills/skills/tilelang; "
              "/usr/local/cuda-12.9/include (low-level CUDA/runtime interfaces only); "
              "/usr/local/cuda-12.9/extras/CUPTI/samples; "
              "/usr/local/cuda-12.9/nsight-compute-2025.2.1/extras/samples; "
              "/usr/local/cuda-12.9/compute-sanitizer/docs")


def profiling_hint(api):
    return ("" if is_ascend(api) else
            "\nFor baseline ncu and later diagnostic profiling, use `bash scripts/ncu.sh` "
            "(optional `--metrics metric1,metric2`); it collects counters with kernel replay. "
            "Use scripts/bench.sh for correctness and performance ranking. "
            "NCU calls share a host lock and retry temporary driver-resource contention. "
            "A resource-busy failure is infrastructure, not evidence against profiling or a kernel; "
            "after it is resolved, targeted profiling remains available in this session. "
            "FP32 correctness uses the KernelBench default atol=rtol=1e-4.\n")


def run_window(api, started):
    """Compatibility hook: do not inject clock-based stopping guidance."""
    return ""


def language_rule(api, language, cross_level=False):
    if cross_level:
        return (
            f"High: TileLang, no torch imports. Low: {native_language(api)}, "
            "host allocation and launch bindings allowed. No framework computation fallback.")
    return ("Write the solution in TileLang; no torch imports or framework tensor "
            "computation in solution/. A plain ModelNew may implement to(...) returning "
            "self and __call__ delegating to forward."
            if language == "tilelang" else
            f"Write the solution in {native_language(api)}. Tensor allocation and launch "
            "bindings are allowed; no framework tensor computation fallback.")


def prepare(api, workspace, reference, language, detect_level=False, deadline=None):
    workspace.mkdir(parents=True)
    shutil.copyfile(reference, workspace / "reference.py")
    (workspace / "solution").mkdir()
    rule = language_rule(api, language, cross_level=detect_level)
    (workspace / "HINTS.md").write_text(
        rule + (" Target ascendc, platform A2.\n" if is_ascend(api) else "\n")
        + (library_rule(api).replace("One optimizer conversation, no auxiliary agents.\n", "")
           if detect_level else library_rule(api))
        + ("\nUse reference.py inputs and the unchanged scripts/bench.sh suite for correctness "
           "and performance ranking. Diagnostic profiling is allowed.\n" if detect_level else
           "\nInputs come from reference.py. `scripts/bench.sh` is the only measurement "
          "that counts; its fixed seed suite is identical on every call, so do not edit "
          "it and do not measure any other way.\n")
        + "Read-only references: " + references(api) + "\n")
    api.install_server_bench_wrapper(workspace, VIEW / "reference.py", language,
                                     detect_level=detect_level, deadline=deadline)
    if is_ascend(api) and language != "tilelang":
        shutil.copyfile(Path(__file__).with_name("native_ascend_runtime.py"),
                        workspace / "scripts/native_ascend_runtime.py")
    with (workspace / "HINTS.md").open("a") as hints:
        hints.write(profiling_hint(api))
        if not is_ascend(api) and not detect_level:
            hints.write(OPTIMIZATION_HINT)
        hints.write(f"Reference: {api.REFERENCE_MS} ms; SPEEDUP is quoted against it.\n")
        if not detect_level and language != "tilelang" and getattr(api, "EXTRA_OPTIMIZATION_HINT", ""):
            hints.write("\n" + api.EXTRA_OPTIMIZATION_HINT.strip() + "\n")


def observed_phase(api, directory):
    """Follow the candidate, not the phase in which the conversation started."""
    if is_ascend(api):
        # Match Ascend's benchmark wrapper. Compiler output lives below the
        # trajectory root and must not relabel an authored High snapshot.
        return "Low" if any(p.is_file() and p.suffix in {".cpp", ".cu", ".h"}
                            for p in directory.iterdir()) else "High"
    from ours_observe import level_of
    return "High" if level_of(directory) == "High" else "Low"


def refresh_ours_stage(api, workspace, state):
    if state.get("mode") == "ours":
        phase = observed_phase(api, workspace / "solution")
        state.update(stage=phase, backend="tilelang" if phase == "High" else native_language(api))


def mirror(api, root, workspace, phase, *, cross_level=False):
    for output in workspace.glob("trajectory/*/output.txt"):
        name = output.parent.name
        stamp, label = name.rsplit("_", 1)
        raw = output.read_text(errors="replace")
        observed = re.search(r"^STAGE: (High|Low)$", raw, re.M)
        measured_phase = (observed.group(1) if observed else
                          observed_phase(api, output.parent) if cross_level else phase)
        destination = root / "trajectory" / f"{stamp}_{measured_phase}_{label}"
        destination.mkdir(parents=True, exist_ok=True)
        text = raw if observed else f"STAGE: {measured_phase}\n" + raw
        target = destination / "output.txt"
        if not target.exists() or target.read_text() != text:
            target.write_text(text)
    entries = []
    for number, title in ((1, "High"), (2, "Low")):
        path = root / "sessions" / str(number) / "ITERATIONS.md"
        if path.exists():
            if cross_level:
                title = "Ours — one conversation across High and Low"
            entries.append(f"# {title}\n\n" + path.read_text())
    (root / "ITERATIONS.md").write_text("\n\n".join(entries))


def latest_private_session(workspace):
    for path in sorted((workspace / agent_driver.PRIVATE).glob("**/*.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        with path.open() as stream:
            try:
                first = json.loads(stream.readline())
            except ValueError:
                continue
        if first.get("type") == "session_meta" and first.get("payload", {}).get("id"):
            return first["payload"]["id"]
    raise RuntimeError("Fixed High optimizer did not leave a resumable private session")


def first_correct(workspace):
    """Has any benchmark in this workspace ever reported a correct candidate?"""
    for output in Path(workspace).glob("trajectory/*/output.txt"):
        try:
            if "CORRECT: True" in output.read_text(errors="replace"):
                return True
        except OSError:
            continue
    return False


def launch(api, root, workspace, phase, state, *, resume_session=None, prompt_override=None):
    prompt = (f"Continue this same optimization run in {VIEW}. Re-read HINTS.md and {api.AKO_SKILL}; "
              "the controller has restored this existing workspace after an infrastructure interruption. "
              "Keep this run's existing source and history and follow the current language contract."
              if resume_session else
              f"Read and follow {api.AKO_SKILL}; complete the AKO4ALL run in {VIEW}.")
    env = {key: value for key, value in os.environ.items() if not key.startswith("KERNELBENCH_")}
    env["PATH"] = f"{Path(api.PYTHON).parent}:{env.get('PATH', '')}"
    env["TORCH_EXTENSIONS_DIR"] = str(VIEW / ".cache" / "torch_extensions")
    # Each call owns a fresh conversation. Capacity retries within that call
    # resume it; Fixed's Low call never receives High's conversation id.
    session_id = resume_session or agent_driver.new_session_id()
    resume = resume_session
    if state.get("mode") == "ours":
        prompt = f"{'Continue following' if resume_session else 'Follow'} {api.AKO_SKILL} in {VIEW}."
    else:
        prompt += " Work alone. Do not spawn/delegate agents or calculate hashes/checksums."
    if prompt_override is not None:
        prompt = prompt_override
    isolation = dict(workspace_view=VIEW, private_history=True, readonly_workspace_paths=("scripts",))
    for attempt in range(1, api.CAPACITY_RETRIES + 2):
        command = agent_driver.build(api.MODEL, prompt, VIEW, single_agent=True,
                                     json_output=True, resume=resume, session_id=session_id)
        command = api.isolated_agent_command(workspace, command, **isolation)
        state.update(wall_envelope_seconds=api.ACTIVE_WALLTIME_SECONDS,
                     first_correct_seconds=api.FIRST_CORRECT_SECONDS, stop_policy="agent_decides",
                     stage=phase, phase="agent", agent_attempt=attempt,
                     agent_driver=agent_driver.NAME, reasoning_effort=agent_driver.EFFORT)
        api.publish(root / "state.json", state)
        with (root / "agent-output.txt").open("a") as log, (root / "agent-events.jsonl").open("a") as raw:
            log.write(f"\n[{api.now()}] {phase} agent · model={api.MODEL} · attempt={attempt}\n")
            log.flush()
            proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # Both halves of the first-correct judgement are per phase. A
            # cascade's second agent opens a fresh workspace long after the run
            # began, so timing it from the run's start condemned it before its
            # first bench could exist.
            phase_started = time.time()
            poller = selectors.DefaultSelector()
            poller.register(proc.stdout, selectors.EVENT_READ)
            chunks = []
            # A streamed driver emits one JSON event per line; render whole
            # lines so a split read never corrupts the readable transcript.
            stream, pending = agent_driver.Stream(), ""
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

            def record(text):
                if not text:
                    return
                raw.write(text)
                raw.flush()
                rendered = "".join(agent_driver.render(line + "\n", stream)
                                   for line in text.splitlines())
                if stream.session_id:
                    state["agent_session_id"] = stream.session_id
                    state.setdefault("agent_sessions", {})[workspace.name] = stream.session_id
                log.write(rendered)
                log.flush()
                chunks.append(rendered)
                state["agent_output_tail"] = "".join(chunks)[-4000:]

            while poller.get_map():
                if api.ACTIVE_WALLTIME_SECONDS is not None and time.time() - state["started_at_epoch"] >= api.ACTIVE_WALLTIME_SECONDS:
                    proc.terminate()
                    raise TimeoutError("run envelope exhausted")
                # A run with nothing correct after the first window is stuck,
                # not slow, and will not become comparable by being given the
                # rest of the envelope.
                if (api.FIRST_CORRECT_SECONDS is not None and time.time() - phase_started >= api.FIRST_CORRECT_SECONDS
                        and not state.get("first_correct_at") and not first_correct(workspace)):
                    proc.terminate()
                    raise TimeoutError("no correct candidate within the first-correct window")
                for key, _ in poller.select(timeout=2):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        poller.unregister(key.fileobj)
                        continue
                    pending += decoder.decode(chunk)
                    complete, _, pending = pending.rpartition("\n")
                    # The rendered log is for reading; this is the record. Any
                    # event the renderer does not know about is still kept.
                    if complete:
                        record(complete + "\n")
                refresh_ours_stage(api, workspace, state)
                mirror(api, root, workspace, phase, cross_level=state.get("mode") == "ours")
                api.publish(root / "state.json", state)
            pending += decoder.decode(b"", final=True)
            if pending:
                record(pending + "\n")
            poller.close()
            code = proc.wait()
            proc.stdout.close()
        session_id = stream.session_id or session_id
        if session_id:
            state["agent_session_id"] = session_id
            state.setdefault("agent_sessions", {})[workspace.name] = session_id
            api.publish(root / "state.json", state)
        if code == 0:
            git = subprocess.run(["git", "-C", str(workspace), "log", "--format=%H %aI %s"],
                                 capture_output=True, text=True)
            (workspace / "git-log.txt").write_text(git.stdout)
            return
        if not agent_driver.retryable("".join(chunks)) or attempt > api.CAPACITY_RETRIES:
            raise RuntimeError(f"{phase} agent exited with {code}: {''.join(chunks)[-1500:]}")
        resume = session_id
        time.sleep(api.CAPACITY_BACKOFF)


def require_cuda_contract(api, source, backend):
    if is_ascend(api):
        return
    from validate_candidate import validate
    source = Path(source)
    errors = validate(source, 'High' if backend == 'tilelang' else 'Native', source.parent)
    if errors:
        raise ValueError('Candidate contract rejected: ' + '; '.join(errors))


def evaluate(api, root, source, backend, output):
    # Native-only calls this helper without entering run_fixed, which sets up
    # the controller's metrics import path for the other two modes.
    require_cuda_contract(api, source, backend)
    sys.path.insert(0, str(api.ROOT / "scripts"))
    from kernelbench_metrics import parse_benchmark
    command = [api.PYTHON, str(evaluator(api)),
               "--ref", str(root / "reference.py"), "--solution", str(source), "--backend", backend]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=api.BENCH_TIMEOUT_SECONDS)
    output.write_text(proc.stdout)
    result = parse_benchmark(proc.stdout)
    if proc.returncode or result.get("correctness") is not True:
        raise RuntimeError(f"replay failed: {output}")
    return result


def selected_evaluation(api, root, source, backend, output):
    """Reuse the agent's measured selection; final labeler remains independent."""
    require_cuda_contract(api, source, backend)
    sys.path.insert(0, str(api.ROOT / "scripts"))
    from benchmark_evidence import reuse_cuda_suite
    return reuse_cuda_suite(api, root, source.parent, output) or evaluate(api, root, source, backend, output)


def copy_source(source, target):
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", ".cache", "__pycache__", "torch_extensions", "*.so", "*.o"))


def validated_final(workspace):
    """Require recorded full-suite final validation of the source left at HEAD."""
    from kernelbench_metrics import parse_benchmark
    from ours_observe import source_files
    outputs = sorted(workspace.glob("trajectory/*_final/output.txt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not outputs:
        raise ValueError("Agent exited without a final full-suite benchmark")
    output = outputs[0]
    text = output.read_text(errors="replace")
    metrics = parse_benchmark(text)
    if not (metrics.get("compiled") and metrics.get("correctness")
            and metrics.get("candidate_median_ms", 0) > 0):
        raise ValueError("Final benchmark failed or has no valid latency")
    current = source_files(workspace / "solution")
    frozen = source_files(output.parent)
    if not current or any(frozen.get(name) != value for name, value in current.items()):
        raise ValueError("Final benchmark source differs from the current candidate")
    suffixes = {".py", ".cpp", ".cu", ".cc", ".cxx", ".h", ".hpp", ".cuh"}
    if ({name for name in frozen if "/" not in name and Path(name).suffix in suffixes}
            != {name for name in current if "/" not in name and Path(name).suffix in suffixes}):
        raise ValueError("Final benchmark contains a different source file set")
    rows = []
    for line in text.splitlines():
        if line.startswith(("MSPROF_DATA: ", "ROCM_DATA: ")):
            rows.append(json.loads(line.split(": ", 1)[1]))
    if (workspace / "scripts/ascend_bench.py").exists() and "=== Fixed-seed aggregate ===" not in text:
        raise ValueError("Final Ascend benchmark has no full-suite aggregate")
    if (workspace / "scripts/ascend_bench.py").exists() or (workspace / "scripts/rocm_bench.py").exists():
        if [row.get("seed") for row in rows] != [42, 43, 50000] or not all(
                row.get("correct") is True and row.get("compiled") is True for row in rows):
            raise ValueError("Final benchmark lacks the complete successful seed suite")
    elif "ASCEND_SUITE_BINDING:" in text or "ROCM_DATA:" in text:
        raise ValueError("Final benchmark evidence is incomplete")
    metrics["trajectory"] = str(output.relative_to(workspace))
    return metrics


def final_metrics(workspace):
    """What this workspace's own benchmarks last measured, without re-running.

    AKO ends its loop by restoring its best iteration and benchmarking it, and
    every one of those runs used the same fixed-seed suite. Reading that
    result is not a second measurement of it.
    """
    from kernelbench_metrics import parse_benchmark
    outputs = sorted(Path(workspace).glob("trajectory/*/output.txt"),
                     key=lambda path: path.stat().st_mtime, reverse=True)
    for path in outputs:
        metrics = parse_benchmark(path.read_text(errors="replace"))
        runtime = metrics.get("candidate_median_ms")
        if metrics.get("correctness") is True and isinstance(runtime, (int, float)) and runtime > 0:
            return metrics
    return None


def run_fixed(api, run_id, task, resume=False, resume_export=False):
    """The controller owns the schedule: High first, then Native, once each.

    This is the whole of what makes the arm Fixed. The optimizer never chooses
    its level; it is handed TileLang, and then it is handed the AscendC that
    TileLang compiled to. Everything else here is the same launcher the other
    arms use, and nothing is re-measured afterwards -- each invocation ends
    with AKO's own restore and final, on the same fixed-seed suite.
    """
    if resume or resume_export:
        raise ValueError("Fixed runs once from an empty TileLang solution; resume is not supported")
    configure(api)
    root = api.RUN_ROOT / run_id
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = root / "reference.py"
    if reference.exists() or (root / "sessions").exists():
        lock.close()
        raise ValueError("Fixed run already initialized; a fresh run id is required")
    shutil.copyfile(api.resolve_reference(task), reference)

    state = dict(run_id=run_id, task=task, mode="fixed", status="running", stage="High",
                 backend="tilelang", device=api.DEVICE, agent_model=api.MODEL, high_only=False,
                 high_iterations=0, low_iterations=0, run_scope="full_fixed_cascade",
                 started_at=api.now(), started_at_epoch=time.time(), current={}, best={})
    api.publish(root / "state.json", state)
    try:
        first, second = root / "sessions/1", root / "sessions/2"
        prepare(api, first, reference, "tilelang")
        launch(api, root, first, "High", state)
        high = final_metrics(first)

        # The controller performs the lowering, because in this arm the
        # optimizer is not allowed to decide when it happens.
        transfer = root / "transfer"
        transfer.mkdir(exist_ok=True)
        state.update(phase="materializing")
        api.publish(root / "state.json", state)
        with (transfer / "export-output.txt").open("w") as log:
            subprocess.run([api.PYTHON, str(exporter(api)),
                            "--source", str(first / "solution/ModelNew.py"),
                            "--reference", str(reference),
                            "--destination", str(transfer / "solution"),
                            "--record", str(transfer / "record.json")],
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=api.BENCH_TIMEOUT_SECONDS)

        state.update(phase="validating_export")
        api.publish(root / "state.json", state)
        replay = evaluate(api, root, transfer / "solution/ModelNew.py", native_language(api),
                          transfer / "native-replay.txt")
        api.save(transfer / "validation.json", replay)

        prepare(api, second, reference, native_language(api))
        (second / "solution").rmdir()
        copy_source(transfer / "solution", second / "solution")
        launch(api, root, second, "Low", state)
        low = final_metrics(second)

        # This run yields two arms, not a winner. The first invocation never
        # saw a native representation and never heard of the second, so it is
        # exactly the High-only control and is reported as one; the second is
        # Fixed. Choosing between them here would discard a baseline.
        copy_source(first / "solution", root / "solution-high")
        copy_source(second / "solution", root / "solution-low")
        api.save(root / "results.json", {
            "high-only": dict(metrics=high, session="sessions/1", level="High", backend="tilelang"),
            "fixed": dict(metrics=low, session="sessions/2", level="Low",
                          backend=native_language(api), seeded_from="transfer/solution"),
            "basis": "each invocation's own final fixed-seed benchmark; neither was re-measured",
            "independence": "two AKO runs under the stock skill; neither was told of the other"})
        state.update(status="complete", phase="complete", stage="Low",
                     results={"high-only": high, "fixed": low})
    except Exception as error:
        state.update(failed_operation=state.get("phase"),
                     status="exhausted" if isinstance(error, TimeoutError) else "incomplete",
                     phase="failed", reason=str(error))
    finally:
        api.publish(root / "state.json", state)
        lock.close()
    return 0 if state["status"] == "complete" else 2


def run_fixed_low_restart(api, run_id, task, seed):
    """Fresh Low conversation from a user-selected, already validated export."""
    configure(api)
    root = api.RUN_ROOT / run_id
    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reference = root / "reference.py"
        if reference.exists() or (root / "sessions").exists():
            raise ValueError("Fixed Low restart needs a fresh run id")
        seed = Path(seed)
        source_reference = Path(api.resolve_reference(task))
        if (seed.parent / "reference.py").read_text() != source_reference.read_text():
            raise ValueError("Restart reference differs from the exported candidate's reference")
        shutil.copyfile(source_reference, reference)
        state = dict(run_id=run_id, task=task, mode="fixed", status="running", stage="Low",
                     backend=native_language(api), device=api.DEVICE, agent_model=api.MODEL,
                     high_only=False, high_iterations=0, low_iterations=0,
                     run_scope="fixed_low_restart_from_export", started_at=api.now(),
                     started_at_epoch=time.time(), current={}, best={})
        api.publish(root / "state.json", state)
        try:
            workspace = root / "sessions/2"
            prepare(api, workspace, reference, native_language(api))
            (workspace / "solution").rmdir()
            copy_source(seed, workspace / "solution")
            launch(api, root, workspace, "Low", state)
            metrics = final_metrics(workspace)
            copy_source(workspace / "solution", root / "solution-low")
            api.save(root / "results.json", {"fixed-low-restart": dict(metrics=metrics,
                     session="sessions/2", level="Low", backend=native_language(api)),
                     "basis": "fresh Low session from prior High export; High was not rerun"})
            state.update(status="complete", phase="complete", results={"fixed-low-restart":metrics})
        except Exception as error:
            state.update(status="exhausted" if isinstance(error, TimeoutError) else "incomplete",
                         phase="failed", reason=str(error))
        finally:
            api.publish(root / "state.json", state)
    return 0 if state["status"] == "complete" else 2
