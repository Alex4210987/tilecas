"""Read measured CURRENT/BEST from the original AKO trajectory transcripts."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import agent_driver
except ImportError:
    # Staged runs lay the drivers beside the protocol; the repo keeps them one
    # package over. Either way there is one table, not a copy per reader.
    sys.path.append(str(Path(__file__).resolve().parents[1] / "agents"))
    import agent_driver


def session_usage(path: Path) -> dict:
    """Codex exec resume starts a new per-turn counter in the same rollout."""
    data = path.read_bytes()
    turns = {}
    current = "initial"
    for line in data.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        # Claude Code records one usage block per assistant message; each is
        # already a per-request total, so accumulate them as separate turns.
        if event.get("type") == "assistant" and isinstance(event.get("message"), dict):
            usage = event["message"].get("usage") or {}
            counted = {k: v for k, v in usage.items() if k.endswith("tokens") and type(v) is int and v >= 0}
            if counted:
                turns[event.get("uuid") or event.get("requestId") or str(len(turns))] = counted
            continue
        payload = event.get("payload", {})
        if payload.get("type") == "task_started":
            current = payload.get("turn_id") or str(event.get("timestamp"))
        elif event.get("type") == "turn_context" and payload.get("turn_id"):
            current = payload["turn_id"]
        if payload.get("type") == "token_count" and payload.get("info"):
            usage = payload["info"].get("total_token_usage", {})
            turns[current] = {k: v for k, v in usage.items() if k.endswith("tokens") and type(v) is int and v >= 0}
    total = {}
    for usage in turns.values():
        for key, value in usage.items():
            total[key] = total.get(key, 0) + value
    return {"turns": turns, "total": total, "bytes": len(data)}


def collect_token_usage(run: Path) -> dict | None:
    """Keep a small cost artifact; never reset pre-resume usage or pull caches."""
    target = run / "token-usage.json"
    try:
        cached = json.loads(target.read_text())
    except (OSError, ValueError):
        cached = None
    # A dashboard can read both drivers in one process. Select from this run's
    # metadata, without importing the launcher's environment-bound driver.
    driver = None
    for name in ("state.json", "launch.json"):
        try:
            driver = json.loads((run / name).read_text()).get("agent_driver")
        except (OSError, ValueError):
            continue
        if driver in agent_driver.DRIVERS:
            break
    drivers = [driver] if driver in agent_driver.DRIVERS else list(agent_driver.DRIVERS)
    transcripts = agent_driver.TRANSCRIPT_DIRS
    paths, owners = [], {}

    def include(path, visit, name):
        # The workspace transcript and the home transcript may be symlinks to
        # the same file. Count it once, regardless of which path found it.
        resolved = path.resolve()
        if resolved not in owners:
            paths.append(resolved)
            owners[resolved] = (visit, name)

    for driver in drivers:
        for path in sorted(run.glob(f"sessions/*/.harness/{driver}-sessions/**/*.jsonl")):
            include(path, path.relative_to(run).parts[1], str(path.relative_to(run)))
    # Unix-isolated Ascend sessions own one neutral private optimizer home each.
    # Derive the location from the controller-owned workspace link, not from
    # model-written paths. Older deployments stored transcripts only here.
    for workspace in (run / "sessions").glob("*"):
        if any(visit == workspace.name for visit, _ in owners.values()):
            continue
        resolved = workspace.resolve()
        if resolved.parent != Path('/var/lib/ako/workspaces'):
            continue
        home = Path('/var/lib/ako/homes') / ('ako' + resolved.name[:12])
        for driver in drivers:
            try:
                legacy_paths = list((home / transcripts[driver]).glob('**/*.jsonl'))
            except PermissionError:
                continue
            for path in legacy_paths:
                include(path, workspace.name, f'private-session/{workspace.name}/{path.name}')
    # Current runs use the same Codex-history mounts as Fixed. Locate only
    # UUIDs recorded by this run's own CLI; never inspect another session.
    for log in (run / "worker-logs").glob("*.jsonl"):
        for line in log.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            session = event.get("thread_id", "")
            if event.get("type") != "thread.started" or not re.fullmatch(r"[0-9a-f-]{36}", session):
                continue
            for driver in drivers:
                for path in (Path.home() / transcripts[driver]).glob(f"**/*{session}.jsonl"):
                    include(path, log.stem, f"owned-session/{session}.jsonl")
    if not paths:
        return cached
    identity = {owners[p][1]: [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    if cached and cached.get("source_files") == identity:
        return cached
    workers = {}
    for path in paths:
        visit, source_name = owners[path]
        worker = workers.setdefault(visit, {"total": {}, "sessions": {}})
        usage = session_usage(path)
        worker["sessions"][source_name] = usage
        for key, value in usage["total"].items():
            worker["total"][key] = worker["total"].get(key, 0) + value
    record = {"version": "ako-worker-tokens-v1", "source_files": identity, "workers": workers,
              "semantics": "sum final counters per task_started turn; cached/reasoning tokens are subsets, not additional totals"}
    # Each process has its own temporary name; simultaneous readers are safe.
    temporary = target.with_name(f".token-usage-{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(target)
    except PermissionError:
        # The dashboard account can read transcripts without write access to
        # the root-owned run directory. Caching is optional for that reader.
        pass
    return record


def file_digest(path):
    """Compatibility inventory field: file length, never a checksum."""
    return {"bytes": Path(path).stat().st_size}


def key_manifest(run: Path) -> dict:
    """Select reproducibility records without environments or build caches."""
    collect_token_usage(run)
    names = {"state.json", "result.json", "reference.py", "HINTS.md", "ITERATIONS.md",
             "paper-ledger.json", "paper-labeler-output.txt", "candidate-contract-final.txt",
             "native-endpoint-output.txt",
             "agent-output.txt", "worker.log", "transfer.json", "manifest.json", "costs.jsonl",
             "incumbent.json", "route-summary.json", "token-usage.json", "endpoints.json", "selected-endpoint.json",
             "branch-sessions.json", "pending-profile.json"}
    source_suffixes = {".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".cxx"}
    files = {}
    for directory, dirs, entries in os.walk(run, followlinks=True):
        # Ascend's private Unix-user workspaces have controller-owned links.
        # Follow only these top-level session/review directories, never links
        # created inside candidate code or a profiler output tree.
        def allowed_directory(name):
            path = Path(directory) / name
            if not path.is_symlink():
                return True
            relative = path.relative_to(run)
            return (len(relative.parts) == 2 and relative.parts[0] in {"sessions", "routing"}
                    and path.resolve().parent == Path("/var/lib/ako/workspaces"))
        dirs[:] = [name for name in dirs if allowed_directory(name)]
        dirs[:] = [name for name in dirs if name not in
                   {".git", "__pycache__", ".cache", "torch_extensions", ".agent-state", ".bench-profiles", "profile.raw"}]
        for name in entries:
            path = Path(directory) / name
            relative = path.relative_to(run)
            parts = relative.parts
            selected = (len(parts) == 1 and name in names) or str(relative) == "scripts/bench.sh"
            selected |= parts[0] in {"solution", "trajectory"} and path.suffix in source_suffixes
            selected |= len(parts) == 3 and parts[0] == "trajectory" and name in {"output.txt", "protocol-invalid.json"}
            selected |= parts[0] in {"checkpoints", "transfer"} and path.suffix in source_suffixes | {".json", ".jsonl", ".md", ".txt", ".sh"}
            selected |= parts[0] == "sessions" and path.suffix in source_suffixes | {".json", ".md", ".txt", ".sh"}
            selected |= parts[0] == "solution" and path.suffix == ".json"
            selected |= parts[0] in {"routing", "worker-logs"} and path.suffix in {".json", ".jsonl", ".txt"}
            selected |= parts[0] == "repairs" and path.suffix in source_suffixes | {".json", ".jsonl", ".md", ".txt", ".sh"}
            selected |= parts[0] == "routing" and path.suffix in source_suffixes
            selected |= parts[0] in {"checkpoints", "transfer", "sessions"} and path.suffix in {".csv", ".ncu-rep"}
            if selected:
                files[str(relative)] = file_digest(path)
    git = subprocess.run(["git", "-C", str(run), "log", "--format=%H %aI %s"],
                         capture_output=True, text=True, check=False, timeout=10)
    return {"files": files, "git_log": git.stdout if git.returncode == 0 else ""}


def parse_benchmark(text: str) -> dict:
    result = {}
    for key, field in (("COMPILED", "compiled"), ("CORRECT", "correctness"),
                       ("RUNTIME", "candidate_median_ms"),
                       ("REF_RUNTIME", "reference_median_ms"),
                       ("REF_BASELINE", "reference_median_ms"), ("SPEEDUP", "speedup")):
        matches = re.findall(rf"^{key}:\s*([^\n]+)", text, re.M)
        if not matches:
            continue
        value = matches[-1].strip()
        if key in {"COMPILED", "CORRECT"}:
            result[field] = value.casefold() == "true"
        else:
            try:
                number = float(value.rstrip("x×").strip())
            except ValueError:
                continue
            if math.isfinite(number) and number > 0:
                result[field] = number
    if "CANDIDATE_CONTRACT: FAIL" in text:
        result.update(compiled=False, correctness=False)
    if "MUTATION_SENTINEL: FAIL" in text:
        result["correctness"] = False
    return result


def summarize(run: Path) -> dict:
    rows = []
    iterations = {}
    for path in sorted(run.glob("trajectory/*/output.txt")):
        output = path.read_text(errors="replace")
        row = parse_benchmark(output)
        # Ignore an output file still being written, until it has a verdict.
        if "compiled" not in row:
            continue
        if row["compiled"] and "correctness" not in row:
            continue
        row["trajectory"] = str(path.relative_to(run))
        if (path.parent / "protocol-invalid.json").exists():
            row["protocol_valid"] = False
        stage_match = re.search(r"^STAGE:\s*(High|Low)$", output, re.M)
        stage = stage_match.group(1) if stage_match else "High"
        row["stage"] = stage
        row["profiled"] = bool(re.search(r"(?:^|_)(?:ncu|profile|profiling)(?:[-_]|$)", path.parent.name)
                               or re.search(r"^==PROF==", output, re.M))
        match = re.search(r"_iter-(\d+)$", path.parent.name)
        if match:
            visit_match = re.search(r"^VISIT:\s*(\d+)$", output, re.M)
            key = (stage, visit_match.group(1) if visit_match else "1")
            iterations[key] = max(iterations.get(key, 0), int(match.group(1)))
            row["iteration"] = int(match.group(1))
        rows.append(row)
    # Reporting-only normalization preserves raw evaluator outputs and latency.
    normalization = run / "reference-reporting.json"
    if normalization.exists():
        fixed = json.loads(normalization.read_text())
        reference = run / "reference.py"
        if reference.is_file() and reference.read_text() == fixed.get("reference_text"):
            denominator = float(fixed["reference_ms"])
            if denominator <= 0:
                raise ValueError("Reporting reference must be positive")
            for row in rows:
                latency = row.get("candidate_median_ms", 0)
                if latency > 0:
                    row["reference_median_ms"] = denominator
                    row["speedup"] = denominator / latency
    measured = [row for row in rows if not row["profiled"] and row.get("protocol_valid", True)]
    valid = [row for row in measured if row.get("compiled") and row.get("correctness")
             and row.get("candidate_median_ms", 0) > 0]
    # Pick the lowest candidate latency. For ties use the latest measurement,
    # never the largest speedup caused by a noisy reference denominator.
    best = min(reversed(valid), key=lambda row: row["candidate_median_ms"]) if valid else {}
    result = {"current": measured[-1] if measured else {}, "best": best,
            "iteration_total": sum(iterations.values()), "evaluator_attempts": len(rows),
            "invalid_evaluator_attempts": sum(row.get("protocol_valid") is False for row in rows),
            "high_iterations": sum(n for (stage, _), n in iterations.items() if stage == "High"),
            "low_iterations": sum(n for (stage, _), n in iterations.items() if stage == "Low")}
    try:
        tokens = collect_token_usage(run)
    except OSError:
        # Read-only dashboards must still show measured results.
        tokens = None
    if tokens:
        result["worker_token_usage"] = {visit: data["total"] for visit, data in tokens["workers"].items()}
    return result


if __name__ == "__main__":
    run = Path(sys.argv[1])
    print(json.dumps(key_manifest(run) if "--manifest" in sys.argv[2:] else summarize(run)))
