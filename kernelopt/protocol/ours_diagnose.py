"""Where did the measured device time go? Installed as scripts/diagnose.py.

The evaluator already records every device task of every timed forward, and
the candidate's own compiled sources already declare their device entry
symbols. Nothing joined the two, so a stalled optimizer could see that its
kernel takes 13 ms without seeing which of its kernels that was, or how much
of the forward never entered its kernels at all.

The plain entry point reuses or collects source-matched cross-level evidence.
One no-argument entry point reports all kernels.
Diagnostic work never produces a new candidate-ranking measurement.
"""
import json
import shutil
import subprocess
from pathlib import Path
import re
import sys

import attribution


def latest_output(workspace, explicit=None):
    """The most recent benchmark attempt, including failed attempts."""
    if explicit:
        return Path(explicit)
    candidates = [path for path in workspace.glob("trajectory/*/output.txt")
                  if not re.search(r"(?:^|_)(?:ncu|profile)(?:[-_]|$)", path.parent.name)]
    if not candidates:
        raise SystemExit("No bench output yet. Run scripts/bench.sh first.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


EXPORT_CACHE = ".cache/lowering"


def lowering_map(workspace, python, refresh=False):
    """Which high-level call produced which kernel, from the compiler itself.

    The measured trace names every kernel `main_kernel`, and the compiler's
    cache directories are content hashes that collapse two identical kernels
    into one -- so neither can say which launch belongs to which call. The
    exporter can: it runs the program, records the call site of each kernel in
    the order they are issued, and writes the native source beside it.

    Re-running it is not free, but it compiles against the cache the benchmark
    just filled, so it is one trace of the Python rather than a rebuild. The
    result is kept until `solution/` changes.
    """
    solution = workspace / "solution"
    # Staleness is decided by the source text itself, not a digest of it: the
    # run's rules forbid computing checksums, and agents were bypassing this
    # helper to honour that.
    stamp = repr(sorted(
        (str(path.relative_to(solution)), path.read_bytes().decode("utf-8", "surrogateescape"))
        for path in solution.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
        and ".cache" not in path.parts and "__pycache__" not in path.parts))
    cache = workspace / EXPORT_CACHE
    record = cache / "export.json"
    if record.is_file() and (cache / "stamp").is_file() and (cache / "stamp").read_text() == stamp:
        return json.loads(record.read_text()), cache

    if not refresh:
        return None, "no source-matched lowering cache"
    if cache.exists():
        shutil.rmtree(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [python, "scripts/export.py", "--source", "solution/ModelNew.py",
         "--reference", "reference.py", "--destination", str(cache),
         "--record", str(record)],
        cwd=workspace, capture_output=True, text=True, timeout=1800)
    if result.returncode or not record.is_file():
        return None, result.stderr[-400:] or result.stdout[-400:]
    (cache / "stamp").write_text(stamp)
    return json.loads(record.read_text()), cache


def source_line(workspace, callsite):
    """The high-level line itself, so the report shows code and not a number."""
    if not callsite or not callsite.get("line"):
        return ""
    path = workspace / callsite["file"]
    try:
        if not path.is_file():
            path = workspace / Path(callsite["file"]).name
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    index = callsite["line"] - 1
    return lines[index].strip() if 0 <= index < len(lines) else ""


def compiled_objects(workspace, output, lowering=None, exported=None):
    """The source objects that own device work, in the order they are issued."""
    solution = workspace / "solution"
    native = {str(path.relative_to(solution)): path.read_text(errors="replace")
              for path in sorted(solution.rglob("*"))
              if path.is_file() and path.suffix in attribution.NATIVE_SUFFIXES}
    if native:
        return attribution.compiled_objects({"level": "Native", "source_files": native}, None)
    if lowering:
        kernels = []
        for kernel in lowering["kernels"]:
            path = Path(exported) / kernel["source"]
            kernels.append(dict(kernel, device_text=path.read_text(errors="replace")
                                if path.is_file() else kernel.get("device_text", "")))
        return attribution.compiled_objects({"level": "High"}, {"kernels": kernels})
    return []


# Each hardware pipe is fed by one kind of statement, so the pipe that owns a
# kernel's time says which lines of that kernel to look at.
# Each hardware pipe is fed by one kind of statement, and the compiler emits
# one kind of instruction for it. Knowing both turns a busy pipe into lines --
# in what you wrote, and in what it became.
PIPES = (
    ("aic_mte2", "global -> L1 loads",
     r"T\.copy\(\s*(?:%s)\b", r"copy_gm_to_l1"),
    ("aic_mte1", "L1 -> L0 loads, issued by the matmul",
     r"T\.(?:gemm|mma)\w*\(", r"LoadData|load_l1_to_l0"),
    ("aic_mac", "matmul on the cube",
     r"T\.(?:gemm|mma)\w*\(", r"Mmad|::gemm|mmad"),
    ("aic_fixpipe", "accumulator -> global writes",
     r"T\.copy\([^,]*,\s*(?:%s)\b", r"copy_l0c_to_"),
    ("aiv_vec", "vector arithmetic",
     r"T\.(?:exp|reduce\w*|max|sum|clear|fill)\w*\(",
     r"AscendC::(?:Exp|Max|Sum|Add|Mul|Sub|Div|Reduce|Cast|Brcb)"),
    ("aiv_mte2", "global -> UB loads", r"T\.copy\(\s*(?:%s)\b", r"copy_gm_to_ub"),
    ("aiv_mte3", "UB -> global stores", r"T\.copy\([^,]*,\s*(?:%s)\b", r"copy_ub_to_gm"),
    ("aic_scalar", "scalar work on the cube", r"for\s|if\s", r"for\s*\(|if\s*\("),
    ("aiv_scalar", "scalar work on the vector cores", r"for\s|if\s", r"for\s*\(|if\s*\("),
)
# A kernel that never touches the cube reports cube-side ratios against a
# negligible base, where a rounding artefact looks like a finding.
CUBE_PIPES = ("aic_mte2", "aic_mte1", "aic_mac", "aic_fixpipe", "aic_scalar")



def contents(root):
    suffixes={'.py','.cpp','.cc','.cxx','.h','.hpp','.json'}
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*')
            if p.is_file() and p.suffix in suffixes and not any(
                part in {'.cache','__pycache__','.git','torch_extensions'} for part in p.relative_to(root).parts)}

def collect_profile(workspace, python):
    import os, time
    before=contents(workspace/'solution')
    if 'ModelNew.py' not in before:
        raise ValueError('No candidate to profile')
    report=workspace/'.profiles'/str(time.time_ns())
    source=report/'source'
    (source/'solution').mkdir(parents=True)
    for name,data in before.items():
        path=source/'solution'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    shutil.copyfile(workspace/'reference.py',source/'reference.py')
    env=os.environ.copy()
    env.update(AKO_PROFILE_OUTPUT=str(report/'profile.csv'),AKO_PROFILE_METRICS='PipeUtilization')
    command=[python, '-c', "import sys; sys.path.insert(0, 'scripts'); from ascend_bench import profile_candidate; profile_candidate(sys.argv[1], sys.argv[2])", str(source/'reference.py'), str(source/'solution/ModelNew.py')]
    started=time.time()
    with (report/'profile.log').open('w') as output:
        result=subprocess.run(command,cwd=workspace,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=1800)
    record=dict(platform='ascend',returncode=result.returncode,started_at=started,finished_at=time.time(),
        source_unchanged=before==contents(workspace/'solution') and before==contents(source/'solution'),
        metrics='PipeUtilization',scope='one complete forward; summed selected device task durations',
        command=command)
    (report/'profile.record.json').write_text(json.dumps(record,indent=2)+'\n')
    if result.returncode or not record['source_unchanged']:
        raise RuntimeError('Profile failed or source changed: ' + (report/'profile.log').read_text(errors='replace')[-400:])
    return report/'profile.csv'


def pipe_profile(workspace, python):
    """Per-pipe occupancy for the kernels of the current solution.

    `scripts/bench.sh` measures how long each kernel took; this says what it
    spent that time doing. It is a separate collection run, so it is only
    taken when asked for.
    """
    import csv, collections
    try:
        path = collect_profile(workspace, python)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        return None, str(error)
    per = collections.defaultdict(dict)
    for row in csv.DictReader(path.open()):
        try:
            per[int(row["ID"])][row["Metric Name"]] = float(row["Metric Value"])
        except (ValueError, KeyError):
            continue
    return dict(per), None


def dominant_pipe(metrics):
    """The pipe that owns the most of one kernel's time, and its share."""
    # The collector reports cube-side ratios even for a kernel that barely
    # uses the cube, where they are a rounding artefact on a tiny base. Its
    # own utilisation figure is the honest gate.
    idle_cube = (metrics.get("cube_utilization(%)") or 0.0) < 20.0
    best = None
    for key, meaning, _, _ in PIPES:
        if idle_cube and key in CUBE_PIPES:
            continue
        ratio = metrics.get(key + "_ratio")
        if ratio is not None and (best is None or ratio > best[1]):
            best = (key, ratio, meaning)
    return best, idle_cube


def paired_lines(workspace, exported, native_file, high_pattern, native_pattern, buffers):
    """The same statement seen at both levels, matched by the buffers it names.

    A pipe says what kind of work dominates; these say which work. The two
    levels are paired on the identifiers they share, because a copy of Q into
    L1 is called `T.copy(Qh, q_l1)` above and `copy_gm_to_l1(q_l1, Qh)` below,
    and the names survive the lowering even though nothing else does.
    """
    high = matching_lines(None, high_pattern, buffers, root=workspace, limit=12,
                          scope=kernel_body(workspace, buffers))
    native = (matching_lines([workspace / exported / native_file], native_pattern,
                             root=workspace / exported, limit=12)
              if native_file and exported else [])
    taken, pairs = set(), []
    for where, code in high:
        names = set(re.findall(r"[A-Za-z_]\w*", code))
        best, score = None, 0
        for index, (other, text) in enumerate(native):
            if index in taken:
                continue
            shared = len(names & set(re.findall(r"[A-Za-z_]\w*", text)))
            if shared > score:
                best, score = index, shared
        if best is not None and score >= 2:
            taken.add(best)
            pairs.append((where, code, native[best][0], native[best][1]))
        else:
            pairs.append((where, code, None, None))
    for index, (where, code) in enumerate(native):
        if index not in taken:
            pairs.append((None, None, where, code))
    return pairs


def kernel_body(workspace, buffers):
    """The lines of the one prim_func that declares these buffers.

    Without this the search runs over the whole file and returns loops from a
    kernel that has nothing to do with the measurement in hand.
    """
    wanted = {b for b in buffers if b}
    best = None
    for path in sorted((workspace / "solution").rglob("*.py")):
        lines = path.read_text(errors="replace").splitlines()
        opens = [i for i, line in enumerate(lines)
                 if re.search(r"@T\.prim_func|def\s+\w+\s*\(", line)]
        for index, start in enumerate(opens):
            end = opens[index + 1] if index + 1 < len(opens) else len(lines)
            body = lines[start:end]
            declared = set(re.findall(r"[A-Za-z_]\w*", "\n".join(body[:12])))
            score = len(wanted & declared)
            if score and (best is None or score > best[0]):
                best = (score, path, start, end, body)
    if best is None:
        return None, 0, []
    _, path, start, _, body = best
    return path, start, body


def matching_lines(paths, pattern, buffers=(), root=None, limit=5, scope=None):
    """Lines of the kind of statement or instruction that feeds the busy pipe."""
    names = "|".join(re.escape(b) for b in buffers) or r"\w+"
    expression = re.compile(pattern % names if "%s" in pattern else pattern)
    found = []
    if scope:
        path, offset, body = scope
        if path is None:
            return found
        for number, line in enumerate(body, offset + 1):
            if expression.search(line):
                found.append((f"{path.relative_to(root) if root else path.name}:{number}",
                              line.strip()))
                if len(found) >= limit:
                    return found
        return found
    for path in paths:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if expression.search(line):
                where = path.relative_to(root) if root else path.name
                found.append((f"{where}:{number}", line.strip()))
                if len(found) >= limit:
                    return found
    return found


# The generated kernel declares one arena per on-chip memory and sub-allocates
# it by byte offset, so the program's high-water mark is readable straight out
# of the emitted code. Time is what a profiler measures; capacity is what
# actually bounds a tile size or a double buffer, and a loop that sees only the
# first optimises blind on the dimension that binds. One run spent five
# iterations chasing a "compiler bug" that was a 337 KB working set in a 192 KB
# buffer.
# Both the emitted C++ type names and the exporter's dtype names, since the
# same table sizes a register tile and a global staging tensor.
WIDTH = {"half": 2, "float": 4, "float32": 4, "bfloat16_t": 2, "int8_t": 1,
         "uint8_t": 1, "int16_t": 2, "uint16_t": 2, "int32_t": 4, "uint32_t": 4,
         "int64_t": 8, "uint64_t": 8, "float16": 2, "bfloat16": 2, "double": 8,
         "float64": 8, "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
         "int32": 4, "uint32": 4, "int64": 8, "uint64": 8, "bool": 1}
ARENAS = {"l1": "L1", "ub": "UB", "l0a": "L0A", "l0b": "L0B", "l0c": "L0C"}


def onchip_occupancy(text):
    """High-water bytes and capacity per on-chip memory, from the emitted code."""
    capacity = {name: int(size) for name, size in
                re.findall(r"InitBuffer\(\s*ascend_(\w+)\s*,\s*(\d+)\)", text)}
    high = {}
    for arena, dtype, count, offset in re.findall(
            r"ascend_(\w+)\.GetWithOffset<(\w+)>\(\s*(\d+)\s*,\s*(\d+)\s*\)", text):
        width = WIDTH.get(dtype)
        if width is None:
            continue
        high[arena] = max(high.get(arena, 0), int(offset) + int(count) * width)
    rows = []
    for arena, label in ARENAS.items():
        used, total = high.get(arena), capacity.get(arena)
        if used and total:
            rows.append(f"{label} {used/1024:.0f}/{total/1024:.0f} KB ({used/total:.0%})")
    return rows


# Staging tensors live in global memory, so they cost HBM bandwidth on every
# re-read unless the whole set stays resident in last-level cache. Two programs
# can touch the same bytes and differ by milliseconds purely on which memory
# serves them, and nothing in a pipe profile shows it. The sizes are already
# recorded by the exporter.
L2_BYTES = 192 * 1024 * 1024


def gm_workspaces(exported):
    """Global staging tensors the program allocates, and their total."""
    record = Path(exported) / "program.json"
    if not record.is_file():
        return [], 0
    try:
        program = json.loads(record.read_text())
    except ValueError:
        return [], 0
    # Native authors may use program.json as descriptive metadata rather than
    # the exporter's tensor ABI. Missing/unknown fields are not zero traffic.
    if not isinstance(program, dict) or not isinstance(program.get("inputs"), list):
        return [], 0
    if not all(isinstance(name, str) for name in program["inputs"]):
        return [], 0
    if not isinstance(program.get("output"), str):
        return [], 0
    named = set(program["inputs"]) | {program["output"]}
    rows, total, seen = [], 0, set()
    for kernel in program.get("kernels") or []:
        if not isinstance(kernel, dict):
            continue
        for param in kernel.get("params") or []:
            if not isinstance(param, dict):
                continue
            name = param.get("value")
            shape = param.get("shape")
            if (not isinstance(name, str) or not isinstance(shape, list)
                    or not all(type(n) is int and n >= 0 for n in shape)
                    or param.get("dtype") not in WIDTH or name in named or name in seen):
                continue
            seen.add(name)
            count = 1
            for extent in param.get("shape") or []:
                count *= extent
            size = count * WIDTH.get(param.get("dtype"), 4)
            total += size
            rows.append(f"{name} {param.get('dtype')}{list(param.get('shape') or [])}"
                        f" {size/1048576:.0f} MB")
    return rows, total


def report(record, samples, workspace=None, exported=None):
    lines = []
    if record["status"] != "attributed":
        totals, count = {}, max(len(samples), 1)
        for sample in samples:
            for row in sample["rows"]:
                totals[row["kernel"]] = totals.get(row["kernel"], 0.0) + row["value"]
        measured = sum(totals.values()) or 1.0
        lines.append(f"Not aligned across levels: {record.get('reason', record['status'])}.")
        lines.append("Measured device tasks by name:\n")
        for name, value in sorted(totals.items(), key=lambda item: -item[1]):
            lines.append("  {:40s}{:9.4f}{:9.1%}".format(name[:39], value / count, value / measured))
        return "\n".join(lines)

    total = record["mean_device_work_ms"]
    lines.append(f"Measured forward: {total:.3f} ms of device work, "
                 f"attributed over {record['samples']} timed samples.\n")
    top_entries = record.get("top_bottlenecks", record["objects"])
    for entry in top_entries:
        if entry.get("object_id") == "unmapped":
            continue
        call = entry.get("callsite") or {}
        span = entry.get("span") or {}
        share = entry["share_of_device_work"] or 0
        lines.append(f"  {entry['mean_ms']:8.3f} ms  {share:6.1%}  of the forward")
        if call.get("line"):
            code = source_line(workspace, call) if workspace else ""
            where = f"{call['file']}:{call['line']}"
            lines.append(f"      high    {where}" + (f"   {code}" if code else ""))
        if span.get("file"):
            reach = f":{span['line']}-{span['end_line']}" if span.get("end_line") else ""
            lines.append(f"      native  {exported}/{span['file']}{reach}"
                         if exported else f"      native  {span['file']}{reach}")
        if entry.get("signature"):
            lines.append(f"      buffers {entry['signature']}")
        if workspace and span.get("file") and exported:
            emitted = Path(workspace) / exported / span["file"]
            if emitted.is_file():
                rows = onchip_occupancy(emitted.read_text(errors="replace"))
                if rows:
                    lines.append("      on-chip " + "   ".join(rows))
        lines.append("")
    if workspace and exported:
        rows, total = gm_workspaces(Path(workspace) / exported)
        if rows:
            verdict = "allocation footprint only; cache residency and HBM traffic are unmeasured"
            lines.append(f"  Global staging: {total/1048576:.0f} MB — {verdict}.")
            lines.append("      " + "   ".join(rows) + "\n")
    residual = record.get("unexported") or {}
    if residual.get("mean_ms") and any(e.get("object_id") == "unmapped" for e in top_entries):
        lines.append(f"  {residual['mean_ms']:8.3f} ms  {residual['share_of_device_work'] or 0:6.1%}"
                     "  in unmapped device symbols — ownership unknown")
        lines.append(f"      {', '.join(residual.get('names', [])[:4])}\n")
    top = record["objects"][0] if record["objects"] else None
    if top and (top["share_of_device_work"] or 0) >= 0.5:
        lines.append(f"One kernel accounts for {top['share_of_device_work']:.0%} of device work. "
                     "Source sites localize the cost; they do not establish its cause or choose a level.")
    return "\n".join(lines)


SOURCE_SUFFIXES = {".py", ".cpp", ".cu", ".cc", ".cxx", ".h", ".hpp", ".cuh", ".json", ".tir"}


def current_matches_snapshot(workspace, snapshot):
    """Compare authored text, not checksums; never attach old timing to an edit."""
    solution = workspace / "solution"
    current = {str(p.relative_to(solution)): p.read_bytes() for p in solution.rglob("*")
               if p.is_file() and p.suffix in SOURCE_SUFFIXES
               and not {".cache", "__pycache__"}.intersection(p.parts)}
    if not current or not (snapshot / "ModelNew.py").is_file():
        return False
    # Root source files distinguish a native branch from an older High snapshot.
    frozen_root = {p.name for p in snapshot.iterdir() if p.is_file() and p.suffix in SOURCE_SUFFIXES}
    current_root = {name for name in current if "/" not in name}
    if frozen_root != current_root:
        return False
    return all((snapshot / name).is_file() and (snapshot / name).read_bytes() == data
               for name, data in current.items())


def snapshot_objects(snapshot):
    """Native files or compiler output saved with this exact benchmark."""
    native = [p for p in snapshot.iterdir() if p.is_file() and p.suffix in attribution.NATIVE_SUFFIXES]
    if not native:
        # TileLang saves both device text and a wrapped copy with the same entry.
        # Keep one copy per compilation; directory order does not establish launches.
        native = list(snapshot.rglob("kernel.cu")) + list(snapshot.rglob("kernel.cpp"))
    sources = {str(p.relative_to(snapshot)): p.read_text(errors="replace") for p in native}
    return attribution.compiled_objects({"level": "Native", "source_files": sources}, None)


def dataflow_sites(root, entry):
    """Static transfer/sync sites associated with the measured entry, not causality."""
    span = entry.get("span") or {}
    path = root / span.get("file", "")
    if not path.is_file():
        return []
    lines = path.read_text(errors="replace").splitlines()
    begin, end = span.get("line") or 1, span.get("end_line") or len(lines)
    pattern = re.compile(r"copy_|DataCopy|LoadData|Fixpipe|CrossCore|PipeBarrier|SetFlag|WaitFlag")
    return [dict(file=str(path), line=i, text=line.strip(), grade="static_site")
            for i, line in enumerate(lines, 1) if begin <= i <= end and pattern.search(line)][:6]


def diagnose(workspace, output, *, refresh_lowering=False):
    from evaluation_stages import evaluation_stage_observations
    output = output.resolve()
    snapshot = output.parent
    observations = evaluation_stage_observations(output.read_text(errors="replace")) or {}
    samples = observations.get("samples") or []
    matched = current_matches_snapshot(workspace, snapshot)
    objects = snapshot_objects(snapshot)
    source_root = snapshot
    mapping_note = "benchmark snapshot; High source mapping unavailable without recorded provenance"
    native = any(p.is_file() and p.suffix in attribution.NATIVE_SUFFIXES for p in snapshot.iterdir())
    if refresh_lowering and (not matched or native):
        raise ValueError("Collecting lowering requires the unchanged measured High candidate; Native uses its saved source directly")
    if not native and matched:
        lowering, exported = lowering_map(workspace, sys.executable, refresh=refresh_lowering)
        if lowering:
            objects = compiled_objects(workspace, output, lowering, exported)
            source_root = Path(exported)
            mapping_note = "source-matched High lowering; source correspondence is not causal proof"
    if samples and observations.get("status") == "partial":
        record = dict(status="partial_timing", objects=[],
                      reason="incomplete/invalid suite; task timings are diagnostic only")
    elif samples:
        record = attribution.attribute(samples, objects)
    else:
        record = dict(status="no_timing", objects=[], reason="failed/unmeasured candidate: static source evidence only")
    record.update(benchmark=str(output), source_snapshot=str(snapshot),
                  source_root=str(source_root), current_sources_match=matched, mapping_note=mapping_note,
                  suite_status=observations.get("status", "unknown"), suite_issues=observations.get("issues", []))
    top = [dict(entry, dataflow=dict(parameters=entry.get("signature", ""),
                                  sites=dataflow_sites(source_root, entry), grade="static_source"))
           for entry in record.get("objects", []) if entry.get("tasks_per_sample", 0) > 0]
    residual = record.get("unexported") or {}
    if residual.get("mean_ms"):
        top.append(dict(object_id="unmapped", grade="unknown", **residual))
    if record["status"] != "attributed" and samples:
        totals = {}
        for sample in samples:
            for row in sample.get("rows", []):
                totals[row["kernel"]] = totals.get(row["kernel"], 0) + row["value"]
        total = sum(totals.values())
        top = [dict(symbol=name, mean_ms=value/len(samples), share_of_device_work=value/total if total else None,
                    grade="unmapped", source_position=None) for name, value in totals.items()]
    record["top_bottlenecks"] = sorted(top, key=lambda x: x.get("mean_ms", 0), reverse=True)
    record["capacity"] = {str(p.relative_to(snapshot)): onchip_occupancy(p.read_text(errors="replace"))
                          for p in snapshot.rglob("*") if p.is_file() and p.suffix in attribution.NATIVE_SUFFIXES}
    return record, samples


def main():
    if len(sys.argv) != 1:
        raise SystemExit("Usage: python scripts/diagnose.py (no arguments)")
    workspace = Path.cwd()
    output = latest_output(workspace)
    matched = current_matches_snapshot(workspace, output.parent)
    native = any(p.is_file() and p.suffix in attribution.NATIVE_SUFFIXES for p in output.parent.iterdir())
    record, samples = diagnose(workspace, output, refresh_lowering=matched and not native)
    collect = record["current_sources_match"] and bool(samples) and record["status"] not in {"no_timing", "partial_timing"}
    if collect:
        per, error = pipe_profile(workspace, sys.executable)
        record["hardware_profile"] = dict(launch_metrics=per, error=error,
            interpretation="separate counter collection; not benchmark ranking or cross-engine causal attribution")
    print(f"Benchmark: {record['benchmark']}")
    print(f"Source: {record['source_snapshot']}; matches current: {record['current_sources_match']}")
    print(record["mapping_note"])
    print(report(record, samples, workspace=Path(record["source_snapshot"]), exported=record["source_root"]))
    for entry in record["top_bottlenecks"]:
        for site in (entry.get("dataflow") or {}).get("sites", [])[:3]:
            print(f"  static site {site['file']}:{site['line']}  {site['text'][:120]}")
    if not samples:
        for name, occupancy in record["capacity"].items():
            if occupancy:
                print(name + ": " + "; ".join(occupancy))
    if collect:
        print("Hardware counters (separate collection):")
        print(json.dumps(record["hardware_profile"], indent=2))


if __name__ == "__main__":
    main()
