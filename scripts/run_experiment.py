#!/usr/bin/env python3
"""Start one optimization run: pick a backend, an agent and a task.

    run_experiment.py fixed level1/97 --backend ascend --agent codex --model gpt-6-astra

Two axes vary. The backend decides which chip the candidate is compiled and
timed on; the agent decides which code-assistant CLI writes the kernel. The
protocol between them -- the evaluator, the seeds, the tolerances, the
evaluation settings -- are recorded for each run. Optimization has no total
time limit, first-correct deadline, iteration quota or fixed performance target.

Each run is staged into its own directory under /tmp: the harness, the
reference material and the one task are copied there, and the run reads only
that copy. The checkout stays read-only during a run, two runs cannot reach
each other's material, and a run always gets the current reference material
rather than whatever a previous run happened to leave behind.

The card comes from ASCEND_RT_VISIBLE_DEVICES, as it does for every other
tool on this host. Run as root; the agent is dropped to its own account.
"""
from datetime import date
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
KERNELOPT = REPO / "kernelopt"
BACKENDS = KERNELOPT / "backends"
TASKS = REPO / "thirdparty/KernelBench/KernelBench"
MODES = ("ours", "fixed", "native-only", "unrestricted")
# Everything a run may write, kept out of the checkout it was staged from.
STAGE_ROOT = Path(os.environ.get("KERNELBENCH_STAGE", "/tmp/kernelopt"))
sys.path.insert(0, str(KERNELOPT / "agents"))
from agent_driver import DEFAULT_DRIVER, DEFAULT_MODELS, DRIVERS


def backends():
    return sorted(p.name for p in BACKENDS.iterdir() if p.is_dir() and not p.name.startswith("_"))


def resolve(task):
    """Accept level1/97, 1/97, 97_ScaledDotProductAttention or a full stem."""
    text = task.strip().removesuffix(".py")
    level, _, name = text.replace(":", "/").rpartition("/")
    level = "level" + level.removeprefix("level") if level else ""
    root = TASKS / level if level and (TASKS / level).is_dir() else TASKS
    files = list(root.rglob("*.py"))
    wanted = name.replace("-", "_")
    # Most specific first: the whole stem, then the leading task number, then
    # any stem containing the request.
    for matches in ([p for p in files if p.stem.replace("-", "_") == wanted],
                    [p for p in files if p.stem.split("_")[0] == wanted],
                    [p for p in files if wanted in p.stem.replace("-", "_")]):
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise SystemExit(f"Ambiguous task {task!r}: " + ", ".join(sorted(p.stem for p in matches)))
    raise SystemExit(f"No KernelBench reference matches {task!r}")


def stage(mode, backend, reference, run_id):
    """Copy the harness, the references and the one task into a fresh tree.

    The harness modules import each other by plain name, so the protocol,
    the selected backend and the agent drivers are laid side by side in one
    directory rather than nested into packages.  The mode selects which AKO
    skill the run reads, because that is the only prompt difference between
    the arms.
    """
    root = STAGE_ROOT / run_id
    if root.exists():
        raise FileExistsError(f"Staged run already exists: {root}; use a fresh run id")
    scripts = root / "thirdparty/KernelBench/scripts"
    scripts.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for source in (KERNELOPT / "protocol", BACKENDS / backend, KERNELOPT / "agents"):
        for path in source.glob("*.py"):
            shutil.copyfile(path, scripts / path.name)
    # The protocol shells out to a few helpers by path under <root>/scripts:
    # the metrics reader, the endpoint labeler and the single-task evaluator.
    # The labeler differs per backend -- Ascend records source text where
    # NVIDIA records a digest, because its agents may not compute one.
    helpers = root / "scripts"
    helpers.mkdir()
    for name in ("kernelbench_metrics.py", "paper_run.py", f"kernelbench_{backend}.py"):
        # A backend may override a helper: Ascend's labeler records source
        # text where the default records a digest, because its agents are
        # told not to compute one.
        for source in (BACKENDS / backend / name, KERNELOPT / "protocol" / name, REPO / "scripts" / name):
            if source.is_file():
                shutil.copyfile(source, helpers / name)
                break
    shutil.copytree(REPO / "knowledge" / backend, root / "knowledge", ignore=ignore)
    # The skill the optimizer reads. Baselines get the vendored AKO4ALL
    # release unmodified; ours gets the cross-level skill built on that same
    # release. Only the skill and its iteration template are staged -- never
    # upstream's own bench/, whose harness is not this experiment's fixed-seed
    # suite and must not be reachable as an alternative to scripts/bench.sh.
    skill = REPO / ("skills/tilecas" if mode == "ours" else "thirdparty/AKO4ALL")
    staged = root / "knowledge/ako4all"
    staged.mkdir(parents=True)
    for name in ("SKILL.md", "ITERATIONS.md"):
        if (skill / name).is_file():
            shutil.copyfile(skill / name, staged / name)
    for name in (f"{backend}-POLICY.md", f"{backend}-fixed-original-HINTS.md"):
        source = REPO / "configs" / name
        if source.is_file():
            shutil.copyfile(source, root / name.removeprefix(f"{backend}-"))
    task = root / "thirdparty/KernelBench/KernelBench" / reference.parent.name
    task.mkdir(parents=True)
    shutil.copyfile(reference, task / reference.name)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("task", help="KernelBench id, e.g. level1/97")
    parser.add_argument("--backend", default="ascend", choices=backends(),
                        help="which chip the candidate is compiled and timed on")
    parser.add_argument("--agent", choices=DRIVERS,
                        default=os.environ.get("KERNELBENCH_AGENT_DRIVER", DEFAULT_DRIVER),
                        help="which code-assistant CLI writes the kernel")
    parser.add_argument("--model", default=os.environ.get("KERNELBENCH_AGENT_MODEL"),
                        help="model name (default: gpt-6-astra for Codex, opus for Claude)")
    parser.add_argument("--effort", default=os.environ.get("KERNELBENCH_AGENT_EFFORT"), choices=("low", "medium", "high", "xhigh", "max"),
                        help="how far the optimizer explores before it answers")
    parser.add_argument("--shared", action="store_true",
                        help="the card also hosts other work, so label timings provisional")
    parser.add_argument("--run-id", help="override the generated run id")
    parser.add_argument("--dry-run", action="store_true", help="stage and report without launching")
    options = parser.parse_args()
    options.model = options.model or DEFAULT_MODELS[options.agent]
    options.effort = options.effort or ("low" if options.agent == "codex" else "high")
    if options.backend == "ascend" and not options.dry_run and os.geteuid() != 0:
        parser.error("Ascend session isolation requires root; run this launcher with sudo")

    if options.mode == "unrestricted" and options.backend != "ascend":
        parser.error("Unrestricted evaluation is currently available on Ascend")

    reference = resolve(options.task)
    task = f"{reference.parent.name}/{reference.stem}"
    short = reference.stem.split("_", 1)[-1].lower().replace("_", "-")[:32] or reference.stem
    run_id = options.run_id or f"kb-{options.mode}-{options.backend}-{short}-{options.agent}-{date.today():%Y%m%d}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        parser.error("invalid run id")
    runs = Path(os.environ.get("KERNELBENCH_RUNS", REPO / "runs")).resolve()
    if (runs / run_id).exists():
        parser.error(f"Run already exists: {runs / run_id}; use a fresh run id")
    root = stage(options.mode, options.backend, reference, run_id)

    card = int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0"))
    print(json.dumps({"mode": options.mode, "backend": options.backend, "agent": options.agent,
                      "model": options.model, "effort": options.effort, "task": task, "run_id": run_id,
                      "staged": str(root), "runs": str(runs), "device": f"npu:{card}"}, indent=2))
    if options.dry_run:
        return

    environment = dict(os.environ, KERNELBENCH_STAGED=str(root), KERNELBENCH_RUNS=str(runs),
                       KERNELBENCH_SOURCE_REPO=str(REPO),
                       KERNELBENCH_TASK=task, KERNELBENCH_RUN_ID=run_id,
                       KERNELBENCH_AGENT_DRIVER=options.agent, KERNELBENCH_AGENT_MODEL=options.model,
                       KERNELBENCH_AGENT_EFFORT=options.effort)
    command = [sys.executable, str(BACKENDS / options.backend / "launcher.py"), "start",
               options.mode, *(["--shared"] if options.shared else [])]
    raise SystemExit(subprocess.run(command, env=environment, cwd=str(root)).returncode)


if __name__ == "__main__":
    main()
