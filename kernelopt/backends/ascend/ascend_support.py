"""Ascend evaluator entrypoints for the common live controllers."""
from pathlib import Path
import json
import shlex
import shutil
import subprocess
import time


def install_wrapper(api, directory, reference, language, detect_level=False, deadline=None):
    from ascend_isolation import prepare_directory
    workspace = prepare_directory(directory)
    from deepseek_platform_hints import for_model
    notes = for_model(api.MODEL)
    if notes:
        with (workspace / 'HINTS.md').open('a') as hints:
            hints.write(notes)
    scripts = workspace / 'scripts'
    scripts.mkdir(exist_ok=True)
    for name in ('ascend_bench.py', 'validate_candidate.py', 'ascend_evidence.py', 'ascend_component_policy.py'):
        shutil.copyfile(Path(__file__).with_name(name), scripts / name)
    # Preserve the same trajectory/labels and no per-iteration policy quotas.
    command = (f'{shlex.quote(api.PYTHON)} scripts/ascend_bench.py --ref reference.py '
               '--solution solution/ModelNew.py --backend "$BACKEND"')
    # A workspace that owns both levels cannot have its contract fixed when it
    # is created: pinning the backend here made the evaluator refuse the very
    # level change the arm exists to observe. Read it off solution/ instead.
    detect = ('BACKEND={}\n'.format(shlex.quote(language)) if not detect_level else
              'if [ -n "$(ls solution/*.cpp solution/*.cu solution/*.h 2>/dev/null)" ]; then\n'
              '  BACKEND=ascendc\n'
              'else\n'
              '  BACKEND=tilelang\n'
              'fi\n')
    remaining = ""
    wrapper = f'''#!/bin/bash
source /remote-home/S45149/ascend-migration-20260918/env.sh
set -o pipefail
cd "$(dirname "$0")/.."
mkdir -p .cache/tmp
export TMPDIR="$PWD/.cache/tmp"
export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(api.DEVICE.split(':')[-1])}
# The evaluator freshly measures reference and candidate with identical inputs.
# Historical scalar baselines do not override the elapsed-time denominator.
{detect}LABEL="${{1:-manual}}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAJ_DIR="trajectory/${{TIMESTAMP}}_${{LABEL}}"
mkdir -p "$TRAJ_DIR"
# TileLang already writes every kernel it compiles as readable source. Point
# it at this run's own record, so the AscendC a benchmark actually ran sits
# beside that benchmark's output instead of in a cache nobody is told about.
export TILELANG_CACHE_DIR="$PWD/$TRAJ_DIR"
timeout --signal=TERM --kill-after=30s {api.BENCH_TIMEOUT_SECONDS} {command} 2>&1 | tee _bench_output.txt
STATUS=${{PIPESTATUS[0]}}
cp -r solution/* "$TRAJ_DIR/" 2>/dev/null || true
cp _bench_output.txt "$TRAJ_DIR/output.txt"
rm -f _bench_output.txt
{remaining}
exit "$STATUS"
'''
    (scripts / 'bench.sh').write_text(wrapper)
    (scripts / 'bench.sh').chmod(0o755)
    if not detect_level:
        shutil.copyfile(Path(__file__).with_name('ascend_profile.py'), scripts / 'ascend_profile.py')
        msprof = scripts / 'msprof.sh'
        msprof.write_text('#!/bin/bash\nset -e\ncd "$(dirname "$0")/.."\n' + api.PYTHON + ' scripts/ascend_profile.py "$@"\n')
        msprof.chmod(0o755)
    ignore = workspace / '.gitignore' 
    if not ignore.exists():
        ignore.write_text('.bench-profiles/\n.cache/\n.harness/\n__pycache__/\ntrajectory/\n.session-identity.json\n')


def label_endpoint(api, directory, reference, mode, stage):
    from execution_platform import evaluator
    from kernelbench_metrics import parse_benchmark
    backend = 'tilelang' if stage.lower() == 'high' else 'ascendc'
    started = time.time()
    command = [api.PYTHON, str(evaluator(api)), '--ref', str(reference),
               '--solution', str(directory / 'solution/ModelNew.py'), '--backend', backend,
               '--out', str(directory / 'endpoint-profile')]
    result = subprocess.run(command, capture_output=True, text=True, timeout=api.BENCH_TIMEOUT_SECONDS)
    (directory / 'paper-labeler-output.txt').write_text(result.stdout + result.stderr)
    metrics = parse_benchmark(result.stdout)
    valid = result.returncode == 0 and metrics.get('correctness') is True
    files = {str(p.relative_to(directory / 'solution')): (p.read_bytes()).decode("utf-8", errors="surrogateescape")
             for p in (directory / 'solution').rglob('*') if p.is_file() and p.suffix in {'.py','.cpp','.h','.hpp','.json'}
             and '.cache' not in p.parts and '__pycache__' not in p.parts}
    ledger = dict(schema_version='tilecas-live-ascend-v1', mode=mode, backend=backend,
                  task=dict(reference_text=(reference.read_bytes()).decode("utf-8", errors="surrogateescape")),
                  protocol=dict(seeds=[42,43,50000],correct=10,warmup=3,timed=5,timing_input_seed=42,atol=.001,rtol=.001,timing='msprof_forward_elapsed'),
                  final_endpoint=dict(valid=valid, **metrics, source_files=files),
                  resource_costs=dict(wall_seconds=time.time()-started))
    api.save(directory / 'paper-ledger.json', ledger)
    return valid
