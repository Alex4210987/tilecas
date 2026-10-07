"""Platform mechanics shared by all three live policies."""
from pathlib import Path
import os
import subprocess


def cann_environment():
    """Load the installed toolkit environment for controller-spawned tools.

    A private optimizer's login shell initializes CANN independently; successful
    agent benchmarks therefore do not establish the controller's environment.
    Keep credentials and unrelated shell variables out of the returned mapping.
    """
    setup = Path(os.environ.get('KDA_ENV_SCRIPT', '/remote-home/S45149/ascend-migration-20260918/env.sh'))
    if not setup.is_file():
        raise FileNotFoundError(f'CANN environment script unavailable: {setup}')
    raw = subprocess.check_output(['bash', '-c', 'source "$1" >/dev/null && env -0',
                                   'cann-environment', str(setup)], timeout=30)
    keep = {'PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'TBE_IMPL_PATH', 'TOOLCHAIN_HOME', 'DDK_PATH'}
    values = {}
    for entry in raw.split(b'\0'):
        key, sep, value = entry.partition(b'=')
        if not sep:
            continue
        name = os.fsdecode(key)
        if name in keep or name.startswith(('ASCEND_', 'CANN_')):
            values[name] = os.fsdecode(value)
    return values


def is_ascend(api):
    return getattr(api, "BACKEND", "cuda") == "ascend"


def evaluation_stage_observations(api, raw):
    if not is_ascend(api):
        from cuda_evaluation_stages import evaluation_stage_observations as parse_cuda
        return parse_cuda(raw)
    import json
    import re
    from ascend_evaluation_stages import evaluation_stage_observations as parse
    value = parse(raw)
    records = []
    for line in raw.splitlines():
        if line.startswith('MSPROF_DATA: '):
            try:
                records.append(json.loads(line[len('MSPROF_DATA: '):]))
            except ValueError:
                pass  # Parser already records the malformed record.
    if (re.findall(r'^MSPROF_TIMING: (\S+)$', raw, re.M) != ['PASS']
            or re.findall(r'^EVALUATING_SEED: (\d+)$', raw, re.M) != ['42', '43', '50000']
            or len(records) != 1
            or any(not isinstance(row, dict) or row.get('correct') is not True
                   or row.get('compiled') is not True or row.get('seed') != 42
                   or row.get('timing') != 'msprof_forward_elapsed' for row in records)):
        value['status'] = 'partial'
        value['issues'].append({'reason': 'successful per-seed evaluation evidence missing'})
    return value


def native_language(api):
    if getattr(api, "BACKEND", "cuda") == "rocm":
        return "hip"
    return "ascendc" if is_ascend(api) else "cuda"


def exporter(api):
    return Path(__file__).with_name("export_ascend.py" if is_ascend(api) else "export_cuda.py")


def evaluator(api):
    return (api.ROOT / "thirdparty/KernelBench/scripts/ascend_bench.py" if is_ascend(api)
            else api.ROOT / "thirdparty/AKO4ALL/bench/kernelbench/seeded_bench.py")


def configure(api):
    paths = [str(Path(api.PYTHON).parent)]
    if is_ascend(api):
        os.environ.update(cann_environment())
        physical_device = str(api.DEVICE).split(":")[-1]
        if not physical_device.isdigit():
            raise ValueError(f"Invalid physical Ascend device: {api.DEVICE}")
        # set_env.sh inherits the controller environment and may echo a stale
        # visibility mapping.  The run's explicit physical card always wins;
        # evaluator code addresses its sole visible device logically as npu:0.
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = physical_device
    else:
        toolkit = Path("/usr/local/cuda-12.9")
        if toolkit.exists():
            os.environ["CUDA_HOME"] = str(toolkit)
            paths.append(str(toolkit / "bin"))
    os.environ["PATH"] = ":".join(paths + [os.environ.get("PATH", "")])


def references(api):
    return ("knowledge/LIBRARIES.md; knowledge/ascendc/SKILL.md; "
            "/root/tilelang-ascend (TileLang source, docs, component examples); "
            "/root/tilelang-ascend/3rdparty/catlass (CATLASS C++ components); "
            "/usr/local/Ascend/cann-9.0.0 (CANN installed headers/libraries). "
            "Use A2/910B APIs, not CUDA instructions. These are references, not extra agents.")


def library_rule(api):
    from ascend_component_policy import RULE
    return RULE


def profile_script(api, cuda_script):
    if not is_ascend(api):
        return cuda_script
    return '''from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ascend_bench import profile_candidate
profile_candidate(*sys.argv[1:])
'''
