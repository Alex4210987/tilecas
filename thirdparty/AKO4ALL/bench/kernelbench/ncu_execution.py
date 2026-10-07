"""Coordinate NCU across isolated /tmp namespaces; retry resource contention only."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time


RESOURCE_BUSY = 'Profiling failed because a driver resource was unavailable'


def shared_directory():
    # /var/tmp is shared by our bwrap workspaces; /tmp deliberately is not.
    root = Path('/var/tmp') / f'kernelbench-ncu-{os.getuid()}'
    root.mkdir(mode=0o700, exist_ok=True)
    return root


@contextmanager
def profiling_slot(timeout):
    root = shared_directory()
    started = time.monotonic()
    with (root / 'profiling.lock').open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError('Timed out waiting for the shared NCU profiling slot')
                time.sleep(.2)
        try:
            yield root, time.monotonic() - started
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_ncu(command, timeout=600):
    """Return only the successful attempt's output; never retry kernel failures."""
    failed = []
    with profiling_slot(timeout) as (root, queued_seconds):
        environment = os.environ.copy()
        environment['TMPDIR'] = str(root)
        environment['NV_COMPUTE_PROFILER_DISABLE_CONCURRENT_PROFILING'] = '1'
        for attempt in range(3):
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, env=environment, start_new_session=True)
            try:
                output, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
            result = subprocess.CompletedProcess(command, process.returncode, output)
            if not result.returncode or RESOURCE_BUSY not in output or attempt == 2:
                result.ncu_queue_seconds = queued_seconds
                result.failed_resource_attempts = failed
                return result
            failed.append(dict(returncode=result.returncode, output=output))
            time.sleep(2 * (attempt + 1))
