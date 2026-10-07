"""Run a fresh optimizer history and retain the one shared OAuth login."""
import os
from pathlib import Path
import signal
import subprocess
import sys
private=Path(sys.argv[1]); canonical=Path(sys.argv[2])
(private/'sessions').mkdir(exist_ok=True)
transcripts=private.parent/'codex-sessions'
if not transcripts.exists(): transcripts.symlink_to('codex-home/sessions',target_is_directory=True)
workspace=private.parents[1]
if not (workspace/'.git').exists():
    subprocess.run(['git','init','-b','codex/rocm-workspace',str(workspace)],check=True,stdout=subprocess.DEVNULL)
# Random's outer budget owns the process group; its agent must remain in it.
shared_budget_group = bool(os.environ.get('TILECAS_RANDOM_BUDGET_OWNER'))
child=subprocess.Popen(sys.argv[3:],env=dict(os.environ,CODEX_HOME=str(private)),
                       start_new_session=not shared_budget_group)
def stop(signum,frame):
    try:
        if shared_budget_group:
            child.terminate()
        else:
            os.killpg(child.pid,signal.SIGTERM)
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if shared_budget_group:
            child.kill()
        else:
            os.killpg(child.pid,signal.SIGKILL)
        child.wait()
    except ProcessLookupError:
        pass
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
try:
    code=child.wait()
finally:
    auth=private/'auth.json'
    if auth.exists() and not auth.is_symlink():
        tmp=canonical.with_name('auth.rocm-refresh.tmp')
        fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'wb') as out: out.write(auth.read_bytes())
        tmp.replace(canonical)
        auth.unlink(); auth.symlink_to(canonical)
raise SystemExit(code)
