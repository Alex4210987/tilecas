#!/usr/bin/env python3
"""Enforce a 120-minute budget on a foreground workflow and its process group.

Example: python scripts/run_with_paper_budget.py --record /tmp/run-budget.json -- python foreground_workflow.py ...
The command must remain in the foreground and must not detach work or submit remote jobs.
For a remote experiment, run this supervisor on the accelerator host around its foreground worker.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args();command=args.command
    if command[:1]==['--']:command=command[1:]
    if not command:p.error('A foreground workflow command is required')
    args.record.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents overwriting an earlier execution record.
    with args.record.open('x') as f:
        record=dict(command=command,total_budget_seconds=7200,started_at_unix=time.time(),status='starting',enforcement='POSIX process group on this host; foreground descendants only')
        json.dump(record,f,indent=2)
    start=time.monotonic();child=None
    def save():args.record.write_text(json.dumps(record,indent=2)+'\n')
    try:
        child=subprocess.Popen(command,start_new_session=True)
        record.update(pid=child.pid,status='running');save()
        try:
            code=child.wait(timeout=max(0,7200-(time.monotonic()-start)))
            record.update(status='complete' if code==0 else 'failed',returncode=code)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
            record.update(status='timeout',returncode=124,timeout_signal='SIGKILL',timeout_sent_at_unix=time.time())
            code=124
    except BaseException as exc:
        if child is not None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
        record.update(status='interrupted_or_launch_failed',error=repr(exc));raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-start,finished_at_unix=time.time());save()
    return code

if __name__=='__main__':raise SystemExit(main())
