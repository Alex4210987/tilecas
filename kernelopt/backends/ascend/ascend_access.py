"""Read-only platform references and private same-experiment conversation storage."""
from pathlib import Path
import os
import shutil
import subprocess

READ_ONLY_HINTS = "HINTS.md, reference.py and scripts/ are read-only.\n\n"


def configure(name, target, home, api):
    def acl(path, rule, recursive=False):
        if Path(path).exists():
            subprocess.run(['setfacl', *(['-R'] if recursive else []), '-m',
                            f'u:{name}:{rule}', str(path)], check=True)
    # Grant platform references without reopening sibling workspaces, run
    # histories or the controller checkout denied by ascend_isolation.
    acl('/root', '--x')
    roots = [Path(p) for p in ('/root/dsl_codegen_poc', '/root/tilelang-ascend',
                              '/root/catlass_latest')]
    for path in roots:
        acl(path.resolve(), 'r-X', True)
    acl('/var/lib/ako/workspaces', '--x')
    acl('/var/lib/ako/references', '--x')
    refs = Path('/var/lib/ako/references') / os.environ.get('KERNELBENCH_RUN_ID', api.ROOT.name) / 'knowledge'
    if not refs.exists():
        shutil.copytree(api.ROOT/'knowledge', refs)
        for path in refs.rglob('*'):
            path.chmod(0o755 if path.is_dir() else 0o644)
    visible = target/'knowledge'
    if not visible.exists():
        visible.symlink_to(refs, target_is_directory=True)
    # Dashboard account can read experiment outputs, not optimizer credentials.
    subprocess.run(['setfacl', '-R', '-m', 'u:S45149:r-X', str(target)], check=True)
    for directory in [target, *(p for p in target.rglob('*') if p.is_dir() and not p.is_symlink())]:
        subprocess.run(['setfacl', '-m', 'd:u:S45149:r-x', str(directory)], check=True)
    import agent_driver
    sessions = target/agent_driver.PRIVATE
    sessions.mkdir(parents=True, exist_ok=True)
    link = home/agent_driver.TRANSCRIPTS
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_dir() and not link.is_symlink():
        shutil.copytree(link, sessions, dirs_exist_ok=True)
        shutil.rmtree(link)
    if not link.exists():
        link.symlink_to(sessions, target_is_directory=True)

def protect(target):
    hints = target/'HINTS.md'
    if hints.is_file():
        text = hints.read_text()
        if not text.startswith(READ_ONLY_HINTS):
            hints.write_text(READ_ONLY_HINTS + text)
    for top in (target/'scripts', target/'reference.py', target/'HINTS.md'):
        if not top.exists():
            continue
        paths = [top, *top.rglob('*')] if top.is_dir() else [top]
        for path in paths:
            if path.is_symlink():
                continue
            os.chown(path, 0, 0)
            path.chmod(0o755 if path.is_dir() or path.suffix == '.sh' else 0o644)
