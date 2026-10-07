"""Per-session Unix identities for containers that prohibit mount namespaces.

The controller remains root. Optimizers get neutral private directories and
cannot traverse the controller repository, root's Codex history or other sessions.
"""
import json
import uuid
import os
from pathlib import Path
import pwd
import shutil
import subprocess

import agent_driver

BASE = Path('/var/lib/ako')


def _under_workspaces(path):
    """True when this path already sits directly under the workspace root.

    Compared by inode: the workspace root is reachable by more than one
    path, and a string comparison would allocate a second workspace for a
    directory that already is one.
    """
    try:
        return os.path.samefile(Path(path).parent, BASE / 'workspaces')
    except OSError:
        return False


def identity(directory):
    directory = Path(directory).resolve()
    token = directory.name if _under_workspaces(directory) else uuid.uuid4().hex[:16]
    return 'ako' + token[:12], BASE / 'workspaces' / token


def prepare_directory(directory):
    directory = Path(directory)
    name, target = identity(directory)
    # Idempotent. Two callers prepare the same session -- the bench wrapper
    # and the agent launcher -- and a second move would leave a link chain
    # whose real directory belongs to the first caller's account, so the
    # ownership and the ACLs land on the link instead of the workspace.
    if not directory.is_symlink() and not _under_workspaces(directory.resolve()):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError('private workspace already exists')
        shutil.move(str(directory), str(target))
        directory.symlink_to(target, target_is_directory=True)
    target = Path(target).resolve()
    target.chmod(0o700)
    return target


def command(api, directory, argv):
    if os.geteuid() != 0:
        raise RuntimeError('Ascend session isolation requires the root controller')
    original = Path(directory)
    target = prepare_directory(original)
    name, _ = identity(target)
    home = BASE / 'homes' / name
    try:
        account = pwd.getpwnam(name)
    except KeyError:
        subprocess.run(['useradd', '--system', '--create-home', '--home-dir', str(home), '--shell', '/bin/bash', name], check=True)
        account = pwd.getpwnam(name)
    home.chmod(0o700)
    compiler_tmp = home / '.cache/tmp'
    compiler_tmp.mkdir(parents=True, exist_ok=True)
    # This host's /etc/profile exports root-owned compiler/cache directories.
    # Codex invokes bash -lc, so restore private paths after that login setup.
    (home / '.bash_profile').write_text(
        'source /remote-home/S45149/ascend-migration-20260918/env.sh\n'
        f'export TMPDIR="{compiler_tmp}"\n'
        f'export TILELANG_CACHE_DIR="{home}/.cache/tilelang"\n'
        f'export XDG_CACHE_HOME="{home}/.cache"\n'
        f'export TORCH_EXTENSIONS_DIR="{target}/.cache/extensions"\n')
    # Traverse only known approved paths under /root; do not expose root's
    # credentials, experiment source, or result directories to this identity.
    subprocess.run(['setfacl', '-m', f'u:{name}:--x', '/root'], check=True)
    controller = agent_driver.credential_directory()
    for path in (*(Path('/root') / home for home in agent_driver.HOME_DIRS.values()),
                 controller, api.ROOT, api.RUN_ROOT,
                 Path(os.environ.get('KERNELBENCH_SOURCE_REPO', str(api.ROOT))),
                 Path('/root/dsl-native-opt-runs'), Path('/root/.dsl-native-opt')):
        if path.exists():
            subprocess.run(['setfacl', '-m', f'u:{name}:---', str(path)], check=True)
    private = home / agent_driver.HOME_DIR
    private.mkdir(exist_ok=True)
    for filename in agent_driver.CREDENTIALS:
        source = controller / filename
        dest = private / filename
        if source.exists() and not dest.exists():
            shutil.copyfile(source, dest)
            dest.chmod(0o600)
    from ascend_access import configure, protect
    configure(name, target, home, api)
    # Scoped to this run. A fixed name made a later run read whatever an
    # earlier one had left behind, since the copy is skipped when it exists.
    skill = BASE / 'references' / os.environ.get('KERNELBENCH_RUN_ID', 'current') / 'ako4all'
    if skill.exists():
        shutil.rmtree(skill)
    shutil.copytree(api.AKO_SKILL.parent, skill, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    for path in skill.rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644)
    # Driver nodes keep their administrator-defined mode; add the session to
    # existing device groups instead of making devices world-writable.
    import grp
    for gid in {p.stat().st_gid for p in Path('/dev').glob('davinci*')}:
        if gid:
            try:
                group = grp.getgrgid(gid).gr_name
            except KeyError:
                group = f'ako-device-{gid}'
                subprocess.run(['groupadd', '-g', str(gid), group], check=True)
            subprocess.run(['usermod', '-aG', group, name], check=True)
    for root in (target, home):
        # chown -R does not follow a symlinked root, which would leave the
        # real directory owned by root and unreachable by the session.
        subprocess.run(['chown', '-R', f'{account.pw_uid}:{account.pw_gid}',
                        str(Path(root).resolve())], check=True)
    protect(target)
    trusted = subprocess.run(['git', 'config', '--global', '--get-all', 'safe.directory'], capture_output=True, text=True).stdout.splitlines()
    if str(target) not in trusted:
        subprocess.run(['git', 'config', '--global', '--add', 'safe.directory', str(target)], check=True)
    converted = [arg.replace(str(original), str(target)).replace('/tmp/workspace', str(target))
                 .replace(str(api.AKO_SKILL), str(skill / 'SKILL.md')) for arg in argv]
    proxy_environment = []
    # Both optimizer CLIs use the host's managed proxy. Claude may be a native
    # executable, so wrapper text is not a reliable indication of proxy needs.
    # Start it as controller before dropping to the private session identity.
    proxy_bootstrap = Path('/usr/local/bin/mihomo-ensure')
    name_index = next((i for i, arg in enumerate(converted)
                       if arg in agent_driver.DRIVERS), None)
    if name_index is not None:
        converted[name_index] = agent_driver.executable()
    if name_index is not None and proxy_bootstrap.is_file():
        subprocess.run([str(proxy_bootstrap)], check=True, timeout=30)
        # A per-user install sits under a home the session cannot traverse.
        proxy_environment = ['HTTP_PROXY=http://127.0.0.1:7890', 'HTTPS_PROXY=http://127.0.0.1:7890',
                             'NO_PROXY=127.0.0.1,localhost,::1', 'ALL_PROXY=']
    (target / '.session-identity.json').write_text(json.dumps({'name': name, 'home': str(home), 'workspace': str(target)}))
    driver_environment = [f'{key}={value}' for key, value
                          in agent_driver.home_environment(private).items()]
    return ['runuser', '-u', name, '--', 'env', f'HOME={home}', *driver_environment,
            f'TMPDIR={compiler_tmp}',
            f'TILELANG_CACHE_DIR={home}/.cache/tilelang', f'XDG_CACHE_HOME={home}/.cache',
            f'TORCH_EXTENSIONS_DIR={target}/.cache/extensions', *proxy_environment, *converted]
