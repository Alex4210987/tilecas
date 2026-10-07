"""Local candidate snapshots and exact complete-tree restoration, without hashes."""
import argparse
import json
from pathlib import Path
import shutil
import time


def snapshot(root, name, parent, level, edge):
    tree = root / 'candidate-tree'
    destination = tree / name
    if destination.exists():
        raise ValueError('Candidate node already exists')
    if parent and not (tree / parent / 'node.json').is_file():
        raise ValueError('Unknown source parent')
    source = root / 'solution'
    if not source.is_dir() or any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Solution must be a directory without external symlinks')
    destination.mkdir(parents=True)
    shutil.copytree(source, destination / 'solution')
    row = dict(node_id=name, parent=parent, level=level, edge_type=edge, created_at=time.time())
    (destination / 'node.json').write_text(json.dumps(row, indent=2))
    return row


def restore(root, name):
    source = root / 'candidate-tree' / name / 'solution'
    if not source.is_dir():
        raise ValueError('Unknown candidate')
    target = root / 'solution'
    # Preserve unrecorded current work before replacing its complete tree.
    backup = root / 'candidate-tree' / ('before-restore-' + str(time.time_ns()))
    backup.mkdir(parents=True)
    if target.exists():
        if target.is_symlink():
            raise ValueError('Refuse a symlinked solution directory')
        target.rename(backup / 'solution')
    shutil.copytree(source, target)
    return dict(restored=name, previous_tree=str(backup.relative_to(root)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['snapshot', 'restore'])
    parser.add_argument('node')
    parser.add_argument('--parent')
    parser.add_argument('--level', choices=['High', 'Low'], default='High')
    parser.add_argument('--edge', choices=['root', 'tuning', 'structural', 'representation'], default='root')
    args = parser.parse_args()
    for value in (args.node, args.parent):
        if value and (Path(value).name != value or value in {'.', '..'}):
            parser.error('Node names must be single path components')
    root = Path.cwd()
    result = (snapshot(root, args.node, args.parent, args.level, args.edge)
              if args.action == 'snapshot' else restore(root, args.node))
    with (root / 'tree-actions.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(action=args.action, **result)) + '\n')
    print(json.dumps(result))
