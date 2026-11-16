"""Read every relevant public Git source variant without executing historical code."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path('/workspace/scratch/4bed741d4966')
REPO = ROOT / 'repos/lerobot'
OUTPUT = ROOT / 'audit/lerobot-relative-stats-history.json'


def git(*args):
    return subprocess.run(
        ['git', '-C', str(REPO), *args], check=True, capture_output=True, timeout=1200
    ).stdout


assert git('rev-parse', '--is-shallow-repository').strip() == b'false'
references = {}
for name, url, mappings in [
    ('upstream', 'https://github.com/huggingface/lerobot.git', [
        ('refs/heads/', 'refs/audit-upstream/', '*'),
        ('refs/pull/', 'refs/audit-pr/', '*/head'),
        ('refs/tags/', 'refs/audit-upstream-tags/', '*'),
    ]),
    ('fork', 'https://github.com/Afloat16/lerobot.git', [
        ('refs/heads/', 'refs/audit-fork/', '*'),
        ('refs/tags/', 'refs/audit-fork-tags/', '*'),
    ]),
]:
    advertised = git('ls-remote', '--refs', url, *[source + suffix for source, _, suffix in mappings])
    for line in advertised.decode().splitlines():
        sha, ref = line.split('\t')
        source, destination, _ = next(pair for pair in mappings if ref.startswith(pair[0]))
        target = destination + ref[len(source):]
        actual = git('rev-parse', target).decode().strip()
        if actual != sha:
            raise AssertionError(f'Ref changed during snapshot: {name}:{ref}: {actual} != {sha}')
        references[f'{name}:{ref}'] = sha

print(f'Verified {len(references)} advertised public refs', flush=True)

all_names = set(git('log', '--all', '--full-history', '-m', '--no-renames', '--format=', '--name-only').decode().splitlines())
# Name-independent pickaxe includes function moves into generic files as well as renames.
pickaxe = set(git(
    'log', '--all', '--full-history', '-m', '--no-renames', '--format=', '--name-only',
    '-G', 'compute_relative_action_stats|_compute_relative_chunk_batch|_get_valid_chunk_starts|_prev_mean',
).decode().splitlines())
paths = {p for p in all_names | pickaxe if p.endswith('.py') and (
    'compute_stats' in p or 'tdmpc' in p.lower() or p in pickaxe
)}
rename_log = git('log', '--all', '--full-history', '-m', '--find-renames=20%', '--format=', '--name-status').decode()
renames = []
for line in rename_log.splitlines():
    fields = line.split('\t')
    if len(fields) == 3 and fields[0].startswith('R'):
        renames.append((fields[1], fields[2]))
changed = True
while changed:
    changed = False
    for old, new in renames:
        if (old in paths or new in paths) and not {old, new}.issubset(paths):
            paths.update((old, new))
            changed = True

raw = git('log', '--all', '--full-history', '-m', '--no-renames', '--format=', '--raw', '--no-abbrev', '--', *sorted(paths)).decode()
blobs = {}
for line in raw.splitlines():
    if not line.startswith(':'):
        continue
    header, path = line.split('\t', 1)
    fields = header.split()
    for sha in fields[2:4]:
        if set(sha) != {'0'}:
            blobs.setdefault(sha, set()).add(path)

print(f'Inspecting {len(blobs)} source blobs across {len(paths)} historical paths', flush=True)
ids = sorted(blobs)
result = subprocess.run(
    ['git', '-C', str(REPO), 'cat-file', '--batch'],
    input=''.join(sha + '\n' for sha in ids).encode(), check=True,
    capture_output=True, timeout=1200,
)
data = result.stdout
cursor = 0
variants = {}
index = []
failures = []
for expected in ids:
    newline = data.index(b'\n', cursor)
    sha, kind, size = data[cursor:newline].decode().split()
    assert sha == expected and kind == 'blob'
    start = newline + 1
    end = start + int(size)
    source = data[start:end].decode('utf-8')
    cursor = end + 1
    index.append({'blob': sha, 'paths': sorted(blobs[sha]), 'bytes': int(size)})
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        failures.append({'blob': sha, 'paths': sorted(blobs[sha]), 'error': str(exc)})
        continue
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        if not (
            node.name in {'compute_relative_action_stats', '_compute_relative_chunk_batch', '_get_valid_chunk_starts'}
            or ('_prev_mean' in body and ('tdmpc' in ' '.join(blobs[sha]).lower()))
        ):
            continue
        digest = hashlib.sha256(body.encode()).hexdigest()
        entry = variants.setdefault(digest, {'function': node.name, 'source': body, 'blobs': [], 'paths': []})
        entry['blobs'].append(sha)
        entry['paths'] = sorted(set(entry['paths']).union(blobs[sha]))

report = {
    'complete': not failures,
    'base': git('rev-parse', 'HEAD').decode().strip(),
    'repository': 'huggingface/lerobot', 'fork': 'Afloat16/lerobot',
    'merge_parent_diffs_included': True,
    'name_independent_pickaxe': True,
    'ref_counts': {
        'all_references': len(references),
        'upstream_pr_heads': sum(x.startswith('upstream:refs/pull/') for x in references),
        'fork_branches': sum(x.startswith('fork:refs/heads/') for x in references),
    },
    'reachable_commits': int(git('rev-list', '--all', '--count')),
    'references': references, 'paths': sorted(paths),
    'source_blobs': index, 'function_variants': variants, 'parse_failures': failures,
}
OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: report[key] for key in ['complete', 'ref_counts', 'reachable_commits', 'parse_failures']}))
print(f'Saved {len(variants)} function variants to {OUTPUT}', flush=True)
