"""Measure the real relative-action statistics API in a fresh native process."""

import argparse
import json
import resource
import threading
import time
import weakref

import numpy as np

import lerobot.datasets.compute_stats as compute_stats
from lerobot.utils.constants import ACTION, OBS_STATE


parser = argparse.ArgumentParser()
parser.add_argument('--starts', type=int, default=800_000)
parser.add_argument('--workers', type=int, default=2)
args = parser.parse_args()
chunk_size = 32
indices = np.arange(args.starts + chunk_size - 1)
actions = np.column_stack([indices % 101, indices % 23, indices % 37, indices % 11]).astype(np.float32)
states = np.column_stack([indices % 7, indices % 5, indices % 3, indices % 13]).astype(np.float32)
dataset = {ACTION: actions, OBS_STATE: states, 'episode_index': np.zeros(len(indices), dtype=np.int64)}
features = {ACTION: {'shape': (4,), 'names': ['joint_0', 'joint_1', 'joint_2', 'gripper']}}
original = compute_stats._compute_relative_chunk_batch
live = []
peak_live = 0
peak_bytes = 0
calls = 0
lock = threading.Lock()


def track(*positional, **keywords):
    global peak_live, peak_bytes, calls
    array = original(*positional, **keywords)
    with lock:
        live[:] = [(ref, size) for ref, size in live if ref() is not None]
        live.append((weakref.ref(array), array.nbytes))
        peak_live = max(peak_live, len(live))
        peak_bytes = max(peak_bytes, sum(size for _, size in live))
        calls += 1
    return array


compute_stats._compute_relative_chunk_batch = track
start_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
started = time.perf_counter()
statistics = compute_stats.compute_relative_action_stats(
    dataset, features, chunk_size, exclude_joints=['gripper'], num_workers=args.workers
)
elapsed = time.perf_counter() - started
print(json.dumps({
    'starts': args.starts, 'chunk_size': chunk_size, 'action_dim': 4, 'workers': args.workers,
    'calls': calls, 'peak_live_expanded_arrays': peak_live, 'peak_live_expanded_bytes': peak_bytes,
    'start_peak_rss_kib': start_rss_kib,
    'final_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'seconds': elapsed, 'numpy': np.__version__,
    'statistics': {key: value.tolist() for key, value in statistics.items()},
}, sort_keys=True))
