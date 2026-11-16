#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import weakref

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import lerobot.datasets.compute_stats as compute_stats
from lerobot.utils.constants import ACTION, OBS_STATE

pytestmark = pytest.mark.dataset


def _dataset(num_frames: int) -> tuple[dict[str, np.ndarray], dict]:
    indices = np.arange(num_frames)
    dataset = {
        ACTION: np.column_stack((indices % 97, indices % 31)).astype(np.float32),
        OBS_STATE: np.column_stack((indices % 7, indices % 11)).astype(np.float32),
        "episode_index": np.zeros(num_frames, dtype=np.int64),
    }
    features = {ACTION: {"shape": (2,), "names": ["joint", "gripper"]}}
    return dataset, features


@pytest.mark.parametrize("num_workers", [2, 3])
def test_relative_stats_release_completed_chunk_arrays(
    monkeypatch: pytest.MonkeyPatch, num_workers: int
) -> None:
    # Ten expanded batches: retaining completed futures must not retain all ten arrays.
    dataset, features = _dataset(500_003)
    expected = compute_stats.compute_relative_action_stats(
        dataset, features, chunk_size=4, exclude_joints=["gripper"]
    )
    original = compute_stats._compute_relative_chunk_batch
    live: list[weakref.ReferenceType[np.ndarray]] = []
    peak_live = 0
    lock = threading.Lock()

    def track_chunks(*args, **kwargs) -> np.ndarray:
        nonlocal peak_live
        result = original(*args, **kwargs)
        with lock:
            live[:] = [ref for ref in live if ref() is not None]
            live.append(weakref.ref(result))
            peak_live = max(peak_live, len(live))
        return result

    monkeypatch.setattr(compute_stats, "_compute_relative_chunk_batch", track_chunks)
    actual = compute_stats.compute_relative_action_stats(
        dataset, features, chunk_size=4, exclude_joints=["gripper"], num_workers=num_workers
    )
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
    assert all(ref() is None for ref in live)
    assert peak_live <= num_workers + 1


@pytest.mark.parametrize("num_workers", [0, 1, 2, 3])
def test_relative_stats_preserve_all_samples_and_inputs(num_workers: int) -> None:
    dataset, features = _dataset(12)
    dataset["episode_index"][6:] = 1
    originals = {key: value.copy() for key, value in dataset.items()}
    expected_chunks = []
    for start in [0, 1, 2, 3, 6, 7, 8, 9]:
        for offset in range(3):
            action = dataset[ACTION][start + offset].copy()
            action[0] -= dataset[OBS_STATE][start, 0]
            expected_chunks.append(action)
    expected_stats = compute_stats.RunningQuantileStats()
    expected_stats.update(np.stack(expected_chunks))
    expected = expected_stats.get_statistics()

    actual = compute_stats.compute_relative_action_stats(
        dataset, features, chunk_size=3, exclude_joints=["gripper"], num_workers=num_workers
    )
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in dataset:
        np.testing.assert_array_equal(dataset[key], originals[key])


def test_relative_stats_propagate_worker_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, features = _dataset(100_003)
    error = RuntimeError("relative chunk computation failed")

    def fail(*args, **kwargs) -> np.ndarray:
        raise error

    monkeypatch.setattr(compute_stats, "_compute_relative_chunk_batch", fail)
    with pytest.raises(RuntimeError) as caught:
        compute_stats.compute_relative_action_stats(dataset, features, chunk_size=4, num_workers=2)
    assert caught.value is error
