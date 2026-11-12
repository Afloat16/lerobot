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

import numpy as np
import pytest
import torch

pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")
pytest.importorskip("pyarrow", reason="Parquet support is required (install lerobot[dataset])")

import pandas as pd

from lerobot.rewards.sarm.rabc import RABCWeights

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize("chunk_size", [1, 2, 50])
@pytest.mark.parametrize("shuffle", [False, True])
def test_rabc_initialization_preserves_deltas_and_weights(tmp_path, chunk_size, shuffle):
    rows = [
        (0, 7, 0.0),
        (1, 7, 0.1),
        (2, 7, 0.5),
        (3, 7, 0.6),
        (20, 42, 0.9),
        (21, 42, 0.2),
        (22, 42, 0.5),
        (40, 99, np.nan),
        (41, 99, 0.4),
    ]
    frame = pd.DataFrame(rows, columns=["index", "episode_index", "progress_sparse"])
    if shuffle:
        frame = frame.sample(frac=1, random_state=17)
    path = tmp_path / "progress.parquet"
    frame.to_parquet(path, index=False)
    weighter = RABCWeights(path, chunk_size=chunk_size, device=torch.device("cpu"), kappa=0.2)

    progress = {idx: value for idx, _, value in rows if not np.isnan(value)}
    episodes = {idx: episode for idx, episode, _ in rows}
    boundaries = {7: {"start": 0, "end": 4}, 42: {"start": 20, "end": 23}, 99: {"start": 40, "end": 42}}
    assert weighter.progress_lookup == progress
    assert weighter.episode_lookup == episodes
    assert weighter.episode_boundaries == boundaries
    deltas = {
        idx: progress[min(idx + chunk_size, boundaries[episodes[idx]]["end"] - 1)] - value
        for idx, value in progress.items()
    }
    mean = max(float(np.mean(list(deltas.values()))), 0.0)
    std = max(float(np.std(list(deltas.values()))), weighter.epsilon)
    assert weighter.delta_mean == pytest.approx(mean)
    assert weighter.delta_std == pytest.approx(std)

    indices = [idx for idx, _, _ in rows] + [1000]
    expected = []
    for idx in indices:
        delta = deltas.get(idx)
        if delta is None or delta > 0.2:
            expected.append(1.0)
        elif delta < 0:
            expected.append(0.0)
        else:
            expected.append(float(np.clip((delta - (mean - 2 * std)) / (4 * std + weighter.epsilon), 0, 1)))
    expected = torch.tensor(expected)
    expected *= len(expected) / (expected.sum() + weighter.epsilon)
    actual, _ = weighter.compute_batch_weights({"index": torch.tensor(indices)})
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("offset", [2**53, 2**53 + 1])
def test_rabc_preserves_integer_parquet_indices(tmp_path, offset):
    path = tmp_path / "progress.parquet"
    pd.DataFrame(
        {
            "index": [offset, offset + 1, offset + 2],
            "episode_index": [offset + 1] * 3,
            "progress_sparse": [0.0, 0.4, 0.8],
        }
    ).to_parquet(path, index=False)
    weighter = RABCWeights(path, chunk_size=1, device=torch.device("cpu"))
    assert weighter.progress_lookup == {offset: 0.0, offset + 1: 0.4, offset + 2: 0.8}
    assert weighter.episode_lookup == dict.fromkeys(range(offset, offset + 3), offset + 1)
    assert weighter.episode_boundaries == {offset + 1: {"start": offset, "end": offset + 3}}
