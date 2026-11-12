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

"""Time RA-BC construction from real Parquet and fingerprint the resulting weights.

Run on each source revision with the same arguments, for example:
python benchmarks/benchmark_rabc_initialization.py --episodes 20000 --frames-per-episode 10
"""

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.rewards.sarm.rabc import RABCWeights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--frames-per-episode", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.episodes < 1 or args.frames_per_episode < 2 or args.repeats < 1:
        parser.error("Use positive episode/repeat counts and at least two frames per episode.")
    indices = np.arange(args.episodes * args.frames_per_episode, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "index": indices,
            "episode_index": indices // args.frames_per_episode,
            "progress_sparse": (indices % args.frames_per_episode) / (args.frames_per_episode - 1),
        }
    )
    elapsed = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "progress.parquet"
        frame.to_parquet(path, index=False)
        for _ in range(args.repeats):
            start = time.perf_counter()
            weighter = RABCWeights(path, chunk_size=4, device=torch.device("cpu"))
            elapsed.append(time.perf_counter() - start)
        weights, _ = weighter.compute_batch_weights({"index": torch.from_numpy(indices)})
        print(
            json.dumps(
                {
                    "rows": len(indices),
                    "episodes": args.episodes,
                    "construction_seconds": elapsed,
                    "median_seconds": statistics.median(elapsed),
                    "weights_sha256": hashlib.sha256(weights.numpy().tobytes()).hexdigest(),
                    "stats": weighter.get_stats(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
