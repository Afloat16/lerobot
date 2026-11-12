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

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
    create_transition,
)
from lerobot.utils.constants import ACTION, OBS_STATE


@pytest.mark.parametrize("step_type", [NormalizerProcessorStep, UnnormalizerProcessorStep])
@pytest.mark.parametrize("mode", list(NormalizationMode))
@pytest.mark.parametrize("dtype", [torch.int8, torch.int64, torch.uint8])
@pytest.mark.parametrize("feature_type", [FeatureType.STATE, FeatureType.ACTION])
def test_integer_inputs_preserve_fractional_stats(step_type, mode, dtype, feature_type):
    key = ACTION if feature_type == FeatureType.ACTION else OBS_STATE
    stats = {
        "mean": [0.5, 1.5],
        "std": [0.5, 2.5],
        "min": [0.5, 1.5],
        "max": [2.5, 4.5],
        "q01": [0.5, 1.5],
        "q99": [2.5, 4.5],
        "q10": [0.5, 1.5],
        "q90": [2.5, 4.5],
    }
    step = step_type(
        features={key: PolicyFeature(feature_type, (2,))},
        norm_map={feature_type: mode},
        stats={key: stats},
    )
    values = torch.tensor([[1, 2], [2, 3]], dtype=dtype)
    original = values.clone()
    transition = create_transition(
        action=values if feature_type == FeatureType.ACTION else None,
        observation={key: values} if feature_type == FeatureType.STATE else None,
    )
    moved = DeviceProcessorStep(device="cpu", float_dtype="float32")(transition)
    result = step(moved)
    actual = result[ACTION] if feature_type == FeatureType.ACTION else result["observation"][key]
    if mode == NormalizationMode.IDENTITY:
        torch.testing.assert_close(actual, original)
    else:
        expected = values.float()
        inverse = step_type == UnnormalizerProcessorStep
        if mode == NormalizationMode.MEAN_STD:
            mean, std = torch.tensor(stats["mean"]), torch.tensor(stats["std"])
            expected = expected * std + mean if inverse else (expected - mean) / (std + step.eps)
        else:
            lower, width = torch.tensor([0.5, 1.5]), torch.tensor([2.0, 3.0])
            expected = (expected + 1) / 2 * width + lower if inverse else 2 * (expected - lower) / width - 1
        torch.testing.assert_close(actual, expected)
        assert step.dtype == torch.float32
        for name, value in stats.items():
            torch.testing.assert_close(step._tensor_stats[key][name], torch.tensor(value))
    torch.testing.assert_close(values, original)


@pytest.mark.parametrize(
    "mode", [NormalizationMode.MIN_MAX, NormalizationMode.QUANTILES, NormalizationMode.QUANTILE10]
)
@pytest.mark.parametrize("dtype", [torch.int8, torch.int64, torch.uint8])
def test_integer_constant_bounds_use_floating_epsilon(mode, dtype):
    step = NormalizerProcessorStep(
        features={OBS_STATE: PolicyFeature(FeatureType.STATE, (1,))},
        norm_map={FeatureType.STATE: mode},
        stats={OBS_STATE: {name: [3.0] for name in ("min", "max", "q01", "q99", "q10", "q90")}},
    )
    actual = step(create_transition(observation={OBS_STATE: torch.tensor([3], dtype=dtype)}))
    torch.testing.assert_close(actual["observation"][OBS_STATE], torch.tensor([-1.0]))
