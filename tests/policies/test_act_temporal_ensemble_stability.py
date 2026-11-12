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

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


@pytest.mark.parametrize("coefficient", [-100.0, -1.0, -0.1, 0.0, 0.01, 1.0, 100.0])
@pytest.mark.parametrize("chunk_size", [1, 10, 100])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_temporal_ensemble_matches_stable_offline_average(coefficient, chunk_size, dtype):
    generator = torch.Generator().manual_seed(17)
    ensembler = ACTTemporalEnsembler(coefficient, chunk_size)
    history = []
    tolerance = {torch.float16: 0.005, torch.bfloat16: 0.03, torch.float32: 1e-5, torch.float64: 1e-5}

    for step in range(chunk_size + 7):
        actions = torch.randn(2, chunk_size, 3, generator=generator).to(dtype)
        history.append(actions.clone())
        first = max(0, step - chunk_size + 1)
        predictions = torch.stack([history[i][:, step - i] for i in range(first, step + 1)], dim=1)
        weights = (-coefficient * torch.arange(len(predictions[0]), dtype=torch.float64)).softmax(0)
        expected = (predictions.double() * weights[None, :, None]).sum(1)

        actual = ensembler.update(actions)

        assert actual.dtype == dtype
        torch.testing.assert_close(actual.double(), expected, atol=tolerance[dtype], rtol=tolerance[dtype])
        torch.testing.assert_close(actions, history[-1], rtol=0, atol=0)

    ensembler.reset()
    torch.testing.assert_close(ensembler.update(actions), actions[:, 0], rtol=0, atol=0)


def test_act_policy_keeps_negative_coefficient_rollout_finite():
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        chunk_size=100,
        n_action_steps=1,
        temporal_ensemble_coeff=-1.0,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        device="cpu",
    )
    policy = ACTPolicy(config).eval()
    generator = torch.Generator().manual_seed(29)
    with torch.no_grad():
        for _ in range(102):
            batch = {
                OBS_STATE: torch.randn(2, 4, generator=generator),
                OBS_ENV_STATE: torch.randn(2, 3, generator=generator),
            }
            action = policy.select_action(batch)
            assert action.shape == (2, 2)
            assert torch.isfinite(action).all()
