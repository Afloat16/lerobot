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

"""Train TDMPC with the observation modalities accepted by its encoder."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_STATE, REWARD


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("extra_modality", [None, OBS_ENV_STATE, OBS_IMAGE])
def test_forward_backward_with_robot_state(batch_size: int, extra_modality: str | None) -> None:
    torch.manual_seed(1234)
    horizon = 2
    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,))}
    if extra_modality == OBS_ENV_STATE:
        input_features[OBS_ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(4,))
    elif extra_modality == OBS_IMAGE:
        input_features[OBS_IMAGE] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84))

    config = TDMPCConfig(
        device="cpu",
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        horizon=horizon,
        state_encoder_hidden_dim=8,
        image_encoder_hidden_dim=4,
        latent_dim=4,
        mlp_dim=8,
        q_ensemble_size=2,
    )
    policy = TDMPCPolicy(config).train()
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=1e-3)
    batch = {
        OBS_STATE: torch.randn(batch_size, horizon + 1, 3),
        ACTION: torch.rand(batch_size, horizon, 2) * 2 - 1,
        REWARD: torch.randn(batch_size, horizon),
        "index": torch.arange(batch_size),
        f"{OBS_STATE}_is_pad": torch.zeros(batch_size, horizon + 1, dtype=torch.bool),
        f"{ACTION}_is_pad": torch.zeros(batch_size, horizon, dtype=torch.bool),
        f"{REWARD}_is_pad": torch.zeros(batch_size, horizon, dtype=torch.bool),
    }
    if extra_modality == OBS_ENV_STATE:
        batch[OBS_ENV_STATE] = torch.randn(batch_size, horizon + 1, 4)
    elif extra_modality == OBS_IMAGE:
        batch[OBS_IMAGE] = torch.rand(batch_size, horizon + 1, 3, 84, 84)
    original_batch = {key: value.clone() for key, value in batch.items()}
    original_parameters = [parameter.detach().clone() for parameter in policy.model.parameters()]

    loss, _ = policy.forward(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in policy.model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    assert all(parameter.grad is None for parameter in policy.model_target.parameters())

    optimizer.step()
    policy.update()
    assert any(
        not torch.equal(before, after)
        for before, after in zip(original_parameters, policy.model.parameters(), strict=True)
    )
    for key, value in original_batch.items():
        torch.testing.assert_close(batch[key], value)
