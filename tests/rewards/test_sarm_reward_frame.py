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

from lerobot.rewards.sarm.configuration_sarm import SARMConfig
from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel


def make_model_and_inputs(n_obs_steps, single=False):
    torch.manual_seed(31)
    model = SARMRewardModel(
        SARMConfig(
            annotation_mode="dual",
            num_sparse_stages=2,
            num_dense_stages=3,
            n_obs_steps=n_obs_steps,
            max_rewind_steps=0,
            frame_gap=10,
            image_dim=5,
            text_dim=6,
            hidden_dim=8,
            max_state_dim=3,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            device="cpu",
        )
    ).eval()
    inputs = {
        "text_embeddings": torch.randn(2, 6),
        "video_embeddings": torch.randn(2, n_obs_steps + 1, 5),
        "state_features": torch.randn(2, n_obs_steps + 1, 3),
    }
    if single:
        inputs = {key: value[0] for key, value in inputs.items()}
    return model, inputs


@pytest.mark.parametrize("n_obs_steps", [2, 4, 8])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("head_mode", ["sparse", "dense"])
def test_default_reward_uses_the_zero_delta_frame(n_obs_steps, single, head_mode):
    model, inputs = make_model_and_inputs(n_obs_steps, single)
    zero_delta_index = model.config.observation_delta_indices.index(0)
    all_rewards = model.calculate_rewards(**inputs, head_mode=head_mode, return_all_frames=True)
    selected = model.calculate_rewards(**inputs, head_mode=head_mode)
    np.testing.assert_allclose(selected, all_rewards[..., zero_delta_index], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("n_obs_steps", [2, 4, 8])
@pytest.mark.parametrize("single", [False, True])
def test_compute_reward_matches_current_frame(n_obs_steps, single):
    model, inputs = make_model_and_inputs(n_obs_steps, single)
    all_rewards = model.calculate_rewards(**inputs, return_all_frames=True)
    batch = {
        "video_features": inputs["video_embeddings"],
        "text_features": inputs["text_embeddings"],
        "state_features": inputs["state_features"],
    }
    reward = model.compute_reward(batch)
    zero_delta_index = model.config.observation_delta_indices.index(0)
    np.testing.assert_allclose(reward, all_rewards[..., zero_delta_index], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("frame_index", [0, 2, 4, -1])
def test_explicit_frame_selection_is_preserved(single, frame_index):
    model, inputs = make_model_and_inputs(4, single)
    all_rewards = model.calculate_rewards(**inputs, return_all_frames=True)
    selected = model.calculate_rewards(**inputs, frame_index=frame_index)
    np.testing.assert_allclose(selected, all_rewards[..., frame_index], rtol=1e-6, atol=1e-7)
