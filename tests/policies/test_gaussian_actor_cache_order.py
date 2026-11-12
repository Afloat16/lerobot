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

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import (
    ActorNetworkConfig,
    GaussianActorConfig,
)
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
from lerobot.utils.constants import ACTION


def make_policy_and_cache(cameras, shared=True):
    torch.manual_seed(17)
    image_keys = [f"observation.images.camera_{index}" for index in range(cameras)]
    config = GaussianActorConfig(
        input_features={key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)) for key in image_keys},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        image_encoder_hidden_dim=4,
        image_embedding_pooling_dim=2,
        latent_dim=8,
        actor_network_kwargs=ActorNetworkConfig(hidden_dims=[16]),
        shared_encoder=shared,
        freeze_vision_encoder=False,
        device="cpu",
    )
    policy = GaussianActorPolicy(config).eval()
    observations = {key: torch.randn(2, 3, 64, 64) for key in image_keys}
    cache = policy.actor.encoder.get_cached_image_features(observations)
    return policy, observations, cache


@pytest.mark.parametrize("cameras", [1, 2, 3])
@pytest.mark.parametrize("detach", [False, True])
def test_cached_encoding_does_not_depend_on_mapping_order(cameras, detach):
    policy, observations, cache = make_policy_and_cache(cameras)
    reversed_cache = dict(reversed(list(cache.items())))
    expected = policy.actor.encoder(observations, cache=cache, detach=detach)
    actual = policy.actor.encoder(observations, cache=reversed_cache, detach=detach)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("shared", [False, True])
def test_policy_forward_keeps_camera_slots_when_cache_is_reordered(cameras, shared):
    policy, observations, cache = make_policy_and_cache(cameras, shared)
    torch.manual_seed(29)
    expected = policy({"state": observations, "observation_feature": cache})
    torch.manual_seed(29)
    actual = policy({"state": observations, "observation_feature": dict(reversed(list(cache.items())))})
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
