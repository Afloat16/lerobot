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
    CriticNetworkConfig,
    GaussianActorConfig,
)
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def make_config(feature, shared):
    return GaussianActorConfig(
        input_features={
            feature: PolicyFeature(
                type=FeatureType.ENV if feature == OBS_ENV_STATE else FeatureType.STATE, shape=(4,)
            )
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        shared_encoder=shared,
        latent_dim=8,
        actor_network_kwargs=ActorNetworkConfig(hidden_dims=[16]),
        device="cpu",
    )


@pytest.mark.parametrize("feature", [OBS_ENV_STATE, OBS_STATE])
@pytest.mark.parametrize("shared", [False, True])
def test_state_vector_actor_forward(feature, shared):
    config = make_config(feature, shared)
    config.validate_features()
    policy = GaussianActorPolicy(config)
    result = policy({feature: torch.randn(3, 4)})
    assert result["action"].shape == (3, 2)
    assert result["log_prob"].shape == (3,)
    assert all(torch.isfinite(value).all() for value in result.values())
    (-result["log_prob"].mean()).backward()
    assert policy.actor.mean_layer.weight.grad is not None


@pytest.mark.parametrize("shared", [False, True])
def test_environment_state_sac_optimizer_step(shared):
    torch.manual_seed(41)
    config = make_config(OBS_ENV_STATE, shared)
    policy = GaussianActorPolicy(config)
    algorithm = SACAlgorithm(
        policy,
        SACAlgorithmConfig(policy_config=config, critic_network_kwargs=CriticNetworkConfig(hidden_dims=[16])),
    )
    algorithm.make_optimizers_and_scheduler()
    before = policy.actor.mean_layer.weight.detach().clone()
    batch = {
        ACTION: torch.randn(3, 2),
        "state": {OBS_ENV_STATE: torch.randn(3, 4)},
        "next_state": {OBS_ENV_STATE: torch.randn(3, 4)},
        "reward": torch.randn(3),
        "done": torch.zeros(3),
    }
    stats = algorithm.update(iter([batch]))
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.losses.values())
    assert not torch.equal(before, policy.actor.mean_layer.weight)


def test_observation_free_actor_is_still_rejected():
    config = make_config(OBS_STATE, shared=False)
    config.input_features = {}
    with pytest.raises(ValueError, match="provide"):
        config.validate_features()
