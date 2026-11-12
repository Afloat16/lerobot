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


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("with_environment_state", [False, True])
@pytest.mark.parametrize("check", ["reported_norm", "optimizer_update"])
def test_actor_clipping_uses_only_optimizer_parameters(shared, with_environment_state, check):
    torch.manual_seed(71)
    features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))}
    if with_environment_state:
        features[OBS_ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(3,))
    policy_config = GaussianActorConfig(
        input_features=features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        shared_encoder=shared,
        latent_dim=8,
        actor_network_kwargs=ActorNetworkConfig(hidden_dims=[16]),
        device="cpu",
    )
    policy = GaussianActorPolicy(policy_config)
    config = SACAlgorithmConfig(
        policy_config=policy_config,
        critic_network_kwargs=CriticNetworkConfig(hidden_dims=[16]),
        grad_clip_norm=0.05,
    )
    algorithm = SACAlgorithm(policy, config)
    optimizer = algorithm.make_optimizers_and_scheduler()["actor"]
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    encoder_ids = {id(parameter) for parameter in policy.actor.encoder.parameters()}
    if shared:
        assert encoder_ids.isdisjoint({id(parameter) for parameter in parameters})

    reference = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    reference_optimizer = torch.optim.Adam(reference, **optimizer.defaults)
    raw_gradients = {}
    handles = [
        parameter.register_hook(lambda gradient, index=index: raw_gradients.update({index: gradient.clone()}))
        for index, parameter in enumerate(parameters)
    ]
    norms = []
    try:
        for reward_scale in (1000.0, 0.01, 5.0, 200.0):
            batch = {
                ACTION: torch.randn(5, 2),
                "state": {key: torch.randn(5, *feature.shape) for key, feature in features.items()},
                "next_state": {key: torch.randn(5, *feature.shape) for key, feature in features.items()},
                "reward": reward_scale * torch.randn(5),
                "done": torch.zeros(5),
            }
            raw_gradients.clear()
            stats = algorithm.update(iter([batch]))
            assert len(raw_gradients) == len(parameters)
            expected_norm = torch.linalg.vector_norm(
                torch.stack([gradient.norm() for gradient in raw_gradients.values()])
            )
            norms.append((stats.grad_norms["actor"], expected_norm))
            for index, parameter in enumerate(reference):
                parameter.grad = raw_gradients[index]
            torch.nn.utils.clip_grad_norm_(reference, config.grad_clip_norm)
            reference_optimizer.step()
    finally:
        for handle in handles:
            handle.remove()

    if check == "reported_norm":
        for actual_norm, expected_norm in norms:
            torch.testing.assert_close(torch.tensor(actual_norm), expected_norm)
    else:
        for actual, expected in zip(parameters, reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(
                optimizer.state[actual]["exp_avg"], reference_optimizer.state[expected]["exp_avg"]
            )
