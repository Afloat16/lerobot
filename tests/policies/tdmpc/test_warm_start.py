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

"""Exercise warm starts through the public action queue on CPU."""

from collections.abc import Callable

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def make_policy(n_action_steps: int, n_action_repeats: int, horizon: int = 5) -> TDMPCPolicy:
    config = TDMPCConfig(
        device="cpu",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        n_action_steps=n_action_steps,
        n_action_repeats=n_action_repeats,
        horizon=horizon,
        state_encoder_hidden_dim=8,
        latent_dim=4,
        mlp_dim=8,
        q_ensemble_size=2,
        n_gaussian_samples=4,
        n_pi_samples=0,
        n_elites=2,
        cem_iterations=1,
    )
    return TDMPCPolicy(config).eval()


def controlled_proposals(
    monkeypatch: pytest.MonkeyPatch, policy: TDMPCPolicy, batch_size: int
) -> tuple[torch.Tensor, Callable[[], int]]:
    # All proposals have the same first trajectory. Subsequent zero-noise proposals expose the
    # next warm start through the actual CEM loop, dynamics/value networks and selected action.
    # No planner, action queue, learned model, or previous-mean state is replaced.
    initial_plan = torch.linspace(
        0.05, 0.75, policy.config.horizon * batch_size * 2
    ).reshape(policy.config.horizon, batch_size, 2)
    calls = 0

    def sample_proposals(*shape: int, device: torch.device) -> torch.Tensor:
        nonlocal calls
        assert shape == (policy.config.horizon, policy.config.n_gaussian_samples, batch_size, 2)
        if calls == 0:
            result = initial_plan.unsqueeze(1).expand(shape).clone() / policy.config.max_std
        else:
            result = torch.zeros(shape)
        calls += 1
        return result.to(device=device)

    monkeypatch.setattr(torch, "randn", sample_proposals)
    return initial_plan, lambda: calls


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "horizon,n_action_steps,n_action_repeats",
    [(5, 1, 1), (5, 1, 2), (5, 2, 1), (5, 3, 1), (5, 5, 1), (1, 1, 1), (1, 1, 2)],
)
def test_warm_start_advances_past_executed_actions(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    horizon: int,
    n_action_steps: int,
    n_action_repeats: int,
) -> None:
    policy = make_policy(n_action_steps, n_action_repeats, horizon)
    initial_plan, proposal_calls = controlled_proposals(monkeypatch, policy, batch_size)
    observation = {OBS_STATE: torch.zeros(batch_size, 3)}
    selected_actions = max(n_action_steps, n_action_repeats)

    for index in range(selected_actions):
        action = policy.select_action(dict(observation))
        planned_index = 0 if n_action_repeats > 1 else index
        torch.testing.assert_close(action, initial_plan[planned_index])
        assert proposal_calls() == 1, "Planning must wait until the existing action queue is consumed."

    previous_mean = policy._prev_mean.clone()
    # Repeats execute one planned action several times; they must keep the existing one-step
    # warm start. Chunked execution consumes n_action_steps distinct entries of the old plan.
    remaining = previous_mean[n_action_steps:]
    expected = torch.cat((remaining, torch.zeros(n_action_steps, batch_size, 2)), dim=0)

    next_action = policy.select_action(dict(observation))
    assert proposal_calls() == 2
    torch.testing.assert_close(next_action, expected[0])
    torch.testing.assert_close(policy._prev_mean, expected)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_reset_clears_queued_actions_and_the_warm_start(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    policy = make_policy(n_action_steps=3, n_action_repeats=1)
    initial_plan, proposal_calls = controlled_proposals(monkeypatch, policy, batch_size)
    observation = {OBS_STATE: torch.zeros(batch_size, 3)}

    torch.testing.assert_close(policy.select_action(dict(observation)), initial_plan[0])
    assert len(policy._queues[ACTION]) == 2
    policy.reset()
    assert policy._prev_mean is None
    assert len(policy._queues[ACTION]) == 0

    # The second real plan has zero noise. Any retained queue/mean would yield a nonzero action.
    torch.testing.assert_close(policy.select_action(dict(observation)), torch.zeros(batch_size, 2))
    torch.testing.assert_close(policy._prev_mean, torch.zeros(5, batch_size, 2))
    assert proposal_calls() == 2
