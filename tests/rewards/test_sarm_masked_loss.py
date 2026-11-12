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
import torch.nn.functional as F  # noqa: N812

from lerobot.rewards.sarm.configuration_sarm import SARMConfig
from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel, gen_stage_emb


def make_model_and_batch(mode):
    torch.manual_seed(19)
    config = SARMConfig(
        annotation_mode=mode,
        num_sparse_stages=2,
        num_dense_stages=3,
        image_dim=5,
        text_dim=6,
        hidden_dim=8,
        max_state_dim=3,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        device="cpu",
    )
    model = SARMRewardModel(config)
    model.gt_stage_ratio = 1.0
    batch = {
        "video_features": torch.randn(2, 13, 5),
        "text_features": torch.randn(2, 13, 6),
        "state_features": torch.randn(2, 13, 3),
        "lengths": torch.tensor([9, 13]),
        "sparse_targets": torch.rand(2, 13) * config.num_sparse_stages,
    }
    if config.uses_dual_heads:
        batch["dense_targets"] = torch.rand(2, 13) * config.num_dense_stages
    return model, batch


@pytest.mark.parametrize("mode", ["single_stage", "dense_only", "dual"])
def test_padding_targets_do_not_change_loss(mode):
    model, batch = make_model_and_batch(mode)
    altered = {key: value.clone() for key, value in batch.items()}
    altered["sparse_targets"][0, 9:] = 0.999
    if "dense_targets" in altered:
        altered["dense_targets"][0, 9:] = 2.999
    loss, components = model(batch)
    altered_loss, altered_components = model(altered)
    torch.testing.assert_close(loss, altered_loss)
    assert components == pytest.approx(altered_components)


@pytest.mark.parametrize("mode", ["single_stage", "dense_only", "dual"])
@pytest.mark.parametrize("padded", [False, True])
def test_losses_average_only_valid_frames(mode, padded):
    model, batch = make_model_and_batch(mode)
    if not padded:
        batch["lengths"].fill_(13)
    _, components = model(batch)
    valid = torch.arange(13).unsqueeze(0) < batch["lengths"].unsqueeze(1)
    schemes = ["sparse", "dense"] if model.config.uses_dual_heads else ["sparse"]
    for scheme in schemes:
        num_classes = getattr(model.config, f"num_{scheme}_stages")
        targets = batch[f"{scheme}_targets"]
        inputs = (
            batch["video_features"].unsqueeze(1),
            batch["text_features"],
            batch["state_features"],
            batch["lengths"],
        )
        stage = model.stage_model(*inputs, scheme=scheme)
        tau = model.subtask_model(*inputs, gen_stage_emb(num_classes, targets), scheme=scheme)
        expected_stage = F.cross_entropy(stage[valid], targets[valid].floor().long())
        expected_tau = F.mse_loss(tau[valid], targets[valid].remainder(1))
        assert components[f"{scheme}_stage_loss"] == pytest.approx(expected_stage.item())
        assert components[f"{scheme}_subtask_loss"] == pytest.approx(expected_tau.item())


@pytest.mark.parametrize("mode", ["single_stage", "dense_only", "dual"])
def test_padding_has_no_input_gradient(mode):
    model, batch = make_model_and_batch(mode)
    feature_keys = ["video_features", "text_features", "state_features"]
    for key in feature_keys:
        batch[key].requires_grad_()
    loss, _ = model(batch)
    loss.backward()
    for key in feature_keys:
        gradient = batch[key].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient[0, 9:]) == 0, key
        assert gradient[0, :9].abs().sum() > 0, key
