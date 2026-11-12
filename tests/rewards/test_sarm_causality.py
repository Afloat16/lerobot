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

from lerobot.rewards.sarm.modeling_sarm import StageTransformer, SubtaskTransformer


def make_model_and_inputs(model_type, num_cameras=1):
    torch.manual_seed(123)
    model = model_type(
        d_model=8,
        vis_emb_dim=5,
        text_emb_dim=6,
        state_dim=3,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
        num_cameras=num_cameras,
    )
    inputs = {
        "img_seq": torch.randn(2, num_cameras, 5, 5),
        "lang_emb": torch.randn(2, 5, 6),
        "state": torch.randn(2, 5, 3),
        "lengths": torch.tensor([5, 5]),
    }
    if model_type is SubtaskTransformer:
        inputs["stage_prior"] = torch.randn(2, 1, 5, 4)
    return model, inputs


def replace_future(inputs, start):
    altered = {key: value.clone() for key, value in inputs.items()}
    for key in altered:
        if key != "lengths":
            altered[key][..., start:, :] = torch.randn_like(altered[key][..., start:, :]) * 9
    return altered


@pytest.mark.parametrize("model_type", [StageTransformer, SubtaskTransformer])
@pytest.mark.parametrize("num_cameras", [1, 2])
@pytest.mark.parametrize("scheme", ["sparse", "dense"])
@pytest.mark.parametrize("training", [False, True])
def test_predictions_do_not_read_future_frames(model_type, num_cameras, scheme, training):
    model, inputs = make_model_and_inputs(model_type, num_cameras)
    model.train(training)
    with torch.set_grad_enabled(training):
        prediction = model(**inputs, scheme=scheme)
        altered = model(**replace_future(inputs, 2), scheme=scheme)
    torch.testing.assert_close(prediction[:, :2], altered[:, :2], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(prediction[:, 2:], altered[:, 2:])


@pytest.mark.parametrize("model_type", [StageTransformer, SubtaskTransformer])
@pytest.mark.parametrize("scheme", ["sparse", "dense"])
def test_full_sequence_matches_prefix(model_type, scheme):
    model, inputs = make_model_and_inputs(model_type, num_cameras=2)
    model.eval()
    prefix = {
        key: torch.tensor([2, 2]) if key == "lengths" else value[..., :2, :] for key, value in inputs.items()
    }
    with torch.no_grad():
        full_output = model(**inputs, scheme=scheme)
        prefix_output = model(**prefix, scheme=scheme)
    torch.testing.assert_close(full_output[:, :2], prefix_output, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("model_type", [StageTransformer, SubtaskTransformer])
def test_past_loss_has_no_gradient_through_future_inputs(model_type):
    model, inputs = make_model_and_inputs(model_type)
    for key, value in inputs.items():
        if key != "lengths":
            value.requires_grad_()
    model(**inputs)[:, :2].sum().backward()
    for key, value in inputs.items():
        if key != "lengths":
            assert value.grad is not None
            assert torch.count_nonzero(value.grad[..., 2:, :]) == 0, key
            assert torch.isfinite(value.grad).all()
    assert inputs["img_seq"].grad[..., :2, :].abs().sum() > 0


@pytest.mark.parametrize("model_type", [StageTransformer, SubtaskTransformer])
@pytest.mark.parametrize("training", [False, True])
def test_padding_remains_hidden(model_type, training):
    model, inputs = make_model_and_inputs(model_type, num_cameras=2)
    model.train(training)
    inputs["lengths"] = torch.tensor([2, 2])
    with torch.set_grad_enabled(training):
        prediction = model(**inputs)
        altered = model(**replace_future(inputs, 2))
    torch.testing.assert_close(prediction[:, :2], altered[:, :2])


@pytest.mark.parametrize("model_type", [StageTransformer, SubtaskTransformer])
def test_broadcast_language_matches_per_frame_language(model_type):
    model, inputs = make_model_and_inputs(model_type)
    model.eval()
    inputs["lang_emb"] = inputs["lang_emb"][:, 0]
    repeated = dict(inputs, lang_emb=inputs["lang_emb"].unsqueeze(1).expand(-1, 5, -1))
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs), model(**repeated))
