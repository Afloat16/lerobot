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

import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from lerobot.policies.cosmos3.processor_cosmos3 import (
    Cosmos3LiberoActionProcessorStep,
    matrix_to_axis_angle,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3], [-1, 2, -0.5], [0, 1, 1]])
@pytest.mark.parametrize("offset", [0.0, 1e-9, -1e-9, 1e-6])
def test_libero_half_turn_preserves_rotation(dtype, axis, offset):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    expected = Rotation.from_rotvec(axis * (math.pi - offset))
    matrix = torch.tensor(expected.as_matrix(), dtype=dtype)
    position = torch.tensor([0.1, -0.2, 0.3], dtype=dtype)
    action = torch.cat((position, matrix[:, 0], matrix[:, 1], torch.tensor([0.75], dtype=dtype)))

    command = Cosmos3LiberoActionProcessorStep().action(action)

    tolerance = 2e-6 if dtype == torch.float32 else 1e-9
    np.testing.assert_allclose(
        Rotation.from_rotvec(command[3:6].double().numpy()).as_matrix(),
        expected.as_matrix(),
        atol=tolerance,
        rtol=0,
    )
    assert command[3:6].norm().item() == pytest.approx(np.linalg.norm(expected.as_rotvec()), abs=tolerance)
    torch.testing.assert_close(command[:3], position)
    assert command[-1].item() == -0.5
    assert command.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("angle", [0.0, 1e-12, 0.1, math.pi / 2, math.pi - 0.01])
def test_matrix_conversion_preserves_ordinary_batched_rotations(dtype, angle):
    axes = np.array([[1.0, 2.0, -3.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    expected = Rotation.from_rotvec(axes * angle).as_matrix()
    # Transposing the batch axes exercises a noncontiguous, multidimensional batch.
    matrices = torch.tensor(expected, dtype=dtype).reshape(2, 2, 3, 3).transpose(0, 1)
    actual = matrix_to_axis_angle(matrices)

    reconstructed = Rotation.from_rotvec(actual.reshape(-1, 3).double().numpy()).as_matrix()
    np.testing.assert_allclose(
        reconstructed,
        matrices.reshape(-1, 3, 3).double().numpy(),
        atol=2e-5 if dtype == torch.float32 else 1e-12,
        rtol=0,
    )
    assert actual.shape == (2, 2, 3)
    assert actual.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_half_turns_can_mix_with_identity_in_one_batch(dtype):
    matrices = torch.stack((torch.eye(3), torch.diag(torch.tensor([-1.0, 1.0, -1.0])))).to(dtype)
    actual = matrix_to_axis_angle(matrices)
    torch.testing.assert_close(actual[0], torch.zeros(3, dtype=dtype))
    np.testing.assert_allclose(
        Rotation.from_rotvec(actual.double().numpy()).as_matrix(),
        matrices.double().numpy(),
        atol=2e-6 if dtype == torch.float32 else 1e-12,
        rtol=0,
    )
