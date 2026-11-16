# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

import math

import pytest
import torch

from lerobot.policies.g05.action_codec_g05 import G05NativeActionCodec, _BlockDCT


@pytest.mark.parametrize("block_size,horizon", [(2, 6), (3, 6), (4, 4), (4, 11), (8, 13)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_block_dct_roundtrip(block_size: int, horizon: int, dtype: torch.dtype) -> None:
    transform = _BlockDCT(block_size)
    signal = torch.randn(2, horizon, 3, generator=torch.Generator().manual_seed(0), dtype=dtype)

    restored = transform.idct(transform.dct(signal), horizon)

    assert restored.shape == signal.shape
    assert restored.dtype == dtype
    torch.testing.assert_close(restored, signal, atol=4e-6, rtol=4e-6)


def test_block_dct_dc_coefficients_decode_to_constant_signals() -> None:
    transform = _BlockDCT(8)
    coefficients = torch.zeros(1, 16, 2)
    coefficients[:, 0] = math.sqrt(8) * 0.25
    coefficients[:, 8] = math.sqrt(8) * -0.5
    expected = torch.cat((torch.full((1, 8, 2), 0.25), torch.full((1, 8, 2), -0.5)), dim=1)

    torch.testing.assert_close(transform.idct(coefficients, horizon=16), expected)


def test_block_dct_roundtrip_preserves_signal_gradients() -> None:
    transform = _BlockDCT(8)
    generator = torch.Generator().manual_seed(1)
    signal = torch.randn(2, 11, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(signal.shape, generator=generator, dtype=signal.dtype)

    restored = transform.idct(transform.dct(signal), horizon=11)
    gradient = torch.autograd.grad((restored * weights).sum(), signal)[0]

    torch.testing.assert_close(gradient, weights, atol=4e-6, rtol=4e-6)


def _codec_config(block_size: int, enabled: bool) -> dict:
    return {
        "parts_meta": {"control": 3},
        "num_residuals": 2,
        "model_arch": {
            "horizon": 8,
            "horizon_patch_size": 2,
            "max_component_dim": 3,
            "conv_in_action_kernel": 2,
            "encoder_channels": 64,
            "c_mults": [1],
            "strides": [[1, 1]],
            "transformer_depths": [1],
            "latent_dim": 16,
            "num_heads": 1,
            "dim_heads": 64,
            "rope_base": 10_000,
            "ffn_mult": 2,
            "layer_scale_init": 0.01,
            "n_codebooks": 2,
            "codebook_size": 16,
            "codebook_dim": 4,
            "use_block_dct": enabled,
            "block_dct_block_size": block_size,
        },
    }


@pytest.mark.parametrize("block_size", [4, 8])
def test_native_action_codec_decodes_dct_coefficients_in_the_time_domain(block_size: int) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(2)
        codec = G05NativeActionCodec(_codec_config(block_size, enabled=True), action_token_begin=100)
        for quantizer in codec.model.rvq.quantizers:
            quantizer.codebook.normal_()
        coefficient_codec = G05NativeActionCodec(
            _codec_config(block_size, enabled=False), action_token_begin=100
        )
    coefficient_codec.module.load_state_dict(codec.module.state_dict(), strict=True)
    codec.module.eval()
    coefficient_codec.module.eval()
    tokens = []
    for level in range(codec.num_residuals):
        tokens.append(100 + codec.marker_indices[f"<control_{level}>"])
        tokens.extend(100 + (index + level) % codec.codebook_size for index in range(codec.code_length))
    token_ids = torch.tensor(tokens)

    coefficients = coefficient_codec.decode_language_tokens(token_ids, horizon=8, action_dim=3).double()
    # Independent orthonormal DCT-III definition: one weighted cosine sum per output sample.
    expected = torch.zeros_like(coefficients)
    for start in range(0, 8, block_size):
        for time in range(block_size):
            for frequency in range(block_size):
                scale = math.sqrt((1 if frequency == 0 else 2) / block_size)
                angle = math.pi * (time + 0.5) * frequency / block_size
                expected[start + time] += coefficients[start + frequency] * scale * math.cos(angle)

    actual = codec.decode_language_tokens(token_ids, horizon=8, action_dim=3)

    torch.testing.assert_close(actual, expected.to(actual), atol=4e-6, rtol=4e-6)
