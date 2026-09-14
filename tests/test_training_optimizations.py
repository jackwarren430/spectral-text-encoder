import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from config import Config
from model import (
    SpectralAE,
    combine_signal_channels,
    freq_separation_loss,
    freqs_for_separation,
    frequency_anchor_radius,
    frequency_anchors,
)
from run_utils import load_config
from train_clip import (
    _encode_to_signal_unbucketed,
    _per_channel_loss,
    encode_to_signal,
    micro_step_direct,
    micro_step_grad_cache,
    signal_to_embedding,
)


def reference_freq_loss(f, min_sep, valid=None):
    diff = (f.unsqueeze(-1) - f.unsqueeze(-2)).abs()
    K = f.size(-1)
    penalty = torch.relu(min_sep - diff).masked_fill(
        torch.eye(K, dtype=torch.bool, device=f.device), 0.0
    )
    if valid is None:
        return penalty.sum() / (f.size(0) * K * (K - 1))
    penalty = penalty.masked_fill(~(valid.unsqueeze(-1) & valid.unsqueeze(-2)), 0.0)
    n_real = valid.sum(-1).to(f.dtype)
    return (penalty.sum(dim=(-1, -2)) / (n_real * (n_real - 1)).clamp(min=1)).mean()


def reference_per_channel(sig_a, sig_b, logit_scale, cfg):
    total = sig_a.new_zeros(())
    for channel in range(cfg.d_sine):
        xa = sig_a[:, :, channel : channel + 1]
        xb = sig_b[:, :, channel : channel + 1]
        if cfg.clip_embedding_type == "spectral":
            xa = torch.fft.rfft(xa, dim=1).abs()
            xb = torch.fft.rfft(xb, dim=1).abs()
        ea = F.normalize(xa.flatten(1), dim=-1)
        eb = F.normalize(xb.flatten(1), dim=-1)
        logits = (ea @ eb.T) * logit_scale.exp()
        targets = torch.arange(sig_a.size(0))
        total += 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
        )
    return total / cfg.d_sine


class TrainingOptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_triangular_frequency_loss_matches_matrix_loss_and_gradient(self):
        valid = torch.arange(37)[None, :] < torch.tensor([37, 29, 17, 8])[:, None]
        old_f = torch.randn(4, 37, requires_grad=True)
        new_f = old_f.detach().clone().requires_grad_(True)
        expected = reference_freq_loss(old_f, 0.75, valid)
        actual = freq_separation_loss(new_f, 0.75, valid, pair_bucket_size=12)
        expected_grad, = torch.autograd.grad(expected, old_f)
        actual_grad, = torch.autograd.grad(actual, new_f)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_batched_per_channel_loss_matches_loop_and_gradient(self):
        for embedding_type in ("time", "spectral"):
            with self.subTest(embedding_type=embedding_type):
                cfg = SimpleNamespace(d_sine=3, clip_embedding_type=embedding_type)
                old_a = torch.randn(7, 32, 3, requires_grad=True)
                old_b = torch.randn(7, 32, 3, requires_grad=True)
                old_scale = torch.tensor(2.6593, requires_grad=True)
                new_a = old_a.detach().clone().requires_grad_(True)
                new_b = old_b.detach().clone().requires_grad_(True)
                new_scale = old_scale.detach().clone().requires_grad_(True)
                expected = reference_per_channel(old_a, old_b, old_scale, cfg)
                actual = _per_channel_loss(new_a, new_b, new_scale, cfg)
                expected_grad = torch.autograd.grad(expected, (old_a, old_b, old_scale))
                actual_grad = torch.autograd.grad(actual, (new_a, new_b, new_scale))
                torch.testing.assert_close(actual, expected)
                for got, want in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(got, want)

    def test_summed_signal_readout_matches_variance_preserving_definition(self):
        signal = torch.randn(4, 17, 3)
        multi_cfg = SimpleNamespace(signal_channel_mode="multi")
        sum_cfg = SimpleNamespace(signal_channel_mode="sum")
        self.assertIs(combine_signal_channels(signal, multi_cfg), signal)
        expected = signal.sum(dim=-1, keepdim=True) / (3 ** 0.5)
        torch.testing.assert_close(combine_signal_channels(signal, sum_cfg), expected)

    def test_summed_embedding_combines_before_time_or_spectral_projection(self):
        signal = torch.randn(4, 32, 3)
        summed = signal.sum(dim=-1, keepdim=True) / (3 ** 0.5)
        for embedding_type in ("time", "spectral"):
            with self.subTest(embedding_type=embedding_type):
                cfg = SimpleNamespace(
                    signal_channel_mode="sum",
                    clip_embedding_type=embedding_type,
                )
                x = summed
                if embedding_type == "spectral":
                    x = torch.fft.rfft(x, dim=1).abs()
                expected = F.normalize(x.flatten(1), dim=-1)
                torch.testing.assert_close(signal_to_embedding(signal, cfg), expected)

    def test_execution_bucketing_preserves_rows_and_real_frequencies(self):
        cfg = Config(
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_dim=64,
            d_sine=2,
            n_samples=32,
            dropout=0.0,
            clip_batch_size=5,
            clip_length_bucket_size=4,
        )
        model = SpectralAE(cfg).eval()
        lengths = torch.tensor([3, 11, 5, 16, 8])
        tokens = torch.randint(0, cfg.vocab_size, (5, 16))
        mask = torch.arange(16)[None, :] >= lengths[:, None]
        with torch.no_grad():
            expected_signal, expected_f = _encode_to_signal_unbucketed(
                model, tokens, mask, cfg
            )
            actual_signal, actual_f = encode_to_signal(model, tokens, mask, cfg)
        real = (~mask).unsqueeze(-1).expand_as(expected_f)
        torch.testing.assert_close(actual_signal, expected_signal, atol=2e-4, rtol=2e-5)
        torch.testing.assert_close(actual_f[real], expected_f[real], atol=1e-4, rtol=2e-5)

    def test_anchored_frequencies_stay_inside_fixed_channel_regions(self):
        cfg = Config(
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_dim=64,
            d_sine=3,
            n_samples=32,
            f_min=10.0,
            f_max=130.0,
            dropout=0.0,
            frequency_param_mode="anchored",
            freq_anchor_radius_frac=0.45,
        )
        model = SpectralAE(cfg).eval()
        tokens = torch.randint(0, cfg.vocab_size, (4, 7))
        with torch.no_grad():
            _, f, _ = model.encoder(tokens)
        anchors = frequency_anchors(cfg, f.device, f.dtype)
        radius = frequency_anchor_radius(cfg)
        self.assertTrue(torch.all((f - anchors.view(1, 1, -1)).abs() < radius))
        # The fixed guard keeps adjacent channel regions disjoint.
        self.assertTrue(torch.all(anchors[:-1] + radius < anchors[1:] - radius))

    def test_anchored_separation_groups_tokens_by_observable_channel(self):
        cfg = Config(d_sine=2, frequency_param_mode="anchored")
        # Cross-channel frequency reuse is harmless. Inside each channel the
        # two token waves are 10 Hz apart, so a 4 Hz hinge is exactly zero.
        f = torch.tensor([[[10.0, 20.0], [20.0, 10.0]]])
        fk, valid = freqs_for_separation(f, cfg)
        self.assertIsNone(valid)
        torch.testing.assert_close(fk, torch.tensor([[10.0, 20.0], [20.0, 10.0]]))
        torch.testing.assert_close(freq_separation_loss(fk, 4.0), torch.tensor(0.0))

    def test_anchored_separation_repeats_pad_mask_per_channel(self):
        cfg = Config(d_sine=3, frequency_param_mode="anchored")
        f = torch.randn(2, 4, 3)
        pad_mask = torch.tensor([
            [False, False, True, True],
            [False, False, False, True],
        ])
        fk, valid = freqs_for_separation(f, cfg, pad_mask)
        self.assertEqual(fk.shape, (6, 4))
        expected = (~pad_mask).unsqueeze(1).expand(2, 3, 4).reshape(6, 4)
        torch.testing.assert_close(valid, expected)

    def test_legacy_run_resumes_in_fp32(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Config().__dict__.copy()
            for key in ("precision", "cuda_tf32", "fused_optimizer"):
                data.pop(key)
            Path(tmp, "config.json").write_text(json.dumps(data))
            cfg = load_config(tmp)
        self.assertEqual(cfg.precision, "fp32")
        self.assertFalse(cfg.cuda_tf32)
        self.assertFalse(cfg.fused_optimizer)

    def test_legacy_run_resumes_with_global_frequency_parameterization(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Config().__dict__.copy()
            data.pop("frequency_param_mode")
            Path(tmp, "config.json").write_text(json.dumps(data))
            cfg = load_config(tmp)
        self.assertEqual(cfg.frequency_param_mode, "global")

    def test_grad_cache_still_matches_direct_with_execution_buckets(self):
        cfg = Config(
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_dim=64,
            d_sine=2,
            n_samples=32,
            dropout=0.0,
            clip_batch_size=8,
            clip_cache_chunk_size=4,
            clip_length_bucket_size=4,
            precision="fp32",
            num_workers=0,
            pin_memory=False,
        )
        lengths_a = torch.tensor([3, 13, 5, 9, 2, 12, 7, 11])
        lengths_b = torch.tensor([8, 4, 13, 6, 10, 3, 11, 5])
        mask_a = torch.arange(13)[None, :] >= lengths_a[:, None]
        mask_b = torch.arange(13)[None, :] >= lengths_b[:, None]
        batch = (
            torch.randint(0, cfg.vocab_size, (8, 13)),
            mask_a,
            torch.randint(0, cfg.vocab_size, (8, 13)),
            mask_b,
        )
        direct = SpectralAE(cfg)
        cached = SpectralAE(cfg)
        cached.load_state_dict(direct.state_dict())
        direct_scale = torch.nn.Parameter(torch.tensor(cfg.clip_logit_scale_init))
        cached_scale = torch.nn.Parameter(direct_scale.detach().clone())
        min_sep = cfg.freq_sep_min_bins / cfg.duration
        direct_stats = micro_step_direct(
            direct, batch, direct_scale, cfg, 1, min_sep, "cpu"
        )
        cached_stats = micro_step_grad_cache(
            cached, batch, cached_scale, cfg, 1, min_sep, "cpu", 4
        )
        torch.testing.assert_close(
            torch.tensor(direct_stats[:4]), torch.tensor(cached_stats[:4])
        )
        for direct_param, cached_param in zip(
            direct.encoder.parameters(), cached.encoder.parameters()
        ):
            if direct_param.grad is not None:
                torch.testing.assert_close(
                    direct_param.grad, cached_param.grad, atol=5e-5, rtol=2e-4
                )
        torch.testing.assert_close(direct_scale.grad, cached_scale.grad)

    def test_summed_readout_grad_cache_matches_direct(self):
        cfg = Config(
            vocab_size=257,
            d_model=16,
            n_layers=1,
            n_heads=4,
            ffn_dim=32,
            d_sine=3,
            n_samples=24,
            decoder_layers=1,
            dropout=0.0,
            signal_channel_mode="sum",
            clip_batch_size=4,
            clip_cache_chunk_size=2,
            clip_length_bucket_size=4,
            precision="fp32",
            num_workers=0,
            pin_memory=False,
        )
        lengths_a = torch.tensor([3, 7, 4, 6])
        lengths_b = torch.tensor([6, 4, 7, 3])
        mask_a = torch.arange(7)[None, :] >= lengths_a[:, None]
        mask_b = torch.arange(7)[None, :] >= lengths_b[:, None]
        batch = (
            torch.randint(0, cfg.vocab_size, (4, 7)),
            mask_a,
            torch.randint(0, cfg.vocab_size, (4, 7)),
            mask_b,
        )
        direct = SpectralAE(cfg)
        cached = SpectralAE(cfg)
        cached.load_state_dict(direct.state_dict())
        self.assertEqual(direct.decoder.proj.in_features, 1)
        with torch.no_grad():
            logits, targets, aux = direct(batch[0][:2, :4])
        self.assertEqual(logits.shape, (2, 4, cfg.vocab_size))
        self.assertEqual(targets.shape, (2, 4))
        self.assertTrue(torch.isfinite(aux))
        direct_scale = torch.nn.Parameter(torch.tensor(cfg.clip_logit_scale_init))
        cached_scale = torch.nn.Parameter(direct_scale.detach().clone())
        min_sep = cfg.freq_sep_min_bins / cfg.duration
        direct_stats = micro_step_direct(
            direct, batch, direct_scale, cfg, 1, min_sep, "cpu"
        )
        cached_stats = micro_step_grad_cache(
            cached, batch, cached_scale, cfg, 1, min_sep, "cpu", 2
        )
        torch.testing.assert_close(
            torch.tensor(direct_stats[:4]), torch.tensor(cached_stats[:4])
        )
        for direct_param, cached_param in zip(
            direct.encoder.parameters(), cached.encoder.parameters()
        ):
            if direct_param.grad is not None:
                torch.testing.assert_close(
                    direct_param.grad, cached_param.grad, atol=5e-5, rtol=2e-4
                )
        decoder_grad_norm = 0.0
        for direct_param, cached_param in zip(
            direct.decoder.parameters(), cached.decoder.parameters()
        ):
            self.assertIsNotNone(direct_param.grad)
            self.assertIsNotNone(cached_param.grad)
            self.assertTrue(torch.isfinite(direct_param.grad).all())
            self.assertTrue(torch.isfinite(cached_param.grad).all())
            decoder_grad_norm += direct_param.grad.square().sum().item()
            torch.testing.assert_close(
                direct_param.grad, cached_param.grad, atol=5e-5, rtol=2e-4
            )
        self.assertGreater(decoder_grad_norm, 0.0)
        torch.testing.assert_close(direct_scale.grad, cached_scale.grad)


if __name__ == "__main__":
    unittest.main()
