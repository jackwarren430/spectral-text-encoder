import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from config_vae import VAEConfig, load_vae_config, save_vae_config
from data_vae import SentenceDataset, VAECollate
from model_vae import SpectralVAE
from train_vae import kl_anneal_factor, vae_objective, validate_vae


def tiny_config(**overrides):
    values = dict(
        vocab_size=101,
        eos_token_id=100,
        pad_token_id=100,
        vae_max_length=8,
        vae_dataset_specs=(("unused", ""),),
        d_model=16,
        encoder_layers=1,
        n_heads=4,
        ffn_dim=32,
        dropout=0.0,
        n_samples=64,
        duration=1.0,
        f_min=1.0,
        f_max=28.0,
        n_bands=4,
        global_bands=1,
        global_latent_per_band=2,
        token_latent_dim=2,
        spectral_conv_channels=8,
        spectral_conv_layers=1,
        spectral_patch_size=4,
        decoder_layers=1,
        vae_batch_size=3,
        vae_max_steps=3,
        vae_warmup_steps=0,
        kl_reconstruction_warmup_steps=1,
        kl_anneal_steps=2,
        precision="fp32",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        vae_prior_samples=2,
    )
    values.update(overrides)
    return VAEConfig(**values)


def example_batch(cfg):
    tokens = torch.randint(0, cfg.vocab_size - 1, (3, cfg.vae_max_length))
    lengths = torch.tensor([cfg.vae_max_length, 5, 3])
    mask = torch.arange(cfg.vae_max_length)[None, :] >= lengths[:, None]
    tokens[mask] = cfg.pad_token_id
    tokens[torch.arange(3), lengths - 1] = cfg.eos_token_id
    return tokens, mask


class SpectralVAETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_collate_appends_eos_and_uses_fixed_width(self):
        collate = VAECollate(max_length=5, pad_id=100, eos_id=99)
        tokens, mask = collate([[1, 2], [3, 4, 5, 6, 7]])
        torch.testing.assert_close(tokens[0], torch.tensor([1, 2, 99, 100, 100]))
        torch.testing.assert_close(tokens[1], torch.tensor([3, 4, 5, 6, 99]))
        torch.testing.assert_close(
            mask,
            torch.tensor(
                [[False, False, False, True, True], [False] * 5]
            ),
        )

    def test_forward_is_scalar_waveform_bottleneck_with_exact_fft_roundtrip(self):
        cfg = tiny_config()
        model = SpectralVAE(cfg).eval()
        tokens, mask = example_batch(cfg)
        with torch.no_grad():
            output = model(tokens, mask, sample=False)
        self.assertEqual(output.logits.shape, (3, cfg.vae_max_length, cfg.vocab_size))
        self.assertEqual(output.waveform.shape, (3, cfg.n_samples, 1))
        self.assertFalse(
            bool(torch.any(model.latent_layout.global_mask & model.latent_layout.token_mask))
        )
        recovered = torch.fft.rfft(output.waveform.squeeze(-1), norm="ortho")
        torch.testing.assert_close(recovered, output.spectrum, atol=2e-6, rtol=2e-6)
        self.assertTrue(torch.all(output.global_spectrum[:, model.latent_layout.token_mask] == 0))
        self.assertTrue(torch.all(output.token_spectrum[:, model.latent_layout.global_mask] == 0))

    def test_padding_tokens_cannot_change_canonical_waveform(self):
        cfg = tiny_config()
        model = SpectralVAE(cfg).eval()
        tokens, mask = example_batch(cfg)
        changed = tokens.clone()
        changed[mask] = torch.randint(0, cfg.vocab_size - 1, (int(mask.sum()),))
        with torch.no_grad():
            first = model.encode_waveform(tokens, mask, sample=False)
            second = model.encode_waveform(changed, mask, sample=False)
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)

    def test_elbo_backpropagates_through_both_posteriors_fft_and_decoder(self):
        cfg = tiny_config()
        model = SpectralVAE(cfg)
        tokens, mask = example_batch(cfg)
        output = model(tokens, mask, sample=True)
        loss, stats = vae_objective(output, tokens, mask, cfg, anneal=1.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(stats["global_kl_sum"]), 0.0)
        self.assertGreater(float(stats["token_kl_sum"]), 0.0)
        loss.backward()
        parameters = [
            model.encoder.global_head[-1].weight,
            model.encoder.token_head[-1].weight,
            model.latent_layout.global_basis,
            model.decoder.patch.weight,
            model.encoder.token_emb.weight,
        ]
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()), 0.0)

    def test_band_gain_edit_only_changes_requested_fft_band(self):
        cfg = tiny_config()
        model = SpectralVAE(cfg).eval()
        tokens, mask = example_batch(cfg)
        with torch.no_grad():
            waveform = model.encode_waveform(tokens, mask, sample=False)
            edited = model.edit_band_gain(waveform, band=0, gain=1.75)
        original_spectrum = torch.fft.rfft(waveform.squeeze(-1), norm="ortho")
        edited_spectrum = torch.fft.rfft(edited.squeeze(-1), norm="ortho")
        selected = model.latent_layout.band_masks[0]
        torch.testing.assert_close(
            edited_spectrum[:, selected], original_spectrum[:, selected] * 1.75,
            atol=2e-6, rtol=2e-6,
        )
        torch.testing.assert_close(
            edited_spectrum[:, ~selected], original_spectrum[:, ~selected],
            atol=2e-6, rtol=2e-6,
        )

    def test_validation_reports_latent_health_and_signal_ablations(self):
        cfg = tiny_config()
        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10], [11, 12], [13, 14, 15]]
        loader = DataLoader(
            SentenceDataset(sequences), batch_size=3, shuffle=False,
            collate_fn=VAECollate(cfg.vae_max_length, cfg.pad_token_id, cfg.eos_token_id),
        )
        model = SpectralVAE(cfg)
        metrics = validate_vae(
            model, loader, "cpu", cfg, max_batches=2, collect_text_samples=True
        )
        required = {
            "ce", "nll", "ppl", "acc", "exact", "eos_acc", "length_acc",
            "kl_global", "kl_token", "sample_ce", "zero_ce", "shuffle_ce",
            "active_global", "active_token", "prior_diversity", "prior_eos_rate",
        }
        self.assertTrue(required.issubset(metrics))
        self.assertTrue(all(math_value == math_value for math_value in [metrics[k] for k in required]))
        self.assertEqual(metrics["text_samples"]["targets"].shape, (3, cfg.vae_max_length))
        self.assertEqual(metrics["text_samples"]["prior"].shape, (2, cfg.vae_max_length))

    def test_config_roundtrip_and_kl_schedule(self):
        cfg = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            save_vae_config(cfg, directory)
            restored = load_vae_config(directory)
        self.assertEqual(restored.vae_dataset_specs, cfg.vae_dataset_specs)
        self.assertEqual(kl_anneal_factor(0, cfg), 0.0)
        self.assertEqual(kl_anneal_factor(1, cfg), 0.0)
        self.assertEqual(kl_anneal_factor(2, cfg), 0.5)
        self.assertEqual(kl_anneal_factor(3, cfg), 1.0)


if __name__ == "__main__":
    unittest.main()
