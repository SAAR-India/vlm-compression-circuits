import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.compression_checkpoints import (
    load_smoothquant_scales,
    save_smoothquant_scales,
)
from src import compression_configs
from src.crosscoder import config as crosscoder_config
from src.compression_methods import (
    SecondOrderCompressor,
    SmoothQuantLinear,
    install_smoothquant_wrappers,
    smoothquant_weight,
)


class SecondOrderCompressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _calibrated_layer(self) -> tuple[nn.Linear, SecondOrderCompressor]:
        layer = nn.Linear(8, 8, bias=False)
        compressor = SecondOrderCompressor(layer)
        compressor.add_batch(torch.randn(32, 8))
        return layer, compressor

    def test_sparsegpt_reaches_requested_sparsity(self):
        layer, compressor = self._calibrated_layer()
        compressor.prune_sparsegpt(sparsity=0.5, blocksize=8)
        zero_fraction = (layer.weight == 0).float().mean().item()
        self.assertAlmostEqual(zero_fraction, 0.5, delta=0.02)
        self.assertTrue(torch.isfinite(layer.weight).all())

    def test_gptq_uses_at_most_sixteen_levels_per_group(self):
        layer, compressor = self._calibrated_layer()
        original = layer.weight.detach().clone()
        compressor.quantize_gptq(bits=4, group_size=8, blocksize=8)
        self.assertFalse(torch.equal(original, layer.weight))
        self.assertTrue(torch.isfinite(layer.weight).all())
        for row in layer.weight:
            self.assertLessEqual(torch.unique(row).numel(), 16)


class SmoothQuantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_w8a8_wrapper_is_finite_and_close_to_float(self):
        layer = nn.Linear(8, 4)
        inputs = torch.randn(6, 8)
        expected = layer(inputs)
        activation_amax = inputs.abs().amax(dim=0)
        scale = smoothquant_weight(layer, activation_amax, alpha=0.5)
        wrapped = SmoothQuantLinear(layer, scale)
        actual = wrapped(inputs)
        relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-8)
        self.assertLess(relative_error.item(), 0.05)
        self.assertTrue(torch.isfinite(actual).all())

    def test_scale_serialization_and_wrapper_restoration(self):
        model = nn.Sequential(nn.Linear(8, 4))
        scale = torch.linspace(0.5, 1.5, 8)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_smoothquant_scales(tmpdir, {"0": scale})
            loaded = load_smoothquant_scales(tmpdir)
            self.assertTrue(torch.equal(loaded["0"], scale))
            install_smoothquant_wrappers(model, loaded)
        self.assertIsInstance(model[0], SmoothQuantLinear)
        self.assertEqual(tuple(model(torch.randn(2, 8)).shape), (2, 4))


class CompressionConfigurationTests(unittest.TestCase):
    def test_new_methods_share_pipeline_and_crosscoder_registry(self):
        expected = {"sparsegpt", "gptq", "smoothquant"}
        self.assertTrue(expected <= set(compression_configs.METHODS))
        self.assertTrue(expected <= set(crosscoder_config.METHODS))

    def test_compressed_models_path_is_anchored_under_src(self):
        output_path = Path(compression_configs.OUTPUT_DIR)
        self.assertTrue(output_path.is_absolute())
        self.assertEqual(output_path.name, "compressed_models")
        self.assertEqual(output_path.parent.name, "src")


if __name__ == "__main__":
    unittest.main()
