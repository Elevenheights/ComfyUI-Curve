"""Run with: python -m unittest discover -s tests -v"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch


def load_node():
    root = Path(__file__).resolve().parents[1] / 'nodes'
    modules = {}
    for name, directory in [('_curve_test', root),
                            ('_curve_test.core', root / 'core'),
                            ('_curve_test.camera_raw', root / 'camera_raw')]:
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        modules[name] = module
    # Processing tests do not need preset directories or ComfyUI server routes.
    presets = types.ModuleType('_curve_test.core.generic_preset_manager')
    presets.GenericPresetManager = lambda name: None
    modules[presets.__name__] = presets
    spec = importlib.util.spec_from_file_location(
        '_curve_test.camera_raw.enhance', root / 'camera_raw/enhance.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.CameraRawEnhanceNode


CameraRawEnhanceNode = load_node()


class CameraRawPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.node = CameraRawEnhanceNode()
        self.image = torch.from_numpy(
            np.random.default_rng(42).uniform(.1, .9, (2, 32, 96, 3)).astype(np.float32))

    def test_neutral_preserves_pixels_and_promotes_half(self):
        result = self.node.apply_camera_raw_enhance(self.image)[0]
        self.assertTrue(torch.equal(result, self.image))
        result = self.node.apply_camera_raw_enhance(self.image.half())[0]
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.equal(result, self.image.half().float()))

    def test_active_effects_remain_float32(self):
        pixels = self.image[0].numpy()
        effects = ('exposure', 'highlights', 'shadows', 'whites', 'blacks',
                   'saturation', 'contrast', 'texture', 'clarity', 'vibrance', 'dehaze')
        for effect in effects:
            for value in (-5, 5):
                with self.subTest(effect=effect, value=value):
                    result = getattr(self.node, '_apply_' + effect)(pixels.copy(), value)
                    self.assertEqual(result.dtype, np.float32)
                    self.assertTrue(np.isfinite(result).all())

    def test_fine_gradients_survive_effects(self):
        gradient = np.linspace(.4, .401, 1000, dtype=np.float32)[None, :, None]
        pixels = np.tile(gradient, (8, 1, 3))
        for effect in ('texture', 'clarity', 'vibrance', 'dehaze'):
            with self.subTest(effect=effect):
                result = getattr(self.node, '_apply_' + effect)(pixels.copy(), 5)
                self.assertGreater(np.unique(result[4, :, 0]).size, 900)

    def test_white_balance_normalizes_without_indexing_error(self):
        pixels = np.full((4, 6, 3), .99, dtype=np.float32)
        result = self.node._apply_white_balance(pixels, 100, -100)
        expected = pixels * np.array([1.4 * 1.1, 1.15 * .8, .75 * 1.1], dtype=np.float32)
        expected /= np.maximum(expected.max(axis=2, keepdims=True), 1)
        np.testing.assert_allclose(result, expected, atol=1e-7)
        self.assertEqual(result.dtype, np.float32)

    def test_float_hsv_preserves_skin_protection_ranges(self):
        hsv = np.array([[[20, .3, .7], [120, .3, .7], [340, .3, .7]]], dtype=np.float32)
        pixels = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        result = self.node._apply_vibrance(pixels, 20)
        saturation = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)[0, :, 1]
        increase = .2 * (120 / 255) * (1 - .3 ** 2)
        np.testing.assert_allclose(saturation, [.3 + increase * .3, .3 + increase,
                                               .3 + increase * .3], atol=1e-6)

    def test_fractional_mask_feathering_and_batch_selection(self):
        masks = torch.zeros((2, 32, 96), dtype=torch.float32)
        masks[0, :, 48:] = .5001
        masks[1] = 1 - masks[0]
        result = self.node.apply_camera_raw_enhance(
            self.image, exposure=.5, mask=masks, mask_blur=2.5)[0]
        enhanced = self.node.apply_camera_raw_enhance(self.image, exposure=.5)[0]
        for index in range(2):
            smooth = cv2.GaussianBlur(masks[index].numpy(), (11, 11), 2.5)[:, :, None]
            expected = self.image[index].numpy() * (1 - smooth) + enhanced[index].numpy() * smooth
            np.testing.assert_allclose(result[index].numpy(), expected, atol=1e-7)
        self.assertEqual(result.dtype, torch.float32)

    def test_singleton_and_inverted_masks(self):
        mask = torch.full((1, 32, 96), .12345)
        result = self.node.apply_camera_raw_enhance(
            self.image[:1], exposure=.5, mask=mask, mask_blur=2.5, invert_mask=True)[0]
        enhanced = self.node.apply_camera_raw_enhance(self.image[:1], exposure=.5)[0]
        expected = self.image[:1] * .12345 + enhanced * (1 - .12345)
        torch.testing.assert_close(result, expected, atol=1e-7, rtol=0)


if __name__ == '__main__':
    unittest.main()
