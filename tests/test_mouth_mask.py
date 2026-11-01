# Test mouth blending with synthetic segmentation; no weights are loaded.
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional

from musetalk.utils.face_parsing import FaceParsing, mouth_alpha


class DummyParser:
    def __init__(self, labels):
        logits = functional.one_hot(torch.from_numpy(labels.astype(np.int64)), 19)
        self.logits = logits.permute(2, 0, 1).unsqueeze(0).float()
        self.preprocess = lambda image: torch.zeros(3, 64, 64)
        self.net = lambda tensor: [self.logits.to(tensor.device)]
        self.kernel = np.ones((3, 3), np.uint8)
        self.cheek_kernel = np.ones((3, 3), np.uint8)
        self.cheek_mask = np.full((64, 64), 255, np.uint8)


class MouthMaskTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.zeros((64, 64), np.uint8)
        self.labels[4:60, 4:60] = 1
        self.labels[38:43, 23:41] = 12
        self.labels[43:48, 23:41] = 13
        self.image = Image.new('RGB', (64, 64))

    def parse(self, labels, mode):
        return np.asarray(FaceParsing.__call__(DummyParser(labels), self.image,
                                               size=(64, 64), mode=mode))

    def test_alpha_retains_lips_and_restricts_chin_region(self):
        alpha = mouth_alpha(self.labels)
        lips = np.isin(self.labels, [11, 12, 13])
        self.assertIsNotNone(alpha)
        self.assertEqual(alpha.shape, self.labels.shape)
        self.assertEqual(alpha.dtype, np.float32)
        self.assertTrue(np.all(alpha[lips] == 1))
        self.assertEqual(alpha[0, 0], 0)
        jaw = self.parse(self.labels, 'jaw')
        mouth = self.parse(self.labels, 'mouth')
        self.assertTrue(np.all(mouth <= jaw))
        np.testing.assert_array_equal(mouth[lips], jaw[lips])
        self.assertEqual(mouth[5, 5], 0)
        self.assertEqual(jaw[5, 5], 255)

    def test_fewer_than_ten_mouth_pixels_fall_back_to_jaw(self):
        no_lips = np.where(np.isin(self.labels, [11, 12, 13]), 1, self.labels)
        for count in (0, 9):
            with self.subTest(count=count):
                labels = no_lips.copy()
                labels[40, 20:20 + count] = 12
                self.assertIsNone(mouth_alpha(labels))
                np.testing.assert_array_equal(self.parse(labels, 'jaw'),
                                              self.parse(labels, 'mouth'))


if __name__ == '__main__':
    unittest.main()
