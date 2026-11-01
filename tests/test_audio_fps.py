# Verify fractional FPS audio indexing using synthetic CPU tensors.
import math
from types import SimpleNamespace
import unittest

import torch

from musetalk.utils.audio_processor import AudioProcessor


class FakeWhisper:
    def encoder(self, feature, output_hidden_states):
        assert output_hidden_states
        values = torch.arange(500, dtype=torch.float32).reshape(1, 500, 1)
        return SimpleNamespace(hidden_states=(values,))


class AudioFpsTests(unittest.TestCase):
    def chunks(self, fps):
        processor = AudioProcessor.__new__(AudioProcessor)
        return processor.get_whisper_chunk([torch.zeros(1)], device='cpu',
            weight_dtype=torch.float32, whisper=FakeWhisper(),
            librosa_length=160000, fps=fps)

    def test_integer_fps_retains_existing_indexing(self):
        integer = self.chunks(25)
        decimal = self.chunks(25.0)
        self.assertTrue(torch.equal(integer, decimal))
        self.assertEqual(tuple(integer.shape), (250, 10, 1))
        self.assertEqual(integer[200, 0, 0].item(), 396)

    def test_fractional_fps_uses_fractional_frame_count_and_timing(self):
        fps = 30000 / 1001
        chunks = self.chunks(fps)
        self.assertEqual(tuple(chunks.shape), (math.floor(10 * fps), 10, 1))
        expected_index = math.floor(200 * 50 / fps) - math.ceil(50 / fps) * 2
        self.assertEqual(chunks[200, 0, 0].item(), expected_index)
        self.assertNotEqual(chunks[200, 0, 0].item(), math.floor(200 * 50 / int(fps)) - 4)


if __name__ == '__main__':
    unittest.main()
