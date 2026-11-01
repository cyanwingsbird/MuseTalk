# Test both actual inference CLI parsers without loading model weights.
import argparse
import ast
from contextlib import redirect_stderr
import io
from pathlib import Path
import unittest
import warnings

from musetalk.utils.audio_processor import parse_fps

ROOT = Path(__file__).resolve().parents[1]


def inference_parser(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    main = next(node for node in tree.body if isinstance(node, ast.If)
                and any(isinstance(n, ast.Name) and n.id == '__name__'
                        for n in ast.walk(node.test)))
    selected = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args'
                                                 for t in node.targets):
            break
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser'
                                                 for t in node.targets):
            selected.append(node)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute)
              and isinstance(node.value.func.value, ast.Name)
              and node.value.func.value.id == 'parser'):
            selected.append(node)
    namespace = {'argparse': argparse, 'parse_fps': parse_fps}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['parser']


class FpsOptionTests(unittest.TestCase):
    def test_fractional_and_integer_rates_are_accepted_by_both_clis(self):
        for name in ('inference.py', 'realtime_inference.py'):
            with self.subTest(script=name):
                parser = inference_parser(ROOT / 'scripts' / name)
                self.assertAlmostEqual(parser.parse_args(['--fps', '29.97']).fps, 29.97)
                self.assertEqual(parser.parse_args(['--fps', '25']).fps, 25)
                expected_default = None if name == 'realtime_inference.py' else 25
                self.assertEqual(parser.parse_args([]).fps, expected_default)

    def test_realtime_uses_source_fps_or_existing_override(self):
        path = ROOT / 'scripts' / 'realtime_inference.py'
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'resolve_realtime_fps')
        from types import SimpleNamespace
        namespace = {'parse_fps': parse_fps, 'argparse': argparse, 'warnings': warnings,
                     'os': SimpleNamespace(path=SimpleNamespace(isdir=lambda path: path == 'frames')),
                     'get_file_type': lambda path: 'image' if path.endswith('.png') else 'unsupported',
                     'get_video_fps': lambda path: 30000 / 1001}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        resolve = namespace['resolve_realtime_fps']
        self.assertAlmostEqual(resolve('video.mp4', None), 30000 / 1001)
        self.assertAlmostEqual(resolve('video.m4v', None), 30000 / 1001)
        self.assertEqual(resolve('frames', None), 25)
        self.assertEqual(resolve('video.mp4', 24.5), 24.5)
        self.assertEqual(resolve('image.png', None), 25)
        namespace['get_video_fps'] = lambda path: 0
        with self.assertWarnsRegex(RuntimeWarning, 'using 25 fps'):
            self.assertEqual(resolve('bad.mp4', None), 25)

    def test_realtime_passes_per_video_rate_to_inference(self):
        path = ROOT / 'scripts' / 'realtime_inference.py'
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        main = next(n for n in tree.body if isinstance(n, ast.If)
                    and any(isinstance(x, ast.Name) and x.id == '__name__'
                            for x in ast.walk(n.test)))
        loop = next(n for n in main.body if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == 'avatar_id')
        calls = []
        class DummyAvatar:
            def __init__(self, **kwargs):
                pass
            def inference(self, audio, name, fps, skip):
                calls.append((audio, name, fps, skip))
        from types import SimpleNamespace
        configs = {'a': {'preparation': False, 'video_path': 'one.mp4',
                         'audio_clips': {'clip': 'audio.wav'}}}
        namespace = {'inference_config': configs, 'Avatar': DummyAvatar,
                     'args': SimpleNamespace(fps=None, version='v15', batch_size=8,
                                             skip_save_images=True),
                     'resolve_realtime_fps': lambda path, fps: 30000 / 1001}
        with redirect_stderr(io.StringIO()):
            exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), 'exec'), namespace)
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(calls[0][2], 30000 / 1001)

    def test_invalid_rates_are_rejected_by_both_clis(self):
        for name in ('inference.py', 'realtime_inference.py'):
            parser = inference_parser(ROOT / 'scripts' / name)
            for value in ('0', '-1', 'nan', 'inf', 'nonsense'):
                with self.subTest(script=name, fps=value), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(['--fps', value])


if __name__ == '__main__':
    unittest.main()
