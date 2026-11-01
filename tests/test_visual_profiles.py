# Test visual profiles without invoking MuseTalk or a GPU.
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class VisualProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        stub = Path(self.tmp.name) / 'python3'
        stub.write_text('#!/bin/sh\nprintf "ARG:%s\\n" "$@"\n', encoding='utf-8')
        stub.chmod(0o755)
        self.env = os.environ.copy()
        self.env.pop('MUSE_VISUAL_PROFILE', None)
        self.env['PATH'] = self.tmp.name + os.pathsep + self.env.get('PATH', '')

    def tearDown(self):
        self.tmp.cleanup()

    def launch(self, script, *args, profile=None):
        env = self.env.copy()
        if profile is not None:
            env['MUSE_VISUAL_PROFILE'] = profile
        result = subprocess.run(['sh', str(ROOT / script), *args], cwd=ROOT,
                                env=env, capture_output=True, text=True)
        argv = [line[4:] for line in result.stdout.splitlines()
                if line.startswith('ARG:')]
        return result, argv

    def test_default_and_original_keep_upstream_visual_defaults(self):
        for args in [('v1.5', 'normal'), ('v1.5', 'normal', 'original'),
                     ('v1.0', 'normal'), ('v1.5', 'realtime')]:
            with self.subTest(args=args):
                result, argv = self.launch('inference.sh', *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn('--parsing_mode', argv)
                self.assertNotIn('--extra_margin', argv)

    def test_custom_sets_both_visual_options(self):
        for args, env_profile in [(('v1.5', 'normal', 'custom'), None),
                                  (('v1.5', 'normal'), 'custom')]:
            with self.subTest(args=args, env_profile=env_profile):
                result, argv = self.launch('inference.sh', *args, profile=env_profile)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(argv[argv.index('--parsing_mode') + 1], 'mouth')
                self.assertEqual(argv[argv.index('--extra_margin') + 1], '0')
                self.assertIn('test-custom-', argv[argv.index('--result_dir') + 1])

    def test_realtime_wrapper_does_not_force_an_integer_rate(self):
        result, argv = self.launch('inference.sh', 'v1.5', 'realtime')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--fps', argv)

    def test_explicit_argument_overrides_environment(self):
        result, argv = self.launch('inference.sh', 'v1.5', 'normal', 'original', profile='custom')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--parsing_mode', argv)
        self.assertNotIn('--extra_margin', argv)

    def test_invalid_or_unsupported_profiles_do_not_invoke_inference(self):
        for args in [('v1.5', 'normal', 'bad'), ('v1.0', 'normal', 'custom'),
                     ('v1.5', 'realtime', 'custom')]:
            with self.subTest(args=args):
                result, argv = self.launch('inference.sh', *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(argv, [])



if __name__ == '__main__':
    unittest.main()
