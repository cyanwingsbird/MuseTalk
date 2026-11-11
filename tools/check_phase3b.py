"""Checks for the phase-3b options (CPU only; run from the repo root with the training venv):
  python tools/check_phase3b.py <old_dataset.py> <out_dir>
1. options off      -> samples identical to the previous dataset code (same seeds), audio features included
2. voice augmentation in the dataset -> frames, masks and the SyncNet mel unchanged; only the Whisper features change
3. augment_voice    -> same length, pitch moved by the asked amount, timing kept, tilt and noise as asked, speed
4. temporal_diff_loss -> zero for identical clips, known value on a test tensor, gradients, single frame = 0
"""
import sys, os, random, hashlib, importlib.util, time, warnings
import numpy as np, torch
warnings.filterwarnings('ignore')
old_path, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
torch.set_num_threads(2)
import librosa
import musetalk.data.dataset as new
from musetalk.utils.audio_aug import augment_voice, pitch_shift, spectral_tilt, add_noise
from musetalk.loss.temporal import temporal_diff_loss
spec = importlib.util.spec_from_file_location('old_dataset', old_path); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
CFG = {'image_size': 256, 'T': 16, 'sample_method': 'pose_similarity_and_mouth_dissimilarity', 'top_k_ratio': 0.51,
       'contorl_face_min_size': True, 'dataset_key': 'HDTF', 'padding_pixel_mouth': 10, 'whisper_path': './models/whisper',
       'min_face_size': 200, 'cropping_jaw2edge_margin_mean': 10, 'cropping_jaw2edge_margin_std': 10,
       'crop_type': 'dynamic_margin_crop_resize', 'random_margin_method': 'normal',
       'list_path': './dataset/HDTF_stream/train_window.txt', 'epoch_samples': 2000,
       'meta_root': './dataset/HDTF_stream/meta', 'decode_threads': 1}
def seed(i): random.seed(i); np.random.seed(i); torch.manual_seed(i)
KEYS = ('pixel_values_vid', 'pixel_values_ref_img', 'pixel_values_face_mask', 'mel')
def digest(s, keys=KEYS + ('audio_feature',)): return hashlib.sha256(b''.join(torch.as_tensor(s[k]).numpy().tobytes() for k in keys)).hexdigest()
ok = True
def check(name, cond, info=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info, flush=True)

# 1. regression against the previous dataset code
d_old, d_new = old.HDTFDataset(dict(CFG)), new.HDTFDataset(dict(CFG))
LIST = list(d_new.meta_paths)   # the rolling window list changes while this runs: every dataset below uses this snapshot
d_old.meta_paths = list(LIST); d_new.meta_paths = list(LIST)
def make(**kw):
    d = new.HDTFDataset(dict(CFG, **kw)); d.meta_paths = list(LIST); d._build_source_groups(); return d
same = 0
for i in range(4):
    seed(100 + i); a = d_old[0]; seed(100 + i); b = d_new[0]; same += digest(a) == digest(b) and a['wav_path'] == b['wav_path']
check('options off = previous code (frames, masks, mel, Whisper features)', same == 4, f'{same}/4 identical')

# 2. voice augmentation inside the dataset
AUG = dict(pitch_p=1.0, pitch_max=2.0, tilt_p=1.0, tilt_max=3.0, noise_p=1.0, snr_db=[15, 35])
d_aug = make(audio_augment=AUG)
pic_same = mel_same = feat_changed = shape_same = clip_same = 0; cos = []
N = 4
for i in range(N):
    seed(200 + i); a = d_new[0]; seed(200 + i); b = d_aug[0]
    clip_same += a['wav_path'] == b['wav_path']
    pic_same += digest(a, KEYS[:3]) == digest(b, KEYS[:3]); mel_same += torch.equal(a['mel'], b['mel'])
    fa, fb = torch.as_tensor(a['audio_feature']).float(), torch.as_tensor(b['audio_feature']).float()
    shape_same += fa.shape == fb.shape; feat_changed += not torch.equal(fa, fb)
    cos.append(torch.nn.functional.cosine_similarity(fa.flatten(), fb.flatten(), dim=0).item())
check('voice augmentation: same clip, frames, references and masks', clip_same == N and pic_same == N, f'{pic_same}/{N}')
check('voice augmentation: SyncNet mel untouched (made from the original audio)', mel_same == N, f'{mel_same}/{N}')
check('voice augmentation: Whisper features keep their shape and do change', shape_same == N and feat_changed == N,
      f'shape {tuple(fa.shape)}; cosine similarity to the unaugmented features {min(cos):.3f}..{max(cos):.3f}')
d_off = make(audio_augment=dict(pitch_p=0, tilt_p=0, noise_p=0))
seed(300); a = d_new[0]; seed(300); b = d_off[0]
check('all probabilities zero = off', digest(a) == digest(b))

# 3. the augmentation itself
sr = 16000; t = np.arange(sr * 8) / sr
tone = (sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 20)) * 0.05 * (np.sin(2 * np.pi * 1.5 * t) > 0)).astype(np.float32)
def f0(x):
    f, _, _ = librosa.pyin(x[:sr * 2], fmin=80, fmax=400, sr=sr); return float(np.nanmedian(f))
def lag_ms(a, b):
    k = np.ones(20) / 20
    e1 = np.convolve(np.abs(a).reshape(-1, 16).mean(1), k, 'same'); e2 = np.convolve(np.abs(b).reshape(-1, 16).mean(1), k, 'same')
    return int(np.argmax(np.correlate(e2 - e2.mean(), e1 - e1.mean(), 'full')) - (len(e1) - 1))
res = []
for n in (-2.0, -1.0, 1.0, 2.0):
    z = pitch_shift(tone, n); res.append((n, len(z) == len(tone), abs(f0(z) / (150 * 2 ** (n / 12)) - 1), abs(lag_ms(tone, z))))
check('pitch shift: same length, pitch within 1%, timing within 10 ms', all(r[1] and r[2] < 0.01 and r[3] <= 10 for r in res),
      '; '.join(f'{r[0]:+.0f} st: pitch error {r[2] * 100:.2f}%, timing {r[3]} ms' for r in res))
z = spectral_tilt(tone, sr, 3.0); S0, S1 = np.abs(np.fft.rfft(tone)), np.abs(np.fft.rfft(z)); fr = np.fft.rfftfreq(len(tone), 1 / sr)
g = {fq: 20 * np.log10(S1[np.argmin(abs(fr - fq))] / S0[np.argmin(abs(fr - fq))]) for fq in (150, 600, 1050)}
check('spectral tilt +3 dB/octave: gains as asked', all(abs(g[fq] - 3 * np.log2(fq / 1000)) < 0.3 for fq in g) and len(z) == len(tone),
      ', '.join(f'{fq} Hz {v:+.1f} dB' for fq, v in g.items()))
m = [10 * np.log10(np.mean(tone ** 2) / np.mean((add_noise(tone, s_, p_) - tone) ** 2)) for s_ in (15, 35) for p_ in (False, True)]
check('noise: signal-to-noise ratio as asked', all(abs(a - b) < 0.3 for a, b in zip(m, (15, 15, 35, 35))), ' '.join(f'{v:.1f}' for v in m))
random.seed(1); r0 = random.random(); random.seed(1); z, info = augment_voice(tone, sr, None); r1 = random.random()
check('off: waveform returned as is, no random numbers drawn', z is tone and r0 == r1 and info == {})
wav = None
for p in LIST[:50]:
    try:
        import json
        w = json.load(open(p))['wav_path']
        if os.path.exists(w): wav = w; break
    except Exception: pass
if wav:
    y, _ = librosa.load(wav, sr=sr); y = y[:sr * 30]; random.seed(5); t0 = time.time()
    for _ in range(10): z, info = augment_voice(y, sr, AUG)
    dt = (time.time() - t0) / 10
    check('real clip: same length, finite, in range', len(z) == len(y) and np.isfinite(z).all() and np.abs(z).max() <= 1.0,
          f'{len(y) / sr:.1f} s clip, {dt * 1000:.0f} ms per call on this CPU; last draw {info}')
    import soundfile as sf
    try:
        random.seed(6)
        for name, cfg in (('orig', None), ('pitch_up2', None), ('pitch_down2', None), ('all', AUG)):
            z = y if name == 'orig' else pitch_shift(y, 2.0) if name == 'pitch_up2' else pitch_shift(y, -2.0) if name == 'pitch_down2' else augment_voice(y, sr, cfg)[0]
            sf.write(os.path.join(out, f'voice_{name}.wav'), z[:sr * 10], sr)
        print('     listening samples written:', out)
    except Exception as e: print('     (no listening samples:', e, ')')

# 4. temporal loss
torch.manual_seed(0); g_ = torch.rand(2 * 16, 3, 64, 64) * 2 - 1
check('temporal loss: zero for identical clips', temporal_diff_loss(g_.clone(), g_, 16).item() == 0)
check('temporal loss: zero for a constant offset (same motion)', temporal_diff_loss(g_ + 0.1, g_, 16).item() < 1e-6)
p_ = g_.clone().view(2, 16, 3, 64, 64); p_[:, 8:] += 0.5; p_ = p_.view(32, 3, 64, 64)   # one jump of 0.5 between frames 7 and 8
v = temporal_diff_loss(p_, g_, 16).item()
check('temporal loss: known value for one jump', abs(v - 0.5 / 15) < 1e-5, f'{v:.5f} (expected {0.5 / 15:.5f})')
q = g_.clone().requires_grad_(True); temporal_diff_loss(q * 1.1, g_, 16).backward()
check('temporal loss: gradient reaches the generated frames, lower half only', q.grad.abs().sum() > 0 and q.grad[:, :, :32].abs().sum() == 0)
check('temporal loss: single frame gives 0', temporal_diff_loss(g_[:2], g_[:2], 1).item() == 0)
print('ALL PASS' if ok else 'SOME CHECKS FAILED', flush=True)
