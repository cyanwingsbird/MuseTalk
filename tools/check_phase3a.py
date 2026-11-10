"""Checks for the phase-3a options (CPU only; run from the repo root with the training venv):
  python tools/check_phase3a.py <old_dataset.py> <out_dir>
1. options off  -> samples identical to the previous dataset code (same seeds)
2. balance_by_source -> every source video equally likely
3. augment -> shapes/ranges unchanged; a pure flip mirrors frames, references and masks
4. PairedDiffAugment -> shared parameters, gradients, shapes
Writes <out_dir>/augment_grid.png for a visual check."""
import sys, os, random, hashlib, importlib.util, collections
import numpy as np, torch
old_path, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
torch.set_num_threads(2)
import musetalk.data.dataset as new
spec = importlib.util.spec_from_file_location('old_dataset', old_path); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
CFG = {'image_size': 256, 'T': 16, 'sample_method': 'pose_similarity_and_mouth_dissimilarity', 'top_k_ratio': 0.51,
       'contorl_face_min_size': True, 'dataset_key': 'HDTF', 'padding_pixel_mouth': 10, 'whisper_path': './models/whisper',
       'min_face_size': 200, 'cropping_jaw2edge_margin_mean': 10, 'cropping_jaw2edge_margin_std': 10,
       'crop_type': 'dynamic_margin_crop_resize', 'random_margin_method': 'normal',
       'list_path': './dataset/HDTF_stream/train_window.txt', 'epoch_samples': 2000,
       'meta_root': './dataset/HDTF_stream/meta', 'decode_threads': 1}
def seed(i): random.seed(i); np.random.seed(i); torch.manual_seed(i)
def digest(s): return hashlib.sha256(b''.join(s[k].numpy().tobytes() for k in ('pixel_values_vid', 'pixel_values_ref_img', 'pixel_values_face_mask', 'mel')) + np.asarray(s['audio_feature']).tobytes()).hexdigest()[:16]
ok = True
def check(name, cond, info=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info, flush=True)
# 1. regression
d_old, d_new = old.HDTFDataset(dict(CFG)), new.HDTFDataset(dict(CFG))
LIST = list(d_new.meta_paths)   # the rolling window list changes while this runs: every dataset below uses this snapshot
d_old.meta_paths = list(LIST)
def make(**kw):
    d = new.HDTFDataset(dict(CFG, **kw)); d.meta_paths = list(LIST); d._build_source_groups(); return d
same = 0
for i in range(4):
    seed(100 + i); a = d_old[0]; seed(100 + i); b = d_new[0]; same += digest(a) == digest(b) and a['wav_path'] == b['wav_path']
check('options off = previous code', same == 4, f'{same}/4 identical')
# 2. balanced sampling
d_bal = make(balance_by_source=True)
n_src = len(d_bal.source_keys); sizes = sorted(len(v) for v in d_bal.source_groups.values())
seed(1); N = 200 * n_src; cnt = collections.Counter()
key = lambda p: d_bal.source_key_re.match(os.path.splitext(os.path.basename(p))[0]).group(1)
for _ in range(N): cnt[key(d_bal._pick_meta_path())] += 1
freq = np.array([cnt[k] for k in d_bal.source_keys]) / 200
check('balanced: every source drawn about equally', len(cnt) == n_src and freq.min() > 0.7 and freq.max() < 1.3,
      f'{n_src} sources in window, clips per source {sizes[0]}..{sizes[-1]} (median {sizes[len(sizes)//2]}); draw ratio {freq.min():.2f}..{freq.max():.2f}')
seed(1); cu = collections.Counter(key(d_new._pick_meta_path()) for _ in range(N)); fu = np.array([cu[k] for k in d_bal.source_keys]) / 200
print(f'     (unbalanced draw ratio per source: {fu.min():.2f}..{fu.max():.2f})')
seed(7); s = d_bal[0]; check('balanced sample builds', s['pixel_values_vid'].shape == (16, 3, 256, 256))
# 3. augmentation
AUG = dict(color=0.2, flip_p=0.5, blur_p=0.2, blur_max=1.2, down_p=0.2, down_min=0.5, jpeg_p=0.3, jpeg_min=40)
d_aug = make(augment=AUG)
seed(11); s0 = d_new[0]
d_flip = make(augment=dict(flip_p=1.0))
seed(11); s1 = d_flip[0]
check('pure flip mirrors targets', torch.equal(s1['pixel_values_vid'], s0['pixel_values_vid'].flip(-1)))
check('pure flip mirrors references', torch.equal(s1['pixel_values_ref_img'], s0['pixel_values_ref_img'].flip(-1)))
check('pure flip mirrors masks', torch.equal(s1['pixel_values_face_mask'], s0['pixel_values_face_mask'].flip(-1)))
check('pure flip keeps audio', torch.equal(torch.as_tensor(s1['audio_feature']), torch.as_tensor(s0['audio_feature'])) and torch.equal(s1['mel'], s0['mel']))
rows = []; diffs = []; same_clip = 0
for i in range(6):
    seed(20 + i); a = d_new[0]; seed(20 + i); b = d_aug[0]
    assert b['pixel_values_vid'].shape == a['pixel_values_vid'].shape and b['pixel_values_ref_img'].shape == a['pixel_values_ref_img'].shape
    assert -1.0001 <= float(b['pixel_values_vid'].min()) and float(b['pixel_values_vid'].max()) <= 1.0001
    same_clip += a['wav_path'] == b['wav_path'] and torch.equal(a['mel'], b['mel'])
    fl = b['pixel_values_vid'] if torch.equal(a['pixel_values_face_mask'], b['pixel_values_face_mask']) else b['pixel_values_vid'].flip(-1)
    diffs.append(float((a['pixel_values_vid'] - fl).abs().mean()))
    m = lambda t: t[0].repeat(3, 1, 1) * 2 - 1 if t.shape[1] == 1 else t[0] * 2 - 1
    rows.append(torch.cat([a['pixel_values_vid'][0], a['pixel_values_ref_img'][0], b['pixel_values_vid'][0], b['pixel_values_vid'][8], b['pixel_values_ref_img'][0], m(b['pixel_values_face_mask'])], dim=2))
check('augmented samples valid, different, same clip', all(0 < d < 0.5 for d in diffs) and len(set(round(d, 4) for d in diffs)) > 1 and same_clip == 6, 'mean abs change ' + ' '.join(f'{d:.3f}' for d in diffs))
from torchvision.utils import save_image
save_image((torch.cat(rows, dim=1) + 1) / 2, os.path.join(out, 'augment_grid.jpg'))
print('     grid columns: original target | original reference | augmented target f0 | augmented target f8 | augmented reference | mask')
# 4. DiffAugment
from musetalk.utils.diffaug import PairedDiffAugment
da = PairedDiffAugment('color,translation'); seed(3)
x = torch.rand(8, 3, 64, 128) * 2 - 1; g = (x + 0.05 * torch.randn_like(x)).requires_grad_(True)
xr, xg = da.pair(x, g)
check('diffaug shapes', xr.shape == x.shape and xg.shape == g.shape)
p = da.sample(x); check('diffaug shared parameters', torch.equal(da.apply(x, p), da.apply(x.clone(), p)) and not torch.equal(da.apply(x, p), da.apply(x, da.sample(x))))
xg.sum().backward(); check('diffaug gradient reaches the generated image', g.grad is not None and float(g.grad.abs().sum()) > 0)
p0 = {'brightness': torch.zeros(8, 1, 1, 1), 'saturation': torch.ones(8, 1, 1, 1), 'contrast': torch.ones(8, 1, 1, 1), 'ty': torch.zeros(8, 1, 1, dtype=torch.long), 'tx': torch.zeros(8, 1, 1, dtype=torch.long)}
check('diffaug neutral parameters = identity', torch.allclose(da.apply(x, p0), x, atol=1e-6))
pt = dict(p0, ty=torch.full((8, 1, 1), 3, dtype=torch.long), tx=torch.full((8, 1, 1), -5, dtype=torch.long)); y = da.apply(x, pt)
check('diffaug translation shifts with zero fill', torch.allclose(y[:, :, :61, 5:], x[:, :, 3:, :123], atol=1e-6) and float(y[:, :, 61:, :].abs().max()) == 0)
check('diffaug cutout runs', PairedDiffAugment('color,translation,cutout').pair(x, x.clone())[0].shape == x.shape)
print('ALL PASS' if ok else 'SOME FAILED')
