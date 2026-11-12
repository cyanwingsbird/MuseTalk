"""Checks for the phase-3c helpers (CPU only; run from the repo root with the training venv): python tools/check_phase3c.py
The swap itself sits in train.py and is exercised by the GPU smoke test (swap_audio_p 0.5: both kinds of batch)."""
import torch, random
from musetalk.loss.swap import roll_batch, upper_half, upper_mask_like
ok = True
def check(name, cond, info=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info, flush=True)
torch.manual_seed(0)
a = torch.arange(2 * 16 * 50 * 384, dtype=torch.float32).view(2, 16, 50, 384); e = torch.rand(2, 512)
a2, e2 = roll_batch(a, e)
check('batch of 2: each clip gets the other clip\'s audio (prompts and SyncNet embedding)', torch.equal(a2[0], a[1]) and torch.equal(a2[1], a[0]) and torch.equal(e2[0], e[1]) and torch.equal(e2[1], e[0]))
b = torch.arange(3).view(3, 1); (b2,) = roll_batch(b)
check('batch of 3: nobody keeps its own audio', all(int(b2[i]) != i for i in range(3)), str(b2.flatten().tolist()))
check('shapes and frame order unchanged', a2.shape == a.shape and torch.equal(a2[0, 3], a[1, 3]))
f = torch.rand(4, 3, 256, 256) * 2 - 1; p = f.clone().requires_grad_(True)
check('upper half is rows 0..127', upper_half(f).shape == (4, 3, 128, 256) and torch.equal(upper_half(f), f[:, :, :128]))
q = f.clone(); q[:, :, 128:] += 0.7
check('upper-half L1 ignores the lower half', torch.nn.functional.l1_loss(upper_half(f), upper_half(q)).item() == 0)
torch.nn.functional.l1_loss(upper_half(p * 1.1), upper_half(f)).backward()
check('upper-half L1: gradient on the upper half only', p.grad[:, :, :128].abs().sum() > 0 and p.grad[:, :, 128:].abs().sum() == 0)
m = upper_mask_like(f)
check('mask: 1 above, 0 below', m[:, :, :128].min() == 1 and m[:, :, 128:].max() == 0 and m.shape == f.shape)
p2 = f.clone().requires_grad_(True); ((p2 * m) ** 2).sum().backward()
check('masked image: gradient on the upper half only', p2.grad[:, :, 128:].abs().sum() == 0 and p2.grad[:, :, :128].abs().sum() > 0)
src = open('train.py', encoding='utf-8').read()
check('train.py: no random number is drawn when swap_audio_p is 0', "if swap_p > 0 and bsz > 1 and cfg.loss_params.sync_loss > 0 and random.random() < swap_p:" in src)
check('train.py: feature matching skipped in swapped batches (both discriminators)', src.count("cfg.loss_params.fm_loss[0] > 0 and not swap_audio") == 2)
print('ALL PASS' if ok else 'SOME CHECKS FAILED', flush=True)
