"""Differentiable augmentation of discriminator inputs.

After Zhao et al., "Differentiable Augmentation for Data-Efficient GAN Training" (NeurIPS 2020): with little or
narrow training data a discriminator memorises the real images; applying the same kind of random, differentiable
augmentation to the real and the generated images before the discriminator prevents this.

Here the parameters are sampled once per image and shared by the real and the generated image of each pair, so the
feature-matching loss (which compares discriminator features of the two) stays aligned.
"""
import torch
import torch.nn.functional as F


class PairedDiffAugment:
    """policy: comma-separated subset of "color", "translation", "cutout" (images in [-1, 1], shape N x C x H x W)."""

    def __init__(self, policy='color,translation', translation_ratio=0.125, cutout_ratio=0.5):
        self.ops = [p.strip() for p in str(policy).split(',') if p.strip()]
        unknown = set(self.ops) - {'color', 'translation', 'cutout'}
        if unknown:
            raise ValueError(f'unknown diffaug policy entries: {sorted(unknown)}')
        self.translation_ratio = float(translation_ratio)
        self.cutout_ratio = float(cutout_ratio)

    def sample(self, x):
        """Draw the random parameters for a batch shaped like x."""
        n, _, h, w = x.shape
        dev, dt = x.device, x.dtype
        p = {}
        if 'color' in self.ops:
            p['brightness'] = torch.rand(n, 1, 1, 1, device=dev, dtype=dt) - 0.5
            p['saturation'] = torch.rand(n, 1, 1, 1, device=dev, dtype=dt) * 2
            p['contrast'] = torch.rand(n, 1, 1, 1, device=dev, dtype=dt) + 0.5
        if 'translation' in self.ops:
            sy, sx = int(h * self.translation_ratio + 0.5), int(w * self.translation_ratio + 0.5)
            p['ty'] = torch.randint(-sy, sy + 1, (n, 1, 1), device=dev)
            p['tx'] = torch.randint(-sx, sx + 1, (n, 1, 1), device=dev)
        if 'cutout' in self.ops:
            ch, cw = int(h * self.cutout_ratio + 0.5), int(w * self.cutout_ratio + 0.5)
            p['cut_size'] = (ch, cw)
            p['cy'] = torch.randint(0, h + (1 - ch % 2), (n, 1, 1), device=dev)
            p['cx'] = torch.randint(0, w + (1 - cw % 2), (n, 1, 1), device=dev)
        return p

    def apply(self, x, p):
        """Apply parameters drawn by sample() (differentiable with respect to x)."""
        n, _, h, w = x.shape
        if 'brightness' in p:
            x = x + p['brightness']
            mean = x.mean(dim=1, keepdim=True)
            x = (x - mean) * p['saturation'] + mean
            mean = x.mean(dim=[1, 2, 3], keepdim=True)
            x = (x - mean) * p['contrast'] + mean
        if 'ty' in p:
            gb, gy, gx = torch.meshgrid(
                torch.arange(n, dtype=torch.long, device=x.device),
                torch.arange(h, dtype=torch.long, device=x.device),
                torch.arange(w, dtype=torch.long, device=x.device), indexing='ij')
            gy = torch.clamp(gy + p['ty'] + 1, 0, h + 1)
            gx = torch.clamp(gx + p['tx'] + 1, 0, w + 1)
            x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
            x = x_pad.permute(0, 2, 3, 1).contiguous()[gb, gy, gx].permute(0, 3, 1, 2)
        if 'cy' in p:
            ch, cw = p['cut_size']
            gb, gy, gx = torch.meshgrid(
                torch.arange(n, dtype=torch.long, device=x.device),
                torch.arange(ch, dtype=torch.long, device=x.device),
                torch.arange(cw, dtype=torch.long, device=x.device), indexing='ij')
            gy = torch.clamp(gy + p['cy'] - ch // 2, min=0, max=h - 1)
            gx = torch.clamp(gx + p['cx'] - cw // 2, min=0, max=w - 1)
            mask = torch.ones(n, h, w, dtype=x.dtype, device=x.device)
            mask[gb, gy, gx] = 0
            x = x * mask.unsqueeze(1)
        return x

    def pair(self, real, generated):
        """Augment a (real, generated) pair with shared parameters."""
        p = self.sample(real)
        return self.apply(real, p), self.apply(generated, p)
