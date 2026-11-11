"""[canto] Light temporal loss (phase 3b).  Off unless loss_params.temporal_loss > 0.

MuseTalk generates every frame on its own, so nothing in the official losses looks at how the mouth moves from one
frame to the next.  This loss compares the frame-to-frame change of the generated frames with that of the real
frames (L1 on differences of consecutive frames).  It needs no extra network and no extra GPU memory worth
mentioning, unlike a feature-based temporal loss (TREPA in LatentSync needs a video model).
"""
import torch
from einops import rearrange


def temporal_diff_loss(pred, target, num_frames, lower_half=True):
    """pred, target: (batch * num_frames, C, H, W), frames of one sample consecutive in time.
    Returns the mean absolute difference between the generated and the real frame-to-frame changes."""
    if num_frames < 2:
        return pred.new_zeros(())
    p = rearrange(pred, '(b f) c h w -> b f c h w', f=num_frames)
    g = rearrange(target, '(b f) c h w -> b f c h w', f=num_frames)
    if lower_half:   # only the lower half of the face is generated
        h = p.shape[-2]
        p, g = p[..., h // 2:, :], g[..., h // 2:, :]
    return torch.abs((p[:, 1:] - p[:, :-1]) - (g[:, 1:] - g[:, :-1]).detach()).mean()
