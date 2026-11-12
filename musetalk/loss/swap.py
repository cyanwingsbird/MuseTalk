"""[canto] Swapped-audio training batches (phase 3c).  Off unless loss_params.swap_audio_p > 0.

Why: MuseTalk is trained to reconstruct a frame from its own upper face, a reference frame and the matching audio.
The upper face moves with the speech, so the model can learn to read part of the mouth movement from the picture
instead of the audio (measured in this project with the silence test).  In use, the audio is new and disagrees with
the picture, and that learned shortcut works against it.

What: with probability swap_audio_p a whole batch is trained with the audio of the OTHER clip in the batch
(audio_prompts and the SyncNet audio embedding are rolled by one along the batch axis).  For such a batch the real
lower half of the face is no longer a valid target, so
  - L1 and the perceptual loss are computed on the upper half only,
  - the feature-matching terms (which pair a generated image with its real frame) and the temporal loss are skipped,
  - the adversarial terms stay (they only ask for a realistic face),
  - the sync loss is computed against the swapped audio, with weight swap_sync_loss (default: sync_loss).
Normal batches are untouched.  Needs a batch of at least 2 clips and sync_loss > 0.
"""
import torch


def roll_batch(*tensors):
    """Give every sample the audio of its neighbour in the batch (a cyclic shift by one: nobody keeps its own)."""
    return tuple(torch.roll(t, shifts=1, dims=0) for t in tensors)


def upper_half(x):
    """The rows of an image batch (..., H, W) that are not generated: the upper half of the face crop."""
    return x[..., : x.shape[-2] // 2, :]


def upper_mask_like(x):
    """1 on the upper half, 0 on the lower half; multiplying keeps gradients for the upper half only."""
    m = torch.zeros_like(x)
    m[..., : x.shape[-2] // 2, :] = 1
    return m
