# Changes in this MuseTalk fork

This document records changes relative to upstream MuseTalk commit
[`0a89dec`](https://github.com/TMElyralab/MuseTalk/commit/0a89dec45a0192b824e3cf4daf96c239440c5ed8).
Update it whenever fork-specific behavior, defaults, tests, or limitations change.
The MuseTalk v1.5 model weights were not retrained for these changes.

## Visual profiles for v1.5 normal inference

On HPC, run inference inside an allocated GPU Slurm job, not on a login node.
The following commands start inference and can be used within such a job.
Choose a profile with the optional third argument:

```sh
sh inference.sh v1.5 normal original
sh inference.sh v1.5 normal custom
```

`original` is the default for `inference.sh`; its v1.5 visual settings are the
upstream defaults: `--parsing_mode jaw --extra_margin 10`. `custom` selects
`--parsing_mode mouth --extra_margin 0`. The profile changes visual settings
only: the fractional-FPS correction described below stays active in both.
These examples use the existing configuration and model paths; choose suitable
input and output locations before running inference.

The custom profile writes to a new timestamped `results/test-custom-...`
directory, keeping it separate from the upstream default output directory.

Alternatively set `MUSE_VISUAL_PROFILE=original` or `custom` in the environment.
An explicit third argument takes precedence. Unknown values fail rather than
silently choosing a profile. The convenience `custom` profile is restricted to
v1.5 normal inference, the path tested end to end. Existing two-argument v1.0
and realtime uses retain their original visual settings.

For direct `python -m scripts.inference` use, the two controls can be set
independently using the existing flags:

| Control | Upstream visual default | Fork option used in our sample |
| --- | --- | --- |
| Blending mask | `--parsing_mode jaw` | `--parsing_mode mouth` |
| Extra lower crop margin | `--extra_margin 10` | `--extra_margin 0` |

The margin is an existing MuseTalk option, not a change to the model. It changes
the lower edge of the face crop used for generation and reconstruction. A zero
margin is not guaranteed to be best for every face.

### Mouth mask implementation

`musetalk/utils/face_parsing/__init__.py` adds `mouth_alpha` and the optional
`mouth` parsing mode. The existing face parser identifies the mouth and lips
(classes 11, 12 and 13) in the source frame. Their bounds center a soft ellipse;
the existing `jaw` mask is multiplied by the ellipse and passed through the
upstream blending process. The goal is to retain more original chin pixels while
using the generated mouth. If fewer than 10 mouth pixels are detected, the code
falls back to `jaw`. The `jaw`, `raw`, and `neck` modes remain available.

The source-derived mask can be inaccurate under occlusion, large generated mouth
movements, profile views, or segmentation errors. Its center can also change
between frames. The fallback can reintroduce the upstream chin artifact.
Neither this mask nor the margin setting changes model weights or guarantees
correct lips, synchronization, or temporal stability.

## Fractional video FPS

`musetalk/utils/audio_processor.py` retains the frame rate as a float when
selecting Whisper audio chunks. Upstream converts it to an integer, which can
accumulate alignment error for non-integer frame rates. This correction remains
active for both visual profiles; `original` therefore means original **visual**
settings, not a byte-for-byte reproduction of all upstream behavior. Normal
video and realtime video inference use the source video frame rate automatically;
realtime computes it separately for each source video unless `--fps` is supplied.
If source FPS cannot be read (for example, when only cached avatar files remain),
realtime warns and falls back to 25 fps; use the existing `--fps` to override it.
Single images and frame directories have no inherent rate, so the default remains
25 fps; the existing `--fps` argument can specify a finite positive decimal rate,
for example `--fps 29.97`. Normal video inference continues to use the source
rate rather than `--fps`. No new option is needed to enable the fix.
This does not establish precise timing for variable-frame-rate video or prove
synchronization for every input.

## Testing and scope

Run the CPU-only unit tests with the project dependencies installed:

```sh
CUDA_VISIBLE_DEVICES="" python3 -m unittest discover -s tests -p 'test_*.py'
```

The tests use synthetic data and do not load model weights or run inference.
They cover profile selection, mouth-mask coverage and fallback, and integer and
fractional audio-frame indexing.

The mask comparison reused the same generated frames for one person. On a
150-frame segment, mean source-relative difference in the selected chin region
fell from about 4.88 to 2.94 on a 0-255 scale. This is not a visual quality or
synchronization score. A complete 156-second sample was also produced with the
custom mask and margin. Margin experiments on a second person did not establish
a general improvement. Realtime and many poses, faces, and occlusions have not
been validated end to end for these settings.

Realtime avatars cache masks and coordinates: changing mask or margin requires
preparing a separate avatar with the intended settings; reusing an old cache can
retain the old result. Realtime is outside the convenience custom profile.

## Release hygiene

Input media, local environments and launchers, generated results, and accidental
sample-media deletions are not required for these changes. Review the staged
files before committing; model weights are not included.

## Resumable training

Upstream training could not resume. `train.py` wrote only `unet-<step>.pth` during training; its
`accelerator.save_state` call sat in the per-epoch block, which never runs with the default
`save_model_epoch_interval`, so no `checkpoint-*` folder was created and a restarted job began again
from step 0 (in stage 1 also from the initial UNet). The state also lacked the GAN discriminators.

`train.py` now saves a full resume state every `checkpointing_steps`: the UNet, AdamW, LR scheduler,
both discriminators with their optimizers and LR schedulers (registered with
`accelerator.register_for_checkpointing`), and RNG states. Each state is written to
`checkpoint-<step>.partial/`, marked `COMPLETE` and renamed, so an interrupted save never becomes
the newest checkpoint; resume loads the newest complete one. Saves and stops happen only on
optimizer-update boundaries (`global_step` counts batches; the optimizer steps every
`gradient_accumulation_steps` batches), so no partially accumulated gradient is lost, and
`checkpointing_steps` must be a multiple of `gradient_accumulation_steps`.

Optional controls, all off by default:

- `TRAIN_DEADLINE_EPOCH` (Unix time) and `TRAIN_SAVE_MARGIN_SEC` (default 1800): save and exit
  cleanly that long before a scheduler time limit.
- A `STOP` file in the run folder (`<output_dir>/<exp_name>/`): save and exit at the next update.
- `resume_states_to_keep` (default 2): full resume states kept; `total_limit` still bounds the
  `unet-<step>.pth` files.
- A `DONE` file is written when `max_train_steps` is reached, and the epoch loop now stops there
  instead of running one extra batch per remaining epoch.

## Training configuration options

- `solver.lr_scheduler_total_steps` and `solver.lr_scheduler_warmup_steps` (in LR-scheduler steps,
  i.e. optimizer updates) set the schedule length explicitly. This lets a run with a different
  batch-size/accumulation split, for example fewer GPUs, keep upstream's LR and discriminator
  schedules. Without them the upstream formula is used.
- `data.val_list` points the validation loader at a separate list file (upstream drew validation
  images from `train.txt`). Without it behavior is unchanged.

Machine-specific training configs, launch scripts and accelerate configs are kept outside the
repository.

## Faster, exact data loading

These change speed and robustness only; on both stage configs the loader returns tensors identical to
upstream's (checked on original clips and on lossless face-cropped clips, fixed seeds).

- Each sample's frames are decoded in one sorted pass (`get_batch`) instead of one random seek per
  frame. Same pixels; stage 2 (16 targets + 16 references) loads about 3.6x faster.
- `ensure_wav` returns a WAV that is already 16 kHz mono 16-bit PCM unchanged. Before, every sample
  ran ffmpeg twice and wrote `<clip>_16k.wav` beside the data (identical samples), and two data-loader
  workers could race on the same file.
- The spectrogram that `get_audio_file_mel` computed and then discarded is no longer computed.
- A video decode error makes the loader try another sample instead of crashing the worker.
- Face-cropped videos: a meta file may carry `crop_offset` `[x, y]`, the top-left corner of the crop in
  the original frame. Boxes, landmarks and `video_size` stay in full-frame coordinates, so every crop,
  mask and dynamic margin is computed exactly as on the full frame.

## Rolling data window options

All optional; without them behavior is unchanged.

- `data.train_list`, `data.meta_root`: training list file and meta folder (upstream hard-coded
  `./dataset/HDTF/train.txt` and `./dataset/HDTF/meta`); `data.val_meta_root` for validation.
- `data.reload_list_each_epoch`: re-read the training list at the start of every epoch, before data
  workers start, so a list that changes during training (a rolling window of clips) is picked up.
  An empty or unreadable list keeps the previous one.
- `data.epoch_samples`: samples per epoch (upstream fixed 100000). Clips are drawn at random with
  replacement, so this only sets how often the list is re-read and workers restart.

## Decoder threads per data worker (`data.decode_threads`)

`FaceDataset` now passes `num_threads=cfg['decode_threads']` to decord's `VideoReader`
(default `0`, decord's automatic thread count, i.e. upstream behaviour). With many
DataLoader workers on few CPUs, each worker's automatic decode threads oversubscribe the
CPUs and the GPU waits for data. Setting `data.decode_threads: 1` keeps one decode thread
per worker. Samples are bit-identical either way (checked with fixed seeds on stage-1 and
stage-2 settings); only speed changes.

## Optional generalisation options for fine-tuning on few speakers (phase 3a)

All optional and off by default; without them behavior is unchanged (with the options off, samples are
bit-identical to the previous code for fixed seeds: `tools/check_phase3a.py`).

- `data.balance_by_source`: draw a source video first, then one of its clips, instead of drawing every clip
  with equal chance. The source is the clip name without its trailing `-<number>` (`data.source_key_regex`
  overrides the pattern). `data.balance_alpha` (default 0) sets the chance of a source to
  (number of its clips) ** alpha: 0 = every source equally, 1 = every clip equally. Training set only; the
  groups follow the list when `data.reload_list_each_epoch` re-reads it.
- `data.augment`: picture augmentation with one set of random parameters for all frames of a sample; the
  target frames and the reference frames get the same change, the mouth masks are only flipped. Keys:
  `color` (brightness / contrast / saturation factors in [1-s, 1+s]), `flip_p`, `blur_p`, `blur_max`
  (Gaussian radius), `down_p`, `down_min` (down-scale, then back up), `jpeg_p`, `jpeg_min` (quality).
  Training set only.
- `loss_params.diffaug_policy` (for example `"color,translation"`): differentiable augmentation of the
  inputs of both discriminators (`musetalk/utils/diffaug.py`, after Zhao et al., "Differentiable
  Augmentation for Data-Efficient GAN Training", 2020). The parameters are shared by the real and the
  generated image of each pair, so the feature-matching loss stays aligned. The reconstruction, perceptual
  and sync losses still see the unaugmented images.
- `configs/training/canto_stage3a.yaml`: a third fine-tuning phase that starts from a stage-2 checkpoint
  with a flat low learning rate and the three options above; `train_canto.sbatch` accepts `STAGE=stage3a`.
- `tools/check_phase3a.py`: CPU checks for the above.

Status: CPU checks pass; a full GPU run of these options has not been done yet.

## Optional audio-side options for fine-tuning on few speakers (phase 3b)

Both optional and off by default; without them behavior is unchanged (with the options off, samples are
bit-identical to the previous code for fixed seeds: `tools/check_phase3b.py`).

- `data.audio_augment`: voice augmentation of the waveform that goes to the Whisper feature extractor
  (`musetalk/utils/audio_aug.py`): pitch shift (`pitch_p`, `pitch_max` semitones; formants move with the pitch),
  spectral tilt (`tilt_p`, `tilt_max` dB per octave around 1 kHz), white or pink noise (`noise_p`, `snr_db: [low, high]`).
  Length and timing are unchanged (pitch within 0.1%, timing within a few ms), so audio and video stay aligned;
  the mel spectrogram for the SyncNet loss is still made from the original audio.
- `loss_params.temporal_loss`: weight of a light temporal loss (`musetalk/loss/temporal.py`): L1 between the
  frame-to-frame change of the generated lower face and that of the real frames over the consecutive frames of a
  sample. No extra network.
- `configs/training/canto_stage3b.yaml`: the phase-3a config plus voice augmentation (temporal loss present, weight 0),
  from stage-2 step 24,000, 8,000 steps; `STAGE=stage3b` in `train_canto.sbatch`.

## Optional swapped-audio training batches against audio-visual leakage (phase 3c)

Optional and off by default; without it behavior is unchanged.

- `loss_params.swap_audio_p`: share of batches trained with the audio of the other clip in the batch
  (`musetalk/loss/swap.py`; needs a batch of at least 2 clips and `sync_loss > 0`). In such a batch the real lower
  half of the face is not a valid target: L1 and the perceptual loss use the upper half only, the feature-matching
  terms and the temporal loss are skipped, the adversarial terms stay, and the sync loss is computed against the
  swapped audio with weight `loss_params.swap_sync_loss` (default: `sync_loss`).
- Purpose: a model trained only on matched audio and video can read part of the mouth movement from the visible upper
  face; swapped batches make the audio the only source for the mouth.
- `configs/training/canto_stage3c.yaml`: the phase-3a config plus `swap_audio_p: 0.3`, `swap_sync_loss: 0.1`,
  from stage-2 step 24,000, 8,000 steps; `STAGE=stage3c` in `train_canto.sbatch`. `tools/check_phase3c.py` checks the helpers.
