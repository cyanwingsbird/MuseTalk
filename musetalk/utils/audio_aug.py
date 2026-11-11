"""[canto] Voice augmentation for the Whisper input (phase 3b).  Off unless data.audio_augment is set.

The training data come from few speakers; these changes make the same speech sound like other voices and other
recordings while leaving WHAT is said and WHEN unchanged: the output has exactly the input's length and timing,
so the audio stays aligned with the video frames.  Only the waveform that goes to the Whisper feature extractor is
changed; the mel spectrogram for the SyncNet loss is still made from the original audio.

Config keys (all optional; a missing or zero key switches that part off):
  pitch_p, pitch_max   with probability pitch_p shift the pitch by a uniform amount in [-pitch_max, +pitch_max]
                       semitones (phase-vocoder time stretch + resampling: formants move with the pitch, as for a
                       speaker with a shorter or longer vocal tract)
  tilt_p, tilt_max     with probability tilt_p apply a spectral tilt of a uniform amount in [-tilt_max, +tilt_max]
                       dB per octave around 1 kHz (brighter or duller voice / microphone), limited to +-12 dB
  noise_p, snr_db      with probability noise_p add white or pink noise at a signal-to-noise ratio drawn uniformly
                       from snr_db = [low, high] dB
Random numbers come from Python's `random`, like the picture augmentation; nothing is drawn when a part is off.
"""
import random
from fractions import Fraction

import numpy as np


def pitch_shift(y, n_steps):
    """Shift the pitch by n_steps semitones; same length as the input."""
    import warnings
    import librosa
    from scipy.signal import resample_poly
    if abs(n_steps) < 1e-3 or len(y) < 4096:
        return y
    rate = 2.0 ** (-n_steps / 12.0)
    # time stretch with a phase vocoder (32 ms windows at 16 kHz keep the timing within a few ms): length / rate, pitch unchanged
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        D = librosa.stft(y.astype(np.float32), n_fft=512, hop_length=128)
        stretched = librosa.istft(librosa.phase_vocoder(D, rate=rate, hop_length=128), hop_length=128, length=int(round(len(y) / rate)))
    fr = Fraction(rate).limit_denominator(200)                                   # back to the original length: pitch moves
    out = resample_poly(stretched, fr.numerator, fr.denominator).astype(np.float32)
    if len(out) >= len(y):
        return out[:len(y)]
    return np.pad(out, (0, len(y) - len(out)))


def spectral_tilt(y, sr, db_per_octave, limit_db=12.0):
    """Tilt the spectrum around 1 kHz by db_per_octave; same length as the input."""
    if abs(db_per_octave) < 1e-3 or len(y) < 16:
        return y
    spec = np.fft.rfft(y.astype(np.float64))
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    gain_db = np.clip(db_per_octave * np.log2(np.maximum(f, 50.0) / 1000.0), -limit_db, limit_db)
    out = np.fft.irfft(spec * 10.0 ** (gain_db / 20.0), n=len(y))
    return out.astype(np.float32)


def add_noise(y, snr_db, pink=False, rng=None):
    """Add white or pink noise at the given signal-to-noise ratio (relative to the RMS of y)."""
    rng = rng or np.random.default_rng(random.getrandbits(32))
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if len(y) else 0.0
    if rms <= 1e-6:
        return y
    n = rng.standard_normal(len(y))
    if pink:
        spec = np.fft.rfft(n)
        f = np.fft.rfftfreq(len(y), 1.0)
        spec[1:] /= np.sqrt(f[1:]); spec[0] = 0
        n = np.fft.irfft(spec, n=len(y))
    n *= rms / (10.0 ** (snr_db / 20.0)) / max(float(np.sqrt(np.mean(n ** 2))), 1e-12)
    return (y + n).astype(np.float32)


def augment_voice(y, sr, cfg):
    """Apply the configured voice augmentation to a mono float waveform; returns (waveform, description dict)."""
    info = {}
    if not cfg:
        return y, info
    n0, peak0 = len(y), float(np.max(np.abs(y))) if len(y) else 0.0
    p = float(cfg.get('pitch_p', 0) or 0)
    if p > 0 and random.random() < p:
        info['pitch'] = random.uniform(-float(cfg.get('pitch_max', 2.0)), float(cfg.get('pitch_max', 2.0)))
        y = pitch_shift(y, info['pitch'])
    p = float(cfg.get('tilt_p', 0) or 0)
    if p > 0 and random.random() < p:
        info['tilt'] = random.uniform(-float(cfg.get('tilt_max', 3.0)), float(cfg.get('tilt_max', 3.0)))
        y = spectral_tilt(y, sr, info['tilt'])
    p = float(cfg.get('noise_p', 0) or 0)
    if p > 0 and random.random() < p:
        lo, hi = cfg.get('snr_db', [15, 35])
        info['snr'] = random.uniform(float(lo), float(hi)); info['pink'] = random.random() < 0.5
        y = add_noise(y, info['snr'], info['pink'])
    if info:
        y = np.asarray(y, dtype=np.float32)
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > 1.0:                      # keep the waveform in range without changing its shape
            y = y / peak
        assert len(y) == n0
    return y, info
