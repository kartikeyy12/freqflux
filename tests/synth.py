"""Synthetic music with known ground truth, used by the tests and the stress suite.

make_track() builds a loop with kicks, snares, hats and a melody of known notes,
and returns the audio plus every event time so the analysis can be scored.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

MELODY = [(440.0, 9), (523.25, 0), (659.25, 4), (783.99, 7), (698.46, 5), (659.25, 4), (587.33, 2), (523.25, 0)]


def make_track(
    fs: int = 44100,
    seconds: float = 24.0,
    bpm: float = 120.0,
    swing: float = 0.0,          # 0..0.33, delays the off-beat 8ths
    drums: bool = True,
    melody: bool = True,
    rubato: float = 0.0,         # 0..0.06, random per-beat timing drift (fraction of a beat)
    loud: bool = False,          # heavy saturation + noise floor, like a compressed master
    seed: int = 3,
) -> tuple[np.ndarray, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    y = np.zeros(n)
    spb = 60.0 / bpm
    kicks, snares, hats, notes = [], [], [], []
    drift = 0.0
    b = 0
    t_beat = 0.0
    while t_beat < seconds - spb:
        drift = 0.8 * drift + rng.normal(0, rubato * spb) if rubato else 0.0
        t0 = t_beat + drift
        i0 = max(0, int(t0 * fs))
        if drums:
            m = int(0.22 * fs); tt = np.arange(m) / fs; f = 55 + 60 * np.exp(-tt * 40)
            k = np.sin(2 * np.pi * np.cumsum(f) / fs) * np.exp(-tt * 14) * 0.9
            y[i0:i0 + m] += k[: n - i0]; kicks.append(t0)
            if b % 2 == 1:
                m = int(0.18 * fs); tt = np.arange(m) / fs
                sn = (rng.standard_normal(m) * 0.5 + np.sin(2 * np.pi * 190 * tt) * 0.5) * np.exp(-tt * 22) * 0.55
                y[i0:i0 + m] += sn[: n - i0]; snares.append(t0)
            th = t0 + spb * (0.5 + swing)
            ih = int(th * fs); m = int(0.04 * fs)
            if ih + m < n:
                h = np.diff(rng.standard_normal(m + 1)) * np.exp(-np.arange(m) / (0.008 * fs)) * 0.22
                y[ih:ih + m] += h; hats.append(th)
        if melody:
            fr, pc = MELODY[b % len(MELODY)]
            m = int(min(0.45, spb * 0.9) * fs); tt = np.arange(m) / fs
            env = np.minimum(tt / 0.008, 1) * np.exp(-tt * 3.5)
            note = (np.sin(2 * np.pi * fr * tt) + 0.3 * np.sin(2 * np.pi * 2 * fr * tt)) * env * 0.22
            y[i0:i0 + m] += note[: n - i0]; notes.append((t0, pc))
        b += 1
        t_beat += spb
    if loud:
        y = np.tanh(3.0 * y / max(1e-9, np.abs(y).max())) + 0.02 * rng.standard_normal(n)
    y = y / (max(1e-9, np.abs(y).max()) * 1.05)
    gt = dict(kicks=np.array(kicks), snares=np.array(snares), hats=np.array(hats), notes=notes, bpm=bpm, fs=fs)
    return y.astype(np.float32), gt


def write_wav(path: str, y: np.ndarray, fs: int = 44100) -> None:
    import scipy.io.wavfile as wf

    wf.write(path, fs, (np.stack([y, y], 1) * 32767).astype(np.int16))
