"""FreqFlux audio DSP core.

Turns an audio signal (file or live mic) into per-video-frame control signals:

    sub_bass   20-100 Hz      -> scene scale, camera rumble, bloom
    mid        250-2000 Hz    -> shape morphing / geometry
    treble     4-16 kHz       -> particle emission
    onset      spectral flux  -> color shifts, camera cuts
    brightness / hue          -> spectral centroid mapped to a hue sweep

Sync model
----------
Frame i is anchored to sample  floor(i * fs / fps)  computed in exact integer
arithmetic (fps is a rational number). There is no accumulated float error, so
frame 60*3600 of a 60 FPS render sits on sample 44100*3600 exactly, even when
fs / fps is not an integer (for example 59.94 FPS).

Offline analysis uses a window centered on the frame's sample position. The
live analyzer is causal (window ends "now").
"""

from __future__ import annotations

import argparse
import threading
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.fft as sfft
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, get_window

EPS = 1e-12

BANDS: Dict[str, Tuple[float, float]] = {
    "sub_bass": (20.0, 100.0),
    "mid": (250.0, 2000.0),
    "treble": (4000.0, 16000.0),
}
BAND_NAMES: Tuple[str, ...] = ("sub_bass", "mid", "treble")

# (attack_ms, release_ms) per band envelope follower
BAND_ENVELOPE: Dict[str, Tuple[float, float]] = {
    "sub_bass": (5.0, 140.0),
    "mid": (12.0, 160.0),
    "treble": (3.0, 90.0),
}

# Flux is computed per frequency group and normalized per group, then merged
# with max(). A full-band mean would let broadband hats bury a 55 Hz kick.
FLUX_GROUPS: Dict[str, Tuple[float, float]] = {
    "low": (20.0, 250.0),
    "mid": (250.0, 4000.0),
    "high": (4000.0, 16000.0),
}
N_FLUX_GROUPS = len(FLUX_GROUPS)
FLUX_GAMMA = 1000.0        # log compression strength for spectral flux
FLUX_FLOOR = 0.01          # minimum normalization divisor for flux
ONSET_WINDOW_S = 0.5       # adaptive threshold window
ONSET_REFRACTORY_MS = 90.0
ONSET_PULSE_RELEASE_MS = 150.0
ONSET_ABS_FLOOR = 0.05
GAIN_CEIL = 2.0            # band values may overshoot 1.0 up to this with gain
SILENCE_MAG_SUM = 1e-4     # below this the centroid is considered undefined

# Spectrum and pitch features for the visuals
N_SPEC = 48                          # log-spaced bands
SPEC_FMIN, SPEC_FMAX = 40.0, 12000.0
SPEC_EDGES = np.geomspace(SPEC_FMIN, SPEC_FMAX, N_SPEC + 1)
N_HIST = 48                          # frames of spectrum history given to the renderer
CHROMA_FMIN, CHROMA_FMAX = 110.0, 4000.0
EVENT_NAMES = ("kick", "snare", "hat")   # onset streams from the low, mid, high flux groups

FpsLike = Union[int, float, str, Fraction]


# --------------------------------------------------------------------------- #
# Exact frame <-> sample clock
# --------------------------------------------------------------------------- #
class FrameClock:
    """Drift-free mapping between video frame indices and audio samples."""

    def __init__(self, fs: int, fps: FpsLike = 60):
        self.fs = int(fs)
        f = Fraction(fps) if not isinstance(fps, float) else Fraction(fps).limit_denominator(1001)
        if f <= 0:
            raise ValueError("fps must be positive")
        self.fps = f
        self._num = f.numerator
        self._den = f.denominator

    @property
    def fps_float(self) -> float:
        return float(self.fps)

    @property
    def samples_per_frame(self) -> Fraction:
        """Exact fs / fps (N_samples_per_frame)."""
        return Fraction(self.fs) / self.fps

    def sample_at(self, i: int) -> int:
        """Sample index at the start of frame i: floor(i * fs / fps)."""
        return (int(i) * self.fs * self._den) // self._num

    def samples_at(self, idx: np.ndarray) -> np.ndarray:
        """Vectorized sample_at (int64, exact for any realistic track length)."""
        idx = np.asarray(idx, dtype=np.int64)
        return (idx * (self.fs * self._den)) // self._num

    def sample_range(self, i: int) -> Tuple[int, int]:
        """Half-open [start, end) sample span owned by frame i. Spans tile the
        track with no gaps or overlaps."""
        return self.sample_at(i), self.sample_at(i + 1)

    def time_at(self, i: int) -> float:
        return float(Fraction(int(i)) / self.fps)

    def frames_for(self, n_samples: int) -> int:
        """Number of frames needed to cover n_samples: ceil(n * fps / fs)."""
        return -((-int(n_samples) * self._num) // (self.fs * self._den))


# --------------------------------------------------------------------------- #
# Envelope follower (asymmetric one-pole)
# --------------------------------------------------------------------------- #
class EnvelopeFollower:
    """Per-frame attack/release smoothing. attack_ms=0 gives instant attack,
    which turns it into a decaying peak-hold (used for onset pulses)."""

    def __init__(self, fps: float, attack_ms: float, release_ms: float, y0: float = 0.0):
        self.a_att = self._coef(attack_ms, fps)
        self.a_rel = self._coef(release_ms, fps)
        self.y = float(y0)

    @staticmethod
    def _coef(ms: float, fps: float) -> float:
        tau = ms / 1000.0
        return 0.0 if tau <= 0.0 else float(np.exp(-1.0 / (tau * fps)))

    def step(self, x: float) -> float:
        c = self.a_att if x > self.y else self.a_rel
        self.y = c * self.y + (1.0 - c) * x
        return self.y

    def process(self, arr: np.ndarray) -> np.ndarray:
        out = np.empty(len(arr), dtype=np.float32)
        for k, x in enumerate(arr):
            out[k] = self.step(float(x))
        return out


# --------------------------------------------------------------------------- #
# Spectral core shared by offline and live paths
# --------------------------------------------------------------------------- #
class SpectralCore:
    """Windowed spectra, band energies, centroid and log-spectral flux.

    Two resolutions are used on purpose:
      n_fft  (default 4096, ~93 ms @ 44.1k): fine frequency resolution for the
             20-100 Hz band and the centroid.
      n_flux (default 1024, ~23 ms): sharp time resolution for onsets.
    """

    def __init__(self, fs: int, n_fft: int = 4096, n_flux: int = 1024):
        if fs < 32000:
            raise ValueError("fs must be >= 32 kHz so the 4-16 kHz band is valid")
        if n_flux > n_fft:
            raise ValueError("n_flux must be <= n_fft")
        self.fs = int(fs)
        self.n_fft = int(n_fft)
        self.n_flux = int(n_flux)
        self.win = self._hann(self.n_fft)
        self.win_flux = self._hann(self.n_flux)
        self.freqs = sfft.rfftfreq(self.n_fft, 1.0 / self.fs).astype(np.float32)
        self.band_idx: Dict[str, Tuple[int, int]] = {}
        for name, (lo, hi) in BANDS.items():
            a = int(np.searchsorted(self.freqs, lo, side="left"))
            b = int(np.searchsorted(self.freqs, hi, side="right"))
            if b <= a:
                raise ValueError(f"band {name} has no FFT bins; increase n_fft")
            self.band_idx[name] = (a, b)
        freqs_flux = sfft.rfftfreq(self.n_flux, 1.0 / self.fs)
        self.flux_idx: List[Tuple[int, int]] = []
        for name, (lo, hi) in FLUX_GROUPS.items():
            a = max(1, int(np.searchsorted(freqs_flux, lo, side="left")))
            b = int(np.searchsorted(freqs_flux, hi, side="right"))
            if b <= a:
                raise ValueError(f"flux group {name} has no bins; increase n_flux")
            self.flux_idx.append((a, b))
        self.spec_w = self._build_spec_matrix()
        self.chroma_w = self._build_chroma_matrix()

    def _build_spec_matrix(self) -> np.ndarray:
        """(N_SPEC, bins) power-averaging weights. Bands narrower than one FFT
        bin interpolate between the neighbouring bins instead of repeating one."""
        f = self.freqs.astype(np.float64)
        binw = f[1] - f[0]
        W = np.zeros((N_SPEC, len(f)))
        for k in range(N_SPEC):
            lo, hi = SPEC_EDGES[k], SPEC_EDGES[k + 1]
            if hi - lo >= binw:
                sel = (f >= lo) & (f < hi)
                if sel.any():
                    W[k, sel] = 1.0 / sel.sum()
                    continue
            c = np.sqrt(lo * hi)
            j = int(np.searchsorted(f, c))
            j0, j1 = max(j - 1, 0), min(j, len(f) - 1)
            t = 0.0 if j1 == j0 else (c - f[j0]) / (f[j1] - f[j0])
            W[k, j0] += 1.0 - t
            W[k, j1] += t
        return W.astype(np.float32)

    def _build_chroma_matrix(self) -> np.ndarray:
        """(12, bins): each FFT bin votes for its nearest pitch class, weighted by
        how close it sits to that semitone (bins between two notes count less)."""
        f = self.freqs.astype(np.float64)
        W = np.zeros((12, len(f)))
        for i, fr in enumerate(f):
            if CHROMA_FMIN <= fr <= CHROMA_FMAX:
                midi = 69.0 + 12.0 * np.log2(fr / 440.0)
                near = round(midi)
                W[int(near) % 12, i] = np.exp(-((midi - near) ** 2) / (2 * 0.25 ** 2))
        return W.astype(np.float32)

    def spec_db(self, mag: np.ndarray) -> np.ndarray:
        """(N, N_SPEC) log-frequency band energy in dB."""
        return (10.0 * np.log10((mag * mag) @ self.spec_w.T + EPS)).astype(np.float32)

    def chroma_raw(self, mag: np.ndarray) -> np.ndarray:
        """(N, 12) unnormalized pitch-class energy (C=0 ... B=11)."""
        return (mag @ self.chroma_w.T).astype(np.float32)

    @staticmethod
    def _hann(n: int) -> np.ndarray:
        w = get_window("hann", n, fftbins=True)
        return (w / w.sum()).astype(np.float32)

    def spectrum(self, frames: np.ndarray) -> np.ndarray:
        """frames: (N, n_fft) -> magnitude (N, n_fft//2+1). Amplitude-normalized:
        a full-scale sine peaks near 0.5."""
        return np.abs(sfft.rfft(frames * self.win, axis=-1)).astype(np.float32)

    def band_db(self, mag: np.ndarray) -> np.ndarray:
        """Band energy in dB, shape (N, 3) ordered as BAND_NAMES."""
        p = mag * mag
        out = np.empty((mag.shape[0], len(BAND_NAMES)), dtype=np.float32)
        for k, name in enumerate(BAND_NAMES):
            a, b = self.band_idx[name]
            out[:, k] = 10.0 * np.log10(p[:, a:b].sum(axis=1) + EPS)
        return out

    def centroid(self, mag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Spectral centroid in Hz and the magnitude sum (for silence gating).
        DC bin excluded."""
        m = mag[:, 1:]
        den = m.sum(axis=1)
        cen = (m @ self.freqs[1:]) / (den + EPS)
        return cen.astype(np.float32), den.astype(np.float32)

    def log_spectrum_flux(self, frames: np.ndarray) -> np.ndarray:
        """frames: (N, n_flux) -> log-compressed magnitude (N, n_flux//2+1)."""
        mag = np.abs(sfft.rfft(frames * self.win_flux, axis=-1)).astype(np.float32)
        return np.log1p(FLUX_GAMMA * mag)

    def flux(self, lm_cur: np.ndarray, lm_prev: np.ndarray) -> np.ndarray:
        """Half-wave rectified spectral flux per frequency group, averaged over
        the group's bins. Shape (N, N_FLUX_GROUPS)."""
        d = np.maximum(lm_cur - lm_prev, 0.0)
        return np.stack([d[:, a:b].mean(axis=-1) for a, b in self.flux_idx], axis=-1)


# --------------------------------------------------------------------------- #
# Onset detection (shared threshold logic)
# --------------------------------------------------------------------------- #
def _onset_k(sensitivity: float) -> float:
    """Threshold multiplier on local std. sensitivity 1.0 -> 0.5, 0.0 -> 2.5."""
    s = min(max(float(sensitivity), 0.0), 1.0)
    return 2.5 - 2.0 * s


def _onset_strength(flux: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return np.clip(flux / (thr + 1e-6) - 1.0 + 0.25, 0.25, 1.0)


def detect_onsets(flux: np.ndarray, fps: float, sensitivity: float = 0.5) -> np.ndarray:
    """Offline adaptive-threshold peak picking on normalized flux.

    Returns an array (len(flux)) that is 0 everywhere except at onset frames,
    where it holds a strength in [0.25, 1].
    """
    n = len(flux)
    out = np.zeros(n, dtype=np.float32)
    if n < 3:
        return out
    w = max(3, int(round(ONSET_WINDOW_S * fps)))
    f = flux.astype(np.float64)
    mean = uniform_filter1d(f, w, mode="nearest")
    sq = uniform_filter1d(f * f, w, mode="nearest")
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    thr = mean + _onset_k(sensitivity) * std + ONSET_ABS_FLOOR
    dist = max(1, int(round(ONSET_REFRACTORY_MS * fps / 1000.0)))
    peaks, _ = find_peaks(f, height=thr, distance=dist)
    if len(peaks):
        out[peaks] = _onset_strength(f[peaks], thr[peaks]).astype(np.float32)
    return out


class CausalOnsetDetector:
    """Frame-by-frame version for live input (no look-ahead)."""

    def __init__(self, fps: float, sensitivity: float = 0.5):
        self.fps = float(fps)
        self.sensitivity = sensitivity
        self.hist: deque = deque(maxlen=max(5, int(round(ONSET_WINDOW_S * fps))))
        self.refractory = max(1, int(round(ONSET_REFRACTORY_MS * fps / 1000.0)))
        self.cooldown = 0
        self.prev = 0.0

    def update(self, f: float) -> float:
        strength = 0.0
        if self.cooldown > 0:
            self.cooldown -= 1
        elif len(self.hist) >= 5:
            h = np.fromiter(self.hist, dtype=np.float64)
            thr = h.mean() + _onset_k(self.sensitivity) * h.std() + ONSET_ABS_FLOOR
            if f > thr and f >= self.prev:
                strength = float(_onset_strength(np.array([f]), np.array([thr]))[0])
                self.cooldown = self.refractory
        self.hist.append(f)
        self.prev = f
        return strength


# --------------------------------------------------------------------------- #
# Public data structures
# --------------------------------------------------------------------------- #
@dataclass
class ControlParams:
    """UI sensitivity knobs. Gains multiply the 0..1 band values (clipped at
    GAIN_CEIL). onset_sensitivity is 0..1."""
    bass_gain: float = 1.0
    mid_gain: float = 1.0
    treble_gain: float = 1.0
    onset_sensitivity: float = 0.5


@dataclass(frozen=True)
class ControlFrame:
    index: int
    time: float
    sub_bass: float       # 0..GAIN_CEIL, smoothed
    mid: float
    treble: float
    flux: float           # normalized spectral flux (unsmoothed)
    onset: float          # 0 or strength 0.25..1 on the onset frame only
    onset_pulse: float    # onset with instant attack and ~150 ms decay
    centroid_hz: float
    brightness: float     # 0..1, log-mapped centroid (150 Hz .. 6 kHz)
    hue: float            # 0..1 HSV hue sweep: low brightness red, high violet
    kick: float = 0.0     # onset strength on the frame a kick (low band hit) lands, else 0
    snare: float = 0.0    # same for the mid band
    hat: float = 0.0      # same for the high band
    beat: float = 0.0     # continuous beat position (beats since the start of the track)
    chroma: Optional[np.ndarray] = field(default=None, compare=False, repr=False)      # (12,) 0..1, C..B
    spec_hist: Optional[np.ndarray] = field(default=None, compare=False, repr=False)   # (N_HIST, N_SPEC) uint8, row 0 = now


def _brightness_from_centroid(c_hz: np.ndarray) -> np.ndarray:
    return np.clip(np.log2(np.maximum(c_hz, 1.0) / 150.0) / np.log2(6000.0 / 150.0), 0.0, 1.0)


def _hue_from_brightness(b):
    return (0.02 + 0.80 * b) % 1.0


@dataclass
class FeatureTrack:
    """Precomputed per-frame features for a whole track."""
    clock: FrameClock
    n_samples: int
    n_frames: int
    bands: np.ndarray            # (3, n_frames) smoothed, 0..1
    flux: np.ndarray             # (n_frames,) normalized
    centroid_hz: np.ndarray      # (n_frames,)
    brightness: np.ndarray       # (n_frames,) smoothed 0..1
    hue: np.ndarray              # (n_frames,)
    spec: np.ndarray             # (n_frames, N_SPEC) uint8, per-band normalized log spectrum
    chroma: np.ndarray           # (n_frames, 12) uint8, C..B
    flux_groups: np.ndarray      # (n_frames, 3) float32 normalized flux: low, mid, high
    beat_pos: np.ndarray         # (n_frames,) float32 continuous beat position
    tempo: float = 120.0
    _event_cache: Dict[float, np.ndarray] = field(default_factory=dict, repr=False)
    _onset_cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return self.n_frames

    @property
    def duration(self) -> float:
        return self.n_samples / self.clock.fs

    def onsets(self, sensitivity: float) -> Tuple[np.ndarray, np.ndarray]:
        """(strength impulses, decaying pulse) for a sensitivity, cached."""
        key = round(float(sensitivity), 3)
        hit = self._onset_cache.get(key)
        if hit is None:
            strength = detect_onsets(self.flux, self.clock.fps_float, key)
            pulse = EnvelopeFollower(self.clock.fps_float, 0.0, ONSET_PULSE_RELEASE_MS).process(strength)
            hit = (strength, pulse)
            self._onset_cache[key] = hit
        return hit

    def events(self, sensitivity: float) -> np.ndarray:
        """(n_frames, 4) float32: kick, snare, hat, any. Each column is 0 except on
        frames where that stream has an onset (strength 0.25..1). Cached."""
        key = round(float(sensitivity), 3)
        hit = self._event_cache.get(key)
        if hit is None:
            fps = self.clock.fps_float
            cols = [detect_onsets(self.flux_groups[:, g], fps, key) for g in range(3)]
            hit = np.stack(cols + [self.onsets(key)[0]], axis=1).astype(np.float32)
            self._event_cache[key] = hit
        return hit

    def frame(self, i: int, params: Optional[ControlParams] = None) -> ControlFrame:
        p = params or ControlParams()
        i = int(min(max(i, 0), self.n_frames - 1))
        strength, pulse = self.onsets(p.onset_sensitivity)
        g = (p.bass_gain, p.mid_gain, p.treble_gain)
        sb, md, tr = (float(min(self.bands[k, i] * g[k], GAIN_CEIL)) for k in range(3))
        ev = self.events(p.onset_sensitivity)[i]
        hist_idx = np.clip(i - np.arange(N_HIST), 0, self.n_frames - 1)
        return ControlFrame(
            index=i,
            time=self.clock.time_at(i),
            sub_bass=sb,
            mid=md,
            treble=tr,
            flux=float(self.flux[i]),
            onset=float(strength[i]),
            onset_pulse=float(pulse[i]),
            centroid_hz=float(self.centroid_hz[i]),
            brightness=float(self.brightness[i]),
            hue=float(self.hue[i]),
            kick=float(ev[0]), snare=float(ev[1]), hat=float(ev[2]),
            beat=float(self.beat_pos[i]),
            chroma=self.chroma[i].astype(np.float32) / 255.0,
            spec_hist=self.spec[hist_idx],
        )


# --------------------------------------------------------------------------- #
# Offline analysis
# --------------------------------------------------------------------------- #
def load_audio(path: str, fs: int = 44100) -> np.ndarray:
    """Decode any librosa/soundfile/ffmpeg-readable file to mono float32 at fs."""
    import librosa  # lazy: keeps the live/self-test path light

    y, _ = librosa.load(path, sr=fs, mono=True)
    return np.ascontiguousarray(y, dtype=np.float32)


def _percentile_normalize(x: np.ndarray, lo_pct: float, hi_pct: float, min_span: float) -> np.ndarray:
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    hi = max(hi, lo + min_span)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _follow_cols(x: np.ndarray, fps: float, attack_s: float, release_s: float) -> np.ndarray:
    """Asymmetric one-pole smoothing along time (axis 0) for every column at once."""
    ca = 0.0 if attack_s <= 0 else float(np.exp(-1.0 / (attack_s * fps)))
    cr = float(np.exp(-1.0 / (release_s * fps)))
    out = np.empty_like(x, dtype=np.float32)
    y = x[0].astype(np.float32).copy()
    for i in range(len(x)):
        c = np.where(x[i] > y, ca, cr)
        y = c * y + (1.0 - c) * x[i]
        out[i] = y
    return out


def _beat_positions(env: np.ndarray, fps: float) -> Tuple[np.ndarray, float]:
    """Continuous beat position per frame (1.0 = one beat) from librosa's beat
    tracker run on a kick/snare weighted onset envelope. Falls back to a steady
    120 BPM grid when no stable beat is found."""
    n = len(env)
    beats = np.array([])
    tempo = 120.0
    try:
        import librosa

        t, b = librosa.beat.beat_track(onset_envelope=env.astype(np.float32), sr=fps, hop_length=1, tightness=100)
        tempo = float(np.atleast_1d(t)[0])
        beats = np.asarray(b, dtype=np.float64)
    except Exception:  # noqa: BLE001
        pass
    if len(beats) < 4:
        return (np.arange(n) / fps * 2.0).astype(np.float32), 120.0
    iv = float(np.median(np.diff(beats)))
    idx = np.arange(n, dtype=np.float64)
    pos = np.interp(idx, beats, np.arange(len(beats), dtype=np.float64))
    before, after = idx < beats[0], idx > beats[-1]
    pos[before] = (idx[before] - beats[0]) / iv
    pos[after] = (len(beats) - 1) + (idx[after] - beats[-1]) / iv
    return pos.astype(np.float32), tempo


def analyze(
    y: np.ndarray,
    fs: int = 44100,
    fps: FpsLike = 60,
    n_fft: int = 4096,
    n_flux: int = 1024,
    chunk: int = 256,
    progress: Optional[Callable[[float], None]] = None,
) -> FeatureTrack:
    """Analyze a whole mono signal into a FeatureTrack with exactly one feature
    sample per video frame."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError("analyze() expects mono audio")
    clock = FrameClock(fs, fps)
    n = len(y)
    if n == 0:
        raise ValueError("empty audio")
    nf = clock.frames_for(n)
    core = SpectralCore(fs, n_fft, n_flux)

    pad = n_fft
    yp = np.pad(y, (pad, pad))
    pos_all = clock.samples_at(np.arange(nf))
    ar_main = np.arange(n_fft)
    ar_flux = np.arange(n_flux)

    db = np.empty((nf, 3), dtype=np.float32)
    cen = np.empty(nf, dtype=np.float32)
    den = np.empty(nf, dtype=np.float32)
    flux = np.empty((nf, N_FLUX_GROUPS), dtype=np.float32)
    spec_db = np.empty((nf, N_SPEC), dtype=np.float32)
    chroma_raw = np.empty((nf, 12), dtype=np.float32)
    prev_lm: Optional[np.ndarray] = None

    for s in range(0, nf, chunk):
        e = min(nf, s + chunk)
        pos = pos_all[s:e]
        mag = core.spectrum(yp[(pad + pos - n_fft // 2)[:, None] + ar_main])
        db[s:e] = core.band_db(mag)
        cen[s:e], den[s:e] = core.centroid(mag)
        spec_db[s:e] = core.spec_db(mag)
        chroma_raw[s:e] = core.chroma_raw(mag)

        lm = core.log_spectrum_flux(yp[(pad + pos - n_flux // 2)[:, None] + ar_flux])
        if prev_lm is None:
            ref = np.vstack([lm[:1], lm[:-1]])
        else:
            ref = np.vstack([prev_lm[None, :], lm[:-1]])
        flux[s:e] = core.flux(lm, ref)
        prev_lm = lm[-1]
        if progress:
            progress(0.9 * e / nf)

    fpsf = clock.fps_float

    # Bands: percentile-normalize across the track, then attack/release smooth.
    bands = np.empty((3, nf), dtype=np.float32)
    for k, name in enumerate(BAND_NAMES):
        norm = _percentile_normalize(db[:, k], 5.0, 99.5, 12.0)
        att, rel = BAND_ENVELOPE[name]
        bands[k] = EnvelopeFollower(fpsf, att, rel).process(norm)

    # Flux: scale each group so its 99th percentile is ~1, merge with max().
    p99 = np.maximum(np.percentile(flux, 99.0, axis=0), FLUX_FLOOR)
    fg = np.minimum(flux / p99, 3.0).astype(np.float32)
    flux_n = fg.max(axis=1).astype(np.float32)

    # Spectrum: per-band percentile normalization (so a note appearing in a band
    # lights that band), then a fast attack / short release for visible motion.
    spec_n = np.empty_like(spec_db)
    for k in range(N_SPEC):
        spec_n[:, k] = _percentile_normalize(spec_db[:, k], 8.0, 99.5, 10.0)
    spec_s = _follow_cols(spec_n, fpsf, 0.012, 0.12)

    # Chroma: per-frame max normalization, gated so silence shows nothing.
    cmax = chroma_raw.max(axis=1, keepdims=True)
    cn = chroma_raw / (cmax + EPS)
    lvl = den / max(float(np.percentile(den, 90.0)), 1e-6)
    gate = np.clip((lvl - 0.02) / 0.10, 0.0, 1.0)[:, None]
    chroma_s = _follow_cols((cn * gate).astype(np.float32), fpsf, 0.03, 0.20)

    # Beat grid from the kick / snare weighted onset envelope.
    beat_pos, tempo = _beat_positions(np.maximum(fg[:, 0], 0.6 * fg[:, 1]), fpsf)
    if progress:
        progress(1.0)

    # Centroid: forward-fill silent frames, map to brightness, smooth, hue.
    valid = den > SILENCE_MAG_SUM
    if valid.any():
        first = int(np.argmax(valid))
        idx = np.where(valid, np.arange(nf), first)
        np.maximum.accumulate(idx, out=idx)
        cen = cen[idx]
    else:
        cen[:] = 1000.0
    bright = EnvelopeFollower(fpsf, 120.0, 120.0, y0=float(_brightness_from_centroid(cen[:1])[0])).process(
        _brightness_from_centroid(cen)
    )
    hue = _hue_from_brightness(bright).astype(np.float32)

    return FeatureTrack(
        clock=clock,
        n_samples=n,
        n_frames=nf,
        bands=bands,
        flux=flux_n,
        centroid_hz=cen,
        brightness=bright,
        hue=hue,
        spec=np.round(np.clip(spec_s, 0.0, 1.0) * 255.0).astype(np.uint8),
        chroma=np.round(np.clip(chroma_s, 0.0, 1.0) * 255.0).astype(np.uint8),
        flux_groups=fg,
        beat_pos=beat_pos,
        tempo=tempo,
    )


# --------------------------------------------------------------------------- #
# Live input
# --------------------------------------------------------------------------- #
class _PeakTracker:
    """Live level normalizer: peak with slow dB/s decay, fixed dynamic span."""

    def __init__(self, fps: float, span_db: float = 36.0, decay_db_per_s: float = 2.5, hi: float = -30.0):
        self.span = span_db
        self.step = decay_db_per_s / fps
        self.hi = hi

    def update(self, x_db: float) -> float:
        self.hi = max(x_db, self.hi - self.step, -80.0)
        return float(np.clip((x_db - (self.hi - self.span)) / self.span, 0.0, 1.0))


def list_input_devices() -> List[str]:
    import sounddevice as sd

    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append(f"{i}: {d['name']} ({d['max_input_channels']} ch)")
    return out


class LiveAnalyzer:
    """Causal analyzer fed by a sounddevice InputStream.

    Call poll() once per rendered frame. Features are computed from the most
    recent samples, so results do not depend on how often poll() is called.
    """

    def __init__(
        self,
        fs: int = 44100,
        fps: FpsLike = 60,
        device: Optional[Union[int, str]] = None,
        n_fft: int = 4096,
        n_flux: int = 1024,
        blocksize: int = 512,
        params: Optional[ControlParams] = None,
    ):
        self.clock = FrameClock(fs, fps)
        self.fs = int(fs)
        self.fpsf = self.clock.fps_float
        self.core = SpectralCore(fs, n_fft, n_flux)
        self.device = device
        self.blocksize = int(blocksize)
        self.params = params or ControlParams()
        self.hop = int(round(self.fs / self.fpsf))
        self._L = 1 << 15
        if self._L < n_fft + self.hop:
            raise ValueError("ring buffer too small for n_fft + hop")
        self._buf = np.zeros(self._L, dtype=np.float32)
        self._w = 0
        self._lock = threading.Lock()
        self._stream = None
        self._count = 0

        self._peaks = [_PeakTracker(self.fpsf) for _ in BAND_NAMES]
        self._env = [EnvelopeFollower(self.fpsf, *BAND_ENVELOPE[nm]) for nm in BAND_NAMES]
        self._flux_peak = np.full(N_FLUX_GROUPS, FLUX_FLOOR)
        self._flux_decay = 0.8 ** (1.0 / self.fpsf)
        self._onset = CausalOnsetDetector(self.fpsf, self.params.onset_sensitivity)
        self._pulse = EnvelopeFollower(self.fpsf, 0.0, ONSET_PULSE_RELEASE_MS)
        self._bright = EnvelopeFollower(self.fpsf, 120.0, 120.0, y0=0.3)
        self._last_cen = 1000.0
        self._prev_lm_cache: Optional[np.ndarray] = None

    # -- stream control -----------------------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:
        mono = indata[:, 0]
        n = len(mono)
        with self._lock:
            w = self._w % self._L
            first = min(n, self._L - w)
            self._buf[w:w + first] = mono[:first]
            if n > first:
                self._buf[:n - first] = mono[first:]
            self._w += n

    def start(self) -> None:
        import sounddevice as sd

        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.fs,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "LiveAnalyzer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- feeding without a device (tests, browser-mic chunks) ----------------
    def feed(self, samples: np.ndarray) -> None:
        """Push mono float32 samples as if they came from the device."""
        s = np.asarray(samples, dtype=np.float32).reshape(-1)
        for k in range(0, len(s), self._L // 2):
            part = s[k:k + self._L // 2][:, None]
            self._callback(part, len(part), None, None)

    def _latest(self, n: int) -> np.ndarray:
        with self._lock:
            end = self._w
            have = min(end, n)
            idx = (np.arange(end - have, end) % self._L)
            data = self._buf[idx].copy()
        if have < n:
            data = np.concatenate([np.zeros(n - have, dtype=np.float32), data])
        return data

    # -- per-frame features --------------------------------------------------
    def poll(self) -> ControlFrame:
        core = self.core
        chunk = self._latest(core.n_fft + self.hop)
        main = chunk[-core.n_fft:]
        cur = chunk[-core.n_flux:]
        prev = chunk[-core.n_flux - self.hop:-self.hop]

        mag = core.spectrum(main[None, :])
        db = core.band_db(mag)[0]
        cen, den = core.centroid(mag)
        if den[0] > SILENCE_MAG_SUM:
            self._last_cen = float(cen[0])

        lm = core.log_spectrum_flux(np.stack([prev, cur]))
        fl = core.flux(lm[1:2], lm[0:1])[0]
        self._flux_peak = np.maximum(np.maximum(fl, self._flux_peak * self._flux_decay), FLUX_FLOOR)
        fl_n = float(min((fl / self._flux_peak).max(), 3.0))

        self._onset.sensitivity = self.params.onset_sensitivity
        strength = self._onset.update(fl_n)
        pulse = self._pulse.step(strength)

        vals = []
        g = (self.params.bass_gain, self.params.mid_gain, self.params.treble_gain)
        for k in range(3):
            norm = self._peaks[k].update(float(db[k]))
            vals.append(float(min(self._env[k].step(norm) * g[k], GAIN_CEIL)))

        b = self._bright.step(float(_brightness_from_centroid(np.array([self._last_cen]))[0]))
        idx = self._count
        self._count += 1
        return ControlFrame(
            index=idx,
            time=self._w / self.fs,
            sub_bass=vals[0],
            mid=vals[1],
            treble=vals[2],
            flux=fl_n,
            onset=strength,
            onset_pulse=float(pulse),
            centroid_hz=self._last_cen,
            brightness=float(b),
            hue=float(_hue_from_brightness(b)),
        )


# --------------------------------------------------------------------------- #
# Self-test / CLI
# --------------------------------------------------------------------------- #
def _synthetic_track(fs: int, seconds: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """120 BPM kick (55 Hz) on the beat, noise hat on the off-beat, steady mid
    pad. Returns (signal, event_times_seconds)."""
    rng = np.random.default_rng(7)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    y = 0.05 * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 880 * t))
    events = []
    for k in range(int(seconds / 0.25)):
        t0 = k * 0.25
        i0 = int(t0 * fs)
        events.append(t0)
        if k % 2 == 0:
            m = int(0.15 * fs)
            tt = np.arange(m) / fs
            y[i0:i0 + m] += (0.8 * np.sin(2 * np.pi * 55 * tt) * np.exp(-tt * 18))[: n - i0]
        else:
            m = int(0.03 * fs)
            hat = np.diff(rng.standard_normal(m + 1)) * np.exp(-np.arange(m) / (0.008 * fs))
            y[i0:i0 + m] += (0.25 * hat)[: n - i0]
    return y.astype(np.float32), np.array(events)


def _selftest(fs: int = 44100) -> None:
    print("== Frame clock exactness ==")
    for fps in (60, Fraction(60000, 1001), 30, 24):
        c = FrameClock(fs, fps)
        i = int(c.fps * 3600) if c.fps.denominator == 1 else c.fps.numerator * 3600 // c.fps.denominator
        exact = Fraction(i) * fs / c.fps
        ok = c.sample_at(i) == int(exact) and sum(
            c.sample_range(k)[1] - c.sample_range(k)[0] for k in range(0, 5000)
        ) == c.sample_at(5000)
        print(f"fps={float(c.fps):9.5f}  samples/frame={float(c.samples_per_frame):10.5f}  "
              f"frame {i} -> sample {c.sample_at(i)} (exact {float(exact):.3f})  {'OK' if ok else 'FAIL'}")

    print("\n== Offline analysis on synthetic 120 BPM track ==")
    y, events = _synthetic_track(fs)
    ft = analyze(y, fs, 60)
    strength, _ = ft.onsets(0.5)
    times = np.nonzero(strength)[0] / 60.0
    err = np.array([np.min(np.abs(events - t)) for t in times]) * 1000.0
    print(f"frames={ft.n_frames}  expected events={len(events)}  detected={len(times)}  "
          f"median timing error={np.median(err):.1f} ms  max={err.max():.1f} ms")
    kick_frames = (events[::2] * 60).astype(int)
    off_frames = (events[1::2] * 60).astype(int)
    print(f"sub_bass mean on kicks={ft.bands[0, kick_frames + 2].mean():.2f}  "
          f"off-beats={ft.bands[0, off_frames + 2].mean():.2f}")
    print(f"treble   mean on hats ={ft.bands[2, off_frames + 1].mean():.2f}  "
          f"kicks={ft.bands[2, kick_frames + 1].mean():.2f}")
    print(f"centroid median={np.median(ft.centroid_hz):.0f} Hz  hue median={np.median(ft.hue):.2f}")

    print("\n== Live analyzer (fed, no device) ==")
    live = LiveAnalyzer(fs, 60)
    hop = live.hop
    n_on = 0
    last = None
    for k in range(0, len(y) - hop, hop):
        live.feed(y[k:k + hop])
        last = live.poll()
        n_on += last.onset > 0
    print(f"live onsets={n_on}  final frame: sub_bass={last.sub_bass:.2f} mid={last.mid:.2f} "
          f"treble={last.treble:.2f} hue={last.hue:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="FreqFlux DSP core")
    ap.add_argument("audio", nargs="?", help="audio file to analyze")
    ap.add_argument("--fps", default="60")
    ap.add_argument("--fs", type=int, default=44100)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--devices", action="store_true", help="list input devices")
    a = ap.parse_args()

    if a.devices:
        print("\n".join(list_input_devices()))
        return
    if a.selftest or not a.audio:
        _selftest(a.fs)
        return
    y = load_audio(a.audio, a.fs)
    ft = analyze(y, a.fs, Fraction(a.fps))
    s, _ = ft.onsets(0.5)
    print(f"{a.audio}: {ft.duration:.2f}s  {ft.n_frames} frames @ {float(ft.clock.fps)} fps  "
          f"{int((s > 0).sum())} onsets  mean bands "
          f"sub={ft.bands[0].mean():.2f} mid={ft.bands[1].mean():.2f} tre={ft.bands[2].mean():.2f}")


if __name__ == "__main__":
    main()
