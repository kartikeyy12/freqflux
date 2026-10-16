"""Analysis checks against synthetic ground truth."""

from fractions import Fraction

import numpy as np
import pytest

from audio_dsp import N_HIST, N_SPEC, ControlParams, FrameClock, SpectralCore, analyze
from tests.synth import make_track

FS = 44100


# ---------------------------------------------------------------- frame clock
@pytest.mark.parametrize("fps", [60, 30, 24, Fraction(60000, 1001)])
def test_frame_clock_is_exact_and_gapless(fps):
    c = FrameClock(FS, fps)
    # frame i sits on floor(i * fs / fps), in exact rational arithmetic
    for i in (0, 1, 59, 60, 1000, 216000):
        assert c.sample_at(i) == int(Fraction(i) * FS / c.fps)
    # spans tile the track with no gaps or overlaps
    total = sum(c.sample_range(i)[1] - c.sample_range(i)[0] for i in range(5000))
    assert total == c.sample_at(5000)
    ends = [c.sample_range(i)[1] for i in range(100)]
    starts = [c.sample_range(i + 1)[0] for i in range(100)]
    assert ends == starts


def test_no_drift_after_an_hour():
    c = FrameClock(FS, 60)
    assert c.sample_at(60 * 3600) == FS * 3600
    c = FrameClock(FS, Fraction(60000, 1001))
    assert c.sample_at(60000) == FS * 1001      # 1001 s of audio is exactly 60000 frames


def test_frame_count_covers_audio():
    c = FrameClock(FS, 60)
    n = int(FS * 6.37)
    nf = c.frames_for(n)
    assert nf == 383
    assert c.sample_at(nf) >= n > c.sample_at(nf - 1)


def test_spectral_core_band_layout():
    core = SpectralCore(FS)
    assert core.spec_w.shape[0] == N_SPEC
    assert core.chroma_w.shape[0] == 12
    for name in ("sub_bass", "mid", "treble"):
        a, b = core.band_idx[name]
        assert b > a


def test_rejects_low_sample_rate():
    with pytest.raises(ValueError):
        SpectralCore(16000)


# ---------------------------------------------------------------- features
def test_feature_shapes(features, track):
    y, _ = track
    nf = features.n_frames
    assert nf == int(np.ceil(len(y) * 60 / FS))
    assert features.bands.shape == (3, nf)
    assert features.spec.shape == (nf, N_SPEC) and features.spec.dtype == np.uint8
    assert features.chroma.shape == (nf, 12)
    assert features.beat_pos.shape == (nf,)
    assert features.flux_groups.shape == (nf, 3)


def test_control_frame_contents(features):
    f = features.frame(600, ControlParams())
    assert f.spec_hist.shape == (N_HIST, N_SPEC)
    assert f.chroma.shape == (12,)
    assert 0.0 <= f.sub_bass <= 2.0 and 0.0 <= f.brightness <= 1.0
    # frames outside the track are clamped, not an error
    assert features.frame(-5).index == 0
    assert features.frame(10**6).index == features.n_frames - 1


def _match(detected_s, truth_s, tol=0.05):
    hit = np.array([np.min(np.abs(detected_s - t)) for t in truth_s])
    return int((hit < tol).sum())


def test_kick_events_land_on_the_kicks(features, track):
    _, gt = track
    ev = features.events(0.3)
    det = np.nonzero(ev[:, 0])[0] / 60.0
    truth = gt["kicks"]
    assert _match(det, truth) >= len(truth) - 2            # at most two misses
    err = np.array([np.min(np.abs(truth - d)) for d in det])
    assert np.median(err) < 0.02                            # timing within about one frame
    assert len(det) <= int(len(truth) * 1.2)               # not firing on everything


def test_snare_and_hat_events_are_found(features, track):
    _, gt = track
    ev = features.events(0.3)
    assert _match(np.nonzero(ev[:, 1])[0] / 60.0, gt["snares"]) == len(gt["snares"])
    assert _match(np.nonzero(ev[:, 2])[0] / 60.0, gt["hats"]) >= len(gt["hats"]) - 2


def test_sensitivity_changes_event_count(features):
    low = int((features.events(0.0)[:, 3] > 0).sum())
    high = int((features.events(1.0)[:, 3] > 0).sum())
    assert high >= low


def test_tempo_and_beat_grid(features, track):
    _, gt = track
    assert features.tempo == pytest.approx(120.0, abs=2.0)
    bp = features.beat_pos
    assert np.all(np.diff(bp) >= -1e-4)                     # beat position never runs backwards
    fr = np.clip((gt["kicks"][2:-2] * 60).round().astype(int), 0, features.n_frames - 1)
    phase = bp[fr] % 1.0
    # kicks sit on the beat grid (phase near 0 or 1)
    d = np.minimum(phase, 1.0 - phase)
    assert np.median(d) < 0.08


def test_pitch_classes_match_the_melody(features, track):
    _, gt = track
    ok = total = 0
    for t0, pc in gt["notes"][2:-2]:
        c = features.chroma[int((t0 + 0.2) * 60)]
        total += 1
        ok += int(np.argmax(c) == pc)
    assert ok / total >= 0.95


def test_spectrum_separates_kick_from_hat(features, track):
    _, gt = track
    kf = int(gt["kicks"][8] * 60) + 2
    hf = int(gt["hats"][8] * 60) + 1
    assert features.spec[kf, :10].mean() > features.spec[kf + 14, :10].mean()
    assert features.spec[hf, 36:].mean() > features.spec[hf + 8, 36:].mean()


# ---------------------------------------------------------------- robustness
@pytest.mark.parametrize(
    "kw", [dict(bpm=70), dict(bpm=90, swing=0.15), dict(bpm=128, loud=True), dict(bpm=174), dict(bpm=100, rubato=0.04)]
)
def test_beat_grid_stays_on_kicks_across_styles(kw):
    y, gt = make_track(FS, 24.0, **kw)
    ft = analyze(y, FS, 60)
    assert ft.tempo == pytest.approx(kw["bpm"], rel=0.04)
    fr = np.clip((gt["kicks"][2:-2] * 60).round().astype(int), 0, ft.n_frames - 1)
    ph = ft.beat_pos[fr] % 1.0
    conc = abs(np.mean(np.exp(2j * np.pi * ph)))
    assert conc > 0.9


def test_drumless_music_still_analyzes():
    y, _ = make_track(FS, 20.0, drums=False)
    ft = analyze(y, FS, 60)
    assert ft.n_frames == int(np.ceil(len(y) * 60 / FS))
    assert np.isfinite(ft.beat_pos).all()


def test_silence_does_not_crash_or_show_notes():
    ft = analyze(np.zeros(FS * 5, dtype=np.float32), FS, 60)
    assert ft.chroma.max() < 30                              # nothing sounding
    assert np.isfinite(ft.beat_pos).all()
    assert (ft.events(0.3)[:, :3] > 0).sum() <= 2


def test_very_short_audio():
    ft = analyze(np.random.default_rng(0).standard_normal(FS // 4).astype(np.float32) * 0.1, FS, 60)
    assert ft.n_frames >= 1 and np.isfinite(ft.beat_pos).all()
