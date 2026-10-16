"""Evaluate FreqFlux's analysis on real songs.

    python tools/evaluate.py ~/Music/test_songs            # a folder, or one or more files
    python tools/evaluate.py song.mp3 --seconds 90 --frames

For every song it prints one line of metrics, flags likely problems, and writes
reports/<song>_timeline.png (beat grid, kick/snare/hat events, band levels, pitch
classes and the spectrum on one time axis) plus reports/report.md and report.json.

Send the report and the timeline image of any song that looks wrong in the visualizer
and the cause is usually visible in them: kicks that miss the beat grid, a half or
double tempo, or events firing far too often.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_dsp import ControlParams, FeatureTrack, analyze, load_audio  # noqa: E402

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".wma", ".aif", ".aiff"}
SENS = 0.3          # onset sensitivity used by the app by default


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def beat_alignment(ft: FeatureTrack, frames: np.ndarray) -> Dict[str, float]:
    """How well a stream of event frames lines up with the beat grid.
    concentration: circular mean resultant length of the beat phase (0 = random, 1 = locked).
    on_beat: share of events within 0.12 beat of a beat or a half beat."""
    if len(frames) < 8:
        return {"concentration": float("nan"), "on_beat": float("nan")}
    phase = ft.beat_pos[frames] % 1.0
    conc = float(np.abs(np.mean(np.exp(2j * np.pi * phase))))
    d = np.minimum(phase % 0.5, 0.5 - phase % 0.5)
    return {"concentration": conc, "on_beat": float(np.mean(d < 0.12))}


def evaluate_track(name: str, ft: FeatureTrack) -> Dict[str, object]:
    fps = ft.clock.fps_float
    dur = ft.duration
    ev = ft.events(SENS)
    kick_f, snare_f, hat_f = (np.nonzero(ev[:, j])[0] for j in range(3))
    period = 60.0 / ft.tempo if ft.tempo > 0 else float("nan")

    kick_al = beat_alignment(ft, kick_f)
    snare_al = beat_alignment(ft, snare_f)

    bands = ft.bands
    silence = float(np.mean(bands.max(axis=0) < 0.04))
    chroma = ft.chroma.astype(np.float32) / 255.0
    cmax = chroma.max(axis=1)
    active = cmax >= 0.5
    dom = chroma.argmax(axis=1)
    stability = float(np.mean(dom[1:][active[1:]] == dom[:-1][active[1:]])) if active.sum() > 2 else float("nan")
    spread = float(np.mean(bands.std(axis=1)))              # how much the loudness actually moves
    kick_rate = len(kick_f) / dur
    snare_rate = len(snare_f) / dur
    hat_rate = len(hat_f) / dur

    warnings: List[str] = []
    if ft.tempo < 70 or ft.tempo > 175:
        warnings.append(f"tempo {ft.tempo:.0f} BPM is at an extreme: likely a half or double tempo error")
    if np.isfinite(kick_al["concentration"]):
        if kick_al["concentration"] < 0.35:
            warnings.append("kicks do not line up with the beat grid (sway will not feel locked to the beat)")
    elif kick_rate < 0.3:
        warnings.append("almost no kick events: music may have no clear low-frequency hits; ripples will be sparse")
    if kick_rate > 4.5:
        warnings.append(f"{kick_rate:.1f} kick events per second is very busy; lower the onset sensitivity")
    if snare_rate > 6.0:
        warnings.append(f"{snare_rate:.1f} mid-band events per second is very busy (dense mix or vocals)")
    if np.isfinite(stability) and stability < 0.55:
        warnings.append("dominant note flickers a lot (dense or noisy mix); Resonance will look busy, try a lower Dynamics value")
    if spread < 0.07:
        warnings.append("loudness barely moves (heavily compressed master): visuals will look flat; raise Dynamics")
    if silence > 0.25:
        warnings.append(f"{silence * 100:.0f}% of the track is near-silent")

    return {
        "name": name, "duration_s": round(dur, 1), "tempo_bpm": round(ft.tempo, 1), "beat_period_s": round(period, 3),
        "kick_per_s": round(kick_rate, 2), "mid_per_s": round(snare_rate, 2), "hat_per_s": round(hat_rate, 2),
        "kick_beat_lock": round(kick_al["concentration"], 2) if np.isfinite(kick_al["concentration"]) else None,
        "kick_on_beat": round(kick_al["on_beat"], 2) if np.isfinite(kick_al["on_beat"]) else None,
        "mid_beat_lock": round(snare_al["concentration"], 2) if np.isfinite(snare_al["concentration"]) else None,
        "note_stability": round(stability, 2) if np.isfinite(stability) else None,
        "loudness_spread": round(spread, 3), "silence_pct": round(silence * 100, 1), "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Timeline image
# --------------------------------------------------------------------------- #
def timeline_png(ft: FeatureTrack, path: Path, seconds: float = 60.0, start: float = 0.0, width: int = 1800) -> None:
    from PIL import Image, ImageDraw

    fps = ft.clock.fps_float
    i0 = int(start * fps)
    i1 = min(ft.n_frames, int((start + seconds) * fps))
    n = max(1, i1 - i0)
    x = lambda i: int((i - i0) / n * (width - 80)) + 70                      # noqa: E731

    rows = [("beats", 26), ("kick", 20), ("mid hit", 20), ("hat", 20), ("sub", 30), ("mid", 30), ("treble", 30), ("notes", 96), ("spectrum", 144)]
    H = sum(h for _, h in rows) + 40
    img = Image.new("RGB", (width, H), (10, 12, 16))
    d = ImageDraw.Draw(img)
    y = 4
    ev = ft.events(SENS)
    cols = [(90, 160, 255), (255, 170, 80), (190, 190, 200)]
    # time axis labels
    for t in range(int(start), int(start + seconds) + 1, 5):
        xx = x(int(t * fps))
        d.text((xx - 6, H - 14), f"{t}s", fill=(120, 126, 136))
    for label, h in rows:
        d.text((4, y + h // 2 - 5), label, fill=(150, 156, 166))
        d.rectangle([70, y, width - 10, y + h - 2], outline=(30, 34, 42))
        if label == "beats":
            b0, b1 = ft.beat_pos[i0], ft.beat_pos[i1 - 1]
            for b in range(int(np.floor(b0)), int(np.ceil(b1)) + 1):
                idx = np.searchsorted(ft.beat_pos, b)
                if i0 <= idx < i1:
                    strong = b % 4 == 0
                    d.line([x(idx), y, x(idx), y + h - 2], fill=(220, 220, 230) if strong else (95, 100, 112))
        elif label in ("kick", "mid hit", "hat"):
            j = ("kick", "mid hit", "hat").index(label)
            for i in np.nonzero(ev[i0:i1, j])[0] + i0:
                s = ev[i, j]
                d.line([x(i), y + h - 2, x(i), y + h - 2 - int((h - 4) * s)], fill=cols[j], width=2)
        elif label in ("sub", "mid", "treble"):
            k = {"sub": 0, "mid": 1, "treble": 2}[label]
            pts = [(70, y + h - 2)] + [(x(i), y + h - 2 - int((h - 4) * min(ft.bands[k, i], 1.0))) for i in range(i0, i1, max(1, n // width))] + [(width - 10, y + h - 2)]
            d.polygon(pts, fill=(40, 70, 110))
        elif label == "notes":
            names = "C C# D D# E F F# G G# A A# B".split()
            step = max(1, n // (width - 80))
            for i in range(i0, i1, step):
                for k in range(12):
                    v = ft.chroma[i, k]
                    if v > 12:
                        yy = y + (11 - k) * 8
                        d.rectangle([x(i), yy, x(i + step), yy + 7], fill=(int(40 + v * 0.8), int(60 + v * 0.7), int(90 + v * 0.6)))
            for k in range(12):
                d.text((52, y + (11 - k) * 8 - 2), names[k], fill=(120, 126, 136))
        elif label == "spectrum":
            step = max(1, n // (width - 80))
            for i in range(i0, i1, step):
                for k in range(48):
                    v = int(ft.spec[i, k])
                    if v > 20:
                        yy = y + (47 - k) * 3
                        d.rectangle([x(i), yy, x(i + step), yy + 2], fill=(int(v * 0.35), int(v * 0.6), v))
        y += h
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def frames_sheet(audio_path: Path, ft: FeatureTrack, path: Path) -> Optional[str]:
    """Six stills of the Resonance scene across the track (needs headless OpenGL)."""
    try:
        from PIL import Image

        from renderer import FreqFluxRenderer, RenderParams

        r = FreqFluxRenderer(RenderParams(width=640, height=360, scene="Resonance"))
    except Exception as e:  # noqa: BLE001
        return f"skipped (no headless OpenGL: {e})"
    cp = ControlParams(onset_sensitivity=SENS)
    marks = [int(ft.n_frames * f) for f in (0.12, 0.28, 0.44, 0.60, 0.76, 0.92)]
    tiles = []
    for m in marks:
        r.reset()
        for i in range(max(0, m - 150), m + 1):
            img = r.render_array(ft.frame(i, cp))
        tiles.append(Image.fromarray(img.copy()))
    r.close()
    sheet = Image.new("RGB", (640 * 3, 360 * 2))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 3) * 640, (k // 3) * 360))
    sheet.save(path)
    return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def collect(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        elif p.suffix.lower() in AUDIO_EXT:
            out.append(p)
        else:
            print(f"skipping {p}: not an audio file or folder")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate FreqFlux analysis on songs")
    ap.add_argument("paths", nargs="+", help="audio files and/or folders")
    ap.add_argument("--out", default="reports", help="output folder (default: reports)")
    ap.add_argument("--seconds", type=float, default=60.0, help="length of the timeline image")
    ap.add_argument("--start", type=float, default=0.0, help="where the timeline image starts (seconds)")
    ap.add_argument("--frames", action="store_true", help="also render a contact sheet of the Resonance scene")
    a = ap.parse_args()

    files = collect(a.paths)
    if not files:
        sys.exit("no audio files found")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        t0 = time.time()
        print(f"analyzing {f.name} ...", end=" ", flush=True)
        try:
            ft = analyze(load_audio(str(f), 44100), 44100, 60)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {e}")
            continue
        r = evaluate_track(f.name, ft)
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in f.stem)[:50]
        timeline_png(ft, out / f"{stem}_timeline.png", seconds=a.seconds, start=a.start)
        if a.frames:
            note = frames_sheet(f, ft, out / f"{stem}_frames.png")
            if note:
                r["frames"] = note
        results.append(r)
        print(f"{ft.tempo:.0f} BPM, kick lock {r['kick_beat_lock']}, {len(r['warnings'])} warning(s) ({time.time() - t0:.0f}s)")

    (out / "report.json").write_text(json.dumps(results, indent=2))
    head = "| song | len | BPM | kick/s | mid/s | hat/s | kick lock | on beat | note stab. | loud spread | silent % |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    rows = "".join(
        f"| {r['name']} | {r['duration_s']}s | {r['tempo_bpm']} | {r['kick_per_s']} | {r['mid_per_s']} | {r['hat_per_s']} | "
        f"{r['kick_beat_lock']} | {r['kick_on_beat']} | {r['note_stability']} | {r['loudness_spread']} | {r['silence_pct']} |\n"
        for r in results
    )
    warn = "".join(f"- **{r['name']}**: {w}\n" for r in results for w in r["warnings"]) or "- none\n"
    key = ("\nkick lock / on beat: 1.0 means kicks sit exactly on the beat grid, below 0.35 means they do not.\n"
           "note stab.: share of frames whose strongest note matches the previous frame; low means flickering notes.\n")
    (out / "report.md").write_text("# FreqFlux analysis report\n\n" + head + rows + key + "\n## Warnings\n\n" + warn)
    print("\n" + head + rows + "\nWarnings:\n" + warn + f"\nwrote {out}/report.md, report.json and one *_timeline.png per song")


if __name__ == "__main__":
    main()
