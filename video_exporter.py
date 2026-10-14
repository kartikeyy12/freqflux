"""FreqFlux video exporter.

Renders every frame headlessly and streams raw RGB24 into an FFmpeg subprocess
that encodes H.264 and muxes the ORIGINAL audio file.

Sync guarantees
---------------
* Frame i is analyzed at sample floor(i * fs / fps) (audio_dsp.FrameClock),
  exact integer math, so the analysis has no drift over any track length.
* The raw video is declared to FFmpeg as constant frame rate num/den, and every
  frame is written exactly once, so video timestamps are i * den / num exactly.
* The muxed audio comes straight from the source file. It is padded with
  silence (apad) or trimmed to the exact video duration n_frames / fps, so both
  streams end within one audio sample of each other.

Rendering and encoding overlap: the render loop hands frames to a writer thread
through a small queue, so the GPU readback of frame i+1 happens while FFmpeg
consumes frame i.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Callable, Deque, Dict, List, Optional

from audio_dsp import ControlParams, FeatureTrack, analyze, load_audio
from renderer import FreqFluxRenderer, RenderParams
from scenes import DEFAULT_PALETTE, DEFAULT_SCENE, PALETTES, SCENES

ANALYSIS_FS = 44100
ANALYSIS_SHARE = 0.08      # fraction of the progress bar used by audio analysis

# progress(fraction 0..1, message)
ProgressFn = Callable[[float, str], None]


class ExportError(RuntimeError):
    pass


class ExportCancelled(ExportError):
    pass


@dataclass
class ExportResult:
    path: str
    frames: int
    video_seconds: float
    wall_seconds: float

    @property
    def render_fps(self) -> float:
        return self.frames / self.wall_seconds if self.wall_seconds > 0 else 0.0


# --------------------------------------------------------------------------- #
# FFmpeg helpers
# --------------------------------------------------------------------------- #
def _fps_fraction(fps) -> Fraction:
    if isinstance(fps, float):
        return Fraction(fps).limit_denominator(1001)
    return Fraction(fps)


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise ExportError("ffmpeg not found on PATH. Install with: sudo apt-get install -y ffmpeg")
    return path


def check_encoder(ffmpeg: str, encoder: str) -> None:
    out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if not any(line.split()[1:2] == [encoder] for line in out.splitlines() if line.startswith(" V")):
        raise ExportError(f"this ffmpeg build has no {encoder!r} encoder")


def build_ffmpeg_cmd(
    ffmpeg: str,
    audio_path: str,
    out_path: str,
    width: int,
    height: int,
    fps: Fraction,
    video_seconds: Fraction,
    encoder: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    audio_bitrate: str = "320k",
) -> List[str]:
    rate = f"{fps.numerator}/{fps.denominator}"
    dur = f"{float(video_seconds):.6f}"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        # input 0: raw frames from stdin
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-framerate", rate, "-i", "pipe:0",
        # input 1: original audio
        "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        # full-range RGB -> limited-range BT.709 YUV 4:2:0
        "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
    ]
    if encoder == "libx264":
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high",
                "-x264-params", "aq-mode=3:deblock=-1,-1"]   # keep soft dark gradients free of banding
    elif encoder == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(crf + 1), "-b:v", "0",
                "-profile:v", "high"]
    else:
        raise ExportError(f"unsupported encoder {encoder!r} (use libx264 or h264_nvenc)")
    cmd += [
        "-r", rate, "-g", str(max(1, int(round(float(fps) * 2)))),
        # exact length: pad short audio with silence, then cut both at the video duration
        "-af", f"apad=whole_dur={dur}",
        "-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000",
        "-t", dur,
        "-movflags", "+faststart",
        "-f", "mp4", out_path,
    ]
    return cmd


def probe(path: str) -> Dict[str, float]:
    """Inspect an exported file: video packet count and per-stream durations."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ExportError("ffprobe not found on PATH")
    raw = subprocess.run(
        [ffprobe, "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_type,codec_name,width,height,r_frame_rate,duration,nb_read_packets",
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    info: Dict[str, float] = {}
    for s in json.loads(raw)["streams"]:
        if s["codec_type"] == "video":
            info["video_frames"] = int(s["nb_read_packets"])
            info["video_seconds"] = float(s["duration"])
            info["width"], info["height"] = int(s["width"]), int(s["height"])
            info["fps"] = float(Fraction(s["r_frame_rate"]))
        elif s["codec_type"] == "audio":
            info["audio_seconds"] = float(s["duration"])
    if "video_seconds" in info and "audio_seconds" in info:
        info["av_offset_ms"] = (info["audio_seconds"] - info["video_seconds"]) * 1000.0
    return info


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_video(
    audio_path: str,
    out_path: str,
    render: Optional[RenderParams] = None,
    control: Optional[ControlParams] = None,
    encoder: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    audio_bitrate: str = "320k",
    max_seconds: Optional[float] = None,
    progress: Optional[ProgressFn] = None,
    cancel: Optional[threading.Event] = None,
    features: Optional[FeatureTrack] = None,
) -> ExportResult:
    """Render audio_path into an MP4 at out_path.

    render       scene, palette, size, fps and look parameters (default 1080p60)
    control      gains and onset sensitivity (default: onset sensitivity 0.3)
    max_seconds  render only the first N seconds (for quick previews)
    features     a precomputed FeatureTrack (skips analysis; fps must match)
    cancel       set this threading.Event to abort; raises ExportCancelled
    """
    if not os.path.isfile(audio_path):
        raise ExportError(f"audio file not found: {audio_path}")
    params = replace(render) if render else RenderParams()
    fps = _fps_fraction(params.fps)
    params.fps = fps
    control = control or ControlParams(onset_sensitivity=0.3)
    ffmpeg = find_ffmpeg()
    check_encoder(ffmpeg, encoder)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(min(max(frac, 0.0), 1.0), msg)

    def check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise ExportCancelled("export cancelled")

    # 1. analysis
    if features is None:
        report(0.0, "Decoding audio")
        y = load_audio(audio_path, ANALYSIS_FS)
        check_cancel()
        ft = analyze(
            y, ANALYSIS_FS, fps,
            progress=lambda f: report(ANALYSIS_SHARE * f, "Analyzing audio"),
        )
    else:
        ft = features
        if ft.clock.fps != fps or ft.clock.fs != ANALYSIS_FS:
            raise ExportError("precomputed features were made with a different fps or sample rate")

    n_frames = ft.n_frames
    if max_seconds is not None:
        n_frames = min(n_frames, max(1, int(math.ceil(max_seconds * float(fps)))))
    video_seconds = Fraction(n_frames) / fps

    # 2. start FFmpeg
    part = out_path + ".part"
    cmd = build_ffmpeg_cmd(
        ffmpeg, audio_path, part, params.width, params.height, fps, video_seconds,
        encoder=encoder, crf=crf, preset=preset, audio_bitrate=audio_bitrate,
    )
    renderer: Optional[FreqFluxRenderer] = None
    proc: Optional[subprocess.Popen] = None
    t0 = time.time()
    try:
        renderer = FreqFluxRenderer(params)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert proc.stdin is not None and proc.stderr is not None

        err_tail: Deque[str] = deque(maxlen=40)

        def drain_stderr() -> None:
            for line in proc.stderr:  # type: ignore[union-attr]
                err_tail.append(line.decode("utf-8", "replace").rstrip())

        threading.Thread(target=drain_stderr, daemon=True).start()

        q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=6)
        write_error: List[BaseException] = []

        def writer() -> None:
            try:
                while True:
                    item = q.get()
                    if item is None:
                        break
                    proc.stdin.write(item)  # type: ignore[union-attr]
                proc.stdin.close()  # type: ignore[union-attr]
            except BaseException as e:  # noqa: BLE001
                write_error.append(e)

        wt = threading.Thread(target=writer, daemon=True)
        wt.start()

        def put(item: Optional[bytes]) -> None:
            while True:
                if write_error or proc.poll() is not None:
                    raise ExportError("ffmpeg stopped early:\n" + "\n".join(err_tail))
                try:
                    q.put(item, timeout=0.25)
                    return
                except queue.Full:
                    check_cancel()

        # 3. render loop
        for i in range(n_frames):
            check_cancel()
            put(renderer.render(ft.frame(i, control)))
            if i % 15 == 0 or i == n_frames - 1:
                el = time.time() - t0
                rate = (i + 1) / el if el > 0 else 0.0
                eta = (n_frames - i - 1) / rate if rate > 0 else 0.0
                report(
                    ANALYSIS_SHARE + (1 - ANALYSIS_SHARE) * (i + 1) / n_frames,
                    f"Rendering frame {i + 1}/{n_frames}  ({rate:.1f} fps, ETA {int(eta)}s)",
                )

        put(None)
        wt.join()
        if write_error:
            raise ExportError("ffmpeg pipe failed:\n" + "\n".join(err_tail))
        rc = proc.wait()
        if rc != 0:
            raise ExportError(f"ffmpeg exited with code {rc}:\n" + "\n".join(err_tail))
        os.replace(part, out_path)
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        if os.path.exists(part):
            os.remove(part)
        raise
    finally:
        if renderer is not None:
            renderer.close()

    report(1.0, "Done")
    return ExportResult(out_path, n_frames, float(video_seconds), time.time() - t0)


def export_preview(
    audio_path: str,
    out_path: str,
    render: Optional[RenderParams] = None,
    control: Optional[ControlParams] = None,
    seconds: float = 12.0,
    **kw,
) -> ExportResult:
    """Fast low-res clip for the web UI preview player: 854x480, 30 FPS,
    high CRF, veryfast preset, first `seconds` of the track."""
    p = replace(render) if render else RenderParams()
    p.width, p.height, p.fps = 854, 480, 30
    kw.setdefault("crf", 27)
    kw.setdefault("preset", "veryfast")
    kw.setdefault("audio_bitrate", "128k")
    return export_video(audio_path, out_path, render=p, control=control, max_seconds=seconds, **kw)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="FreqFlux video exporter")
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default=None, help="output .mp4 (default renders/<audio name>.mp4)")
    ap.add_argument("--scene", default=DEFAULT_SCENE, choices=list(SCENES))
    ap.add_argument("--palette", default=DEFAULT_PALETTE, choices=list(PALETTES))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", default="60", help="e.g. 60, 30, or 60000/1001")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--encoder", default="libx264", choices=["libx264", "h264_nvenc"])
    ap.add_argument("--max-seconds", type=float, default=None, help="render only the first N seconds")
    ap.add_argument("--contrast", type=float, default=1.4)
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--motion", type=float, default=1.0)
    ap.add_argument("--bass-gain", type=float, default=1.0)
    ap.add_argument("--mid-gain", type=float, default=1.0)
    ap.add_argument("--treble-gain", type=float, default=1.0)
    ap.add_argument("--onset-sensitivity", type=float, default=0.3)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    out = a.out or os.path.join("renders", os.path.splitext(os.path.basename(a.audio))[0] + ".mp4")
    rp = RenderParams(
        width=a.width, height=a.height, fps=Fraction(a.fps), scene=a.scene, palette=a.palette,
        contrast=a.contrast, intensity=a.intensity, motion=a.motion,
    )
    cp = ControlParams(a.bass_gain, a.mid_gain, a.treble_gain, a.onset_sensitivity)

    last = [0.0]

    def prog(frac: float, msg: str) -> None:
        if frac - last[0] >= 0.005 or frac >= 1.0:
            last[0] = frac
            print(f"\r[{frac * 100:5.1f}%] {msg:<64}", end="", flush=True)

    res = export_video(
        a.audio, out, render=rp, control=cp, encoder=a.encoder, crf=a.crf, preset=a.preset,
        max_seconds=a.max_seconds, progress=prog,
    )
    print(f"\nwrote {res.path}: {res.frames} frames, {res.video_seconds:.2f}s video, "
          f"{res.wall_seconds:.1f}s wall ({res.render_fps:.1f} fps)")
    if not a.no_verify:
        info = probe(res.path)
        print(f"verify: {info['video_frames']} video frames @ {info['fps']:.3f} fps, "
              f"video {info['video_seconds']:.4f}s, audio {info['audio_seconds']:.4f}s, "
              f"A/V end offset {info['av_offset_ms']:+.2f} ms")


if __name__ == "__main__":
    main()
