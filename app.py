"""FreqFlux web app.

    python app.py                # serves on 0.0.0.0:7860
    open http://localhost:7860   # use localhost (not the WSL IP): browsers only
                                 # allow microphone access on localhost or https

How it works
------------
* You upload a song. The server decodes and analyzes it with audio_dsp (exact
  60 FPS frame clock) in a background thread.
* The browser plays the original file in an <audio> element and renders the
  visuals live with WebGL2, reading the per-frame features at audio.currentTime.
  The shaders are served from scenes.py, the same ones the MP4 exporter uses.
* "Export video" is optional: it renders the same scene headlessly to an MP4.
"""

from __future__ import annotations

import argparse
import mimetypes
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from audio_dsp import ControlParams, FeatureTrack, _beat_positions, analyze, load_audio
from renderer import RenderParams
from scenes import DEFAULT_PALETTE, DEFAULT_SCENE, PALETTES, SCENES, Palette, web_config
from video_exporter import ANALYSIS_FS, ExportCancelled, ExportError, export_video

APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"
UPLOAD_DIR = APP_DIR / "uploads"
RENDER_DIR = APP_DIR / "renders"
UPLOAD_DIR.mkdir(exist_ok=True)
RENDER_DIR.mkdir(exist_ok=True)

LIVE_FPS = 60
CHANNELS = 6     # per frame: sub_bass, mid, treble, flux, brightness, beat position


# --------------------------------------------------------------------------- #
# Sessions (one per uploaded track)
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    id: str
    name: str
    path: Path
    status: str = "analyzing"      # analyzing | ready | error
    progress: float = 0.0
    error: str = ""
    duration: float = 0.0
    n_frames: int = 0
    _ft: Dict[int, FeatureTrack] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def features(self, fps: int, progress=None) -> FeatureTrack:
        with self._lock:
            ft = self._ft.get(fps)
            if ft is None:
                y = load_audio(str(self.path), ANALYSIS_FS)
                ft = analyze(y, ANALYSIS_FS, Fraction(fps), progress=progress)
                self._ft[fps] = ft
            return ft


SESSIONS: Dict[str, Session] = {}


def _analyze_session(s: Session) -> None:
    try:
        s.progress = 0.03
        y = load_audio(str(s.path), ANALYSIS_FS)
        s.progress = 0.12
        ft = analyze(y, ANALYSIS_FS, Fraction(LIVE_FPS),
                     progress=lambda f: setattr(s, "progress", 0.12 + 0.88 * f))
        s._ft[LIVE_FPS] = ft
        s.duration = ft.duration
        s.n_frames = ft.n_frames
        s.progress = 1.0
        s.status = "ready"
    except Exception as e:  # noqa: BLE001
        s.status = "error"
        s.error = f"Could not read this audio file: {e}"


def _session(sid: str) -> Session:
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "unknown session (the server may have restarted; upload again)")
    return s


# --------------------------------------------------------------------------- #
# Export jobs
# --------------------------------------------------------------------------- #
class ExportReq(BaseModel):
    id: str
    scene: str = DEFAULT_SCENE
    palette: str = DEFAULT_PALETTE
    custom: Optional[List[str]] = None
    width: int = 1920
    height: int = 1080
    fps: int = 60
    motion: float = 1.0
    intensity: float = 1.0
    contrast: float = 1.4
    bass: float = 1.0
    mid: float = 1.0
    treble: float = 1.0
    onset: float = 0.3
    crf: int = 18
    max_seconds: Optional[float] = None


@dataclass
class Job:
    id: str
    session: str
    status: str = "running"        # running | done | error | cancelled
    progress: float = 0.0
    message: str = "Starting"
    file: Optional[Path] = None
    error: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)


JOBS: Dict[str, Job] = {}
_EXPORT_LOCK = threading.Lock()


def _run_export(job: Job, req: ExportReq, s: Session) -> None:
    try:
        palette = Palette.custom("Custom", req.custom) if req.custom else PALETTES[req.palette]
        rp = RenderParams(
            width=req.width, height=req.height, fps=Fraction(req.fps), scene=req.scene, palette=palette,
            contrast=req.contrast, intensity=req.intensity, motion=req.motion, gain=(req.bass, req.mid, req.treble),
        )
        cp = ControlParams(req.bass, req.mid, req.treble, req.onset)
        job.message = "Analyzing audio"
        ft = s.features(req.fps, progress=lambda f: setattr(job, "progress", 0.08 * f))
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(s.name).stem)[:40] or "track"
        slug = re.sub(r"[^a-z0-9]+", "_", req.scene.lower()).strip("_")
        out = RENDER_DIR / f"{stem}_{slug}_{req.height}p{req.fps}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

        def prog(frac: float, msg: str) -> None:
            job.progress, job.message = frac, msg

        export_video(str(s.path), str(out), render=rp, control=cp, crf=req.crf, features=ft,
                     max_seconds=req.max_seconds, progress=prog, cancel=job.cancel)
        job.file, job.status, job.progress, job.message = out, "done", 1.0, "Done"
    except ExportCancelled:
        job.status, job.message = "cancelled", "Cancelled"
    except Exception as e:  # noqa: BLE001
        job.status, job.error, job.message = "error", str(e), "Failed"
    finally:
        _EXPORT_LOCK.release()


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="FreqFlux")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/config")
def config() -> JSONResponse:
    return JSONResponse(web_config())


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    name = file.filename or "track"
    suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(name).suffix.lower())[:8] or ".audio"
    sid = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{sid}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f, length=1024 * 1024)
    s = Session(id=sid, name=name, path=dest)
    SESSIONS[sid] = s
    threading.Thread(target=_analyze_session, args=(s,), daemon=True).start()
    return {"id": sid}


@app.get("/api/session/{sid}")
def session_info(sid: str) -> dict:
    s = _session(sid)
    return {
        "id": s.id, "name": s.name, "status": s.status, "progress": s.progress, "error": s.error,
        "duration": s.duration, "fps": LIVE_FPS, "nFrames": s.n_frames, "channels": CHANNELS,
        "tempo": (s._ft[LIVE_FPS].tempo if LIVE_FPS in s._ft else None),
    }


@app.get("/api/audio/{sid}")
def audio(sid: str) -> FileResponse:
    s = _session(sid)
    mime = mimetypes.guess_type(str(s.path))[0] or "audio/mpeg"
    return FileResponse(s.path, media_type=mime)       # Starlette serves HTTP Range for seeking


def _ready(sid: str) -> FeatureTrack:
    s = _session(sid)
    if s.status != "ready":
        raise HTTPException(409, "analysis not finished")
    return s._ft[LIVE_FPS]


@app.get("/api/features/{sid}")
def features(sid: str) -> Response:
    ft = _ready(sid)
    arr = np.stack([ft.bands[0], ft.bands[1], ft.bands[2], ft.flux, ft.brightness, ft.beat_pos], axis=1).astype("<f4")
    return Response(arr.tobytes(), media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@app.get("/api/spectrum/{sid}")
def spectrum(sid: str) -> Response:
    """uint8, shape (n_frames, 60): 48 log-frequency bands then 12 pitch classes (C..B)."""
    ft = _ready(sid)
    arr = np.concatenate([ft.spec, ft.chroma], axis=1).astype(np.uint8)
    return Response(arr.tobytes(), media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@app.get("/api/onsets/{sid}")
def onsets(sid: str, s: float = Query(0.3, ge=0.0, le=1.0)) -> Response:
    ft = _ready(sid)
    ev = ft.events(s)       # (n_frames, 4): kick, snare, hat, any
    return Response(ev.astype("<f4").tobytes(), media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/export")
def start_export(req: ExportReq) -> dict:
    s = _session(req.id)
    if s.status != "ready":
        raise HTTPException(409, "analysis not finished")
    if req.scene not in SCENES:
        raise HTTPException(400, "unknown scene")
    if not req.custom and req.palette not in PALETTES:
        raise HTTPException(400, "unknown palette")
    if req.fps not in (30, 60) or req.width % 2 or req.height % 2 or not (320 <= req.width <= 3840):
        raise HTTPException(400, "unsupported resolution or frame rate")
    try:
        if req.custom:
            Palette.custom("Custom", req.custom)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not _EXPORT_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another export is already running")
    job = Job(id=uuid.uuid4().hex[:10], session=s.id)
    JOBS[job.id] = job
    threading.Thread(target=_run_export, args=(job, req, s), daemon=True).start()
    return {"job": job.id}


def _job(jid: str) -> Job:
    j = JOBS.get(jid)
    if j is None:
        raise HTTPException(404, "unknown job")
    return j


@app.get("/api/export/{jid}")
def export_status(jid: str) -> dict:
    j = _job(jid)
    return {"status": j.status, "progress": j.progress, "message": j.message, "error": j.error,
            "file": j.file.name if j.file else None}


@app.post("/api/export/{jid}/cancel")
def export_cancel(jid: str) -> dict:
    _job(jid).cancel.set()
    return {"ok": True}


@app.get("/api/export/{jid}/file")
def export_file(jid: str) -> FileResponse:
    j = _job(jid)
    if j.status != "done" or not j.file:
        raise HTTPException(409, "export not finished")
    return FileResponse(j.file, media_type="video/mp4", filename=j.file.name)


def _warmup() -> None:
    """librosa compiles its beat tracker on first use (about 15 s once). Do it at
    startup so the first upload is not slow."""
    try:
        _beat_positions(np.random.default_rng(0).random(1200).astype(np.float32), 60.0)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="FreqFlux web app")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    threading.Thread(target=_warmup, daemon=True).start()
    print(f"\n  FreqFlux is running.  Open  http://localhost:{a.port}  in your browser.\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
