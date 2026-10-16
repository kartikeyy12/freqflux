"""Renderer (needs headless OpenGL, skipped without it) and the web API."""

import hashlib
import io
import time
import wave

import numpy as np
import pytest

from audio_dsp import ControlParams, analyze

FS = 44100


# ---------------------------------------------------------------- scenes (no GL needed)
def test_scene_registry_and_web_conversion():
    from scenes import DEFAULT_SCENE, PALETTES, SCENES, to_webgl, web_config

    assert DEFAULT_SCENE in SCENES and len(SCENES) >= 5
    for s in SCENES.values():
        assert s.frag.startswith("#version 330\n")
        es = to_webgl(s.frag)
        assert es.startswith("#version 300 es\nprecision highp float;")
        assert "#version 330" not in es
    cfg = web_config()
    assert cfg["audio"]["n_spec"] == 48 and len(cfg["audio"]["spec_edges"]) == 49
    assert {p["name"] for p in cfg["palettes"]} == set(PALETTES)


def test_custom_palette_needs_four_colors():
    from scenes import Palette

    p = Palette.custom("t", ["#000000", "#336699", "#6699cc", "#ffffff"])
    assert len(p.a) == len(p.b) == 4
    with pytest.raises(ValueError):
        Palette.custom("bad", ["#000", "#fff"])


# ---------------------------------------------------------------- renderer (GL)
def test_every_scene_renders_and_is_deterministic(features, need_gl):
    from renderer import FreqFluxRenderer, RenderParams
    from scenes import SCENES

    cp = ControlParams(onset_sensitivity=0.3)
    for name in SCENES:
        digests = []
        for _ in range(2):
            r = FreqFluxRenderer(RenderParams(width=320, height=180, scene=name))
            h = hashlib.md5()
            for i in range(0, 90):
                buf = r.render(features.frame(i, cp))
                assert len(buf) == 320 * 180 * 3
                h.update(buf)
            r.close()
            digests.append(h.hexdigest())
        assert digests[0] == digests[1], f"{name} is not deterministic"


def test_frames_are_not_blank_and_change_with_the_music(features, need_gl):
    from renderer import FreqFluxRenderer, RenderParams

    cp = ControlParams(onset_sensitivity=0.3)
    r = FreqFluxRenderer(RenderParams(width=320, height=180, scene="Resonance"))
    a = b = None
    for i in range(0, 700):
        img = r.render_array(features.frame(i, cp))
        if i == 300:
            a = img.copy()
        if i == 690:
            b = img.copy()
    r.close()
    assert a.std() > 3 and b.std() > 3                      # something is drawn
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 1.0   # and it moves


def test_odd_sizes_are_rejected(need_gl):
    from renderer import FreqFluxRenderer, RenderParams

    with pytest.raises(ValueError):
        FreqFluxRenderer(RenderParams(width=641, height=360))


# ---------------------------------------------------------------- web API
@pytest.fixture()
def client(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import app as appmod

    monkeypatch.setattr(appmod, "UPLOAD_DIR", tmp_path)
    return TestClient(appmod.app)


def _wav_bytes(y):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def test_upload_analyze_and_fetch_features(client, track):
    y, _ = track
    r = client.post("/api/upload", files={"file": ("loop.wav", _wav_bytes(y[: FS * 8]), "audio/wav")})
    assert r.status_code == 200
    sid = r.json()["id"]
    info = {}
    for _ in range(120):
        info = client.get(f"/api/session/{sid}").json()
        if info["status"] != "analyzing":
            break
        time.sleep(0.25)
    assert info["status"] == "ready", info
    n = info["nFrames"]
    assert n == 480 and info["fps"] == 60 and info["channels"] == 6
    assert len(client.get(f"/api/features/{sid}").content) == n * 6 * 4
    assert len(client.get(f"/api/spectrum/{sid}").content) == n * 60
    assert len(client.get(f"/api/onsets/{sid}?s=0.3").content) == n * 4 * 4
    # seeking needs HTTP Range support
    part = client.get(f"/api/audio/{sid}", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and len(part.content) == 100


def test_unknown_session_and_bad_export_requests(client):
    assert client.get("/api/session/nope").status_code == 404
    assert client.post("/api/export", json={"id": "nope"}).status_code == 404
    assert client.get("/api/config").status_code == 200
    assert client.get("/").status_code == 200
