import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.synth import make_track  # noqa: E402

FS = 44100


@pytest.fixture(scope="session")
def track():
    """24 s, 120 BPM loop with kicks, snares, hats and a known melody."""
    return make_track(FS, 24.0, bpm=120.0)


@pytest.fixture(scope="session")
def features(track):
    from audio_dsp import analyze

    y, _ = track
    return analyze(y, FS, 60)


@pytest.fixture(scope="session")
def gl_available():
    """Headless OpenGL is optional: GL tests skip when no context can be created."""
    try:
        from renderer import _create_context

        ctx = _create_context()
        ctx.release()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture()
def need_gl(gl_available):
    if not gl_available:
        pytest.skip("no headless OpenGL context available")
