#!/usr/bin/env bash
# FreqFlux WSL2 (Ubuntu 22.04 / 24.04) bootstrap. Run: bash setup_wsl.sh
set -euo pipefail

PROJECT_DIR="${HOME}/projects/freqflux"

echo "[1/5] System packages"
sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-dev python3-pip build-essential pkg-config \
  ffmpeg libsndfile1 \
  libportaudio2 portaudio19-dev libasound2-dev libasound2-plugins libpulse0 pulseaudio-utils \
  xvfb freeglut3-dev libglu1-mesa libgl1 libgl1-mesa-dri libegl1 libosmesa6 mesa-utils \
  libcairo2-dev

echo "[2/5] Project layout at ${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}"/{uploads,renders,cache,tests}
cd "${PROJECT_DIR}"
touch app.py renderer.py video_exporter.py   # audio_dsp.py and requirements.txt are copied in separately

echo "[3/5] Virtual environment"
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
if [[ -f requirements.txt ]]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found in ${PROJECT_DIR}; copy it there and run: pip install -r requirements.txt"
fi

echo "[4/5] Verify FFmpeg"
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E "libx264|aac" || { echo "FFmpeg is missing libx264 or aac"; exit 1; }

echo "[5/5] Verify headless OpenGL (EGL, software fallback)"
python - <<'PY'
import os
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
import moderngl
try:
    ctx = moderngl.create_standalone_context(backend="egl")
except Exception:
    ctx = moderngl.create_standalone_context()
print("OpenGL renderer:", ctx.info["GL_RENDERER"])
PY

echo
echo "Done. Next:  cd ${PROJECT_DIR} && source .venv/bin/activate && python audio_dsp.py --selftest"
echo "If EGL fails, wrap commands with:  xvfb-run -a <command>"
