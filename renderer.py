"""FreqFlux headless renderer (ModernGL).

One full-screen shader pass per frame. The shaders, palettes and motion logic
come from scenes.py, the same source the browser player uses, so exported
videos match the live view.

State advances with a fixed dt and resets at frame index 0, so rendering the
same ControlFrame sequence twice gives identical output.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Optional, Tuple, Union

import moderngl
import numpy as np

from audio_dsp import N_HIST, N_SPEC, ControlFrame
from scenes import DEFAULT_PALETTE, DEFAULT_SCENE, PALETTES, SCENES, VERT, Palette, SceneState


@dataclass
class RenderParams:
    width: int = 1920
    height: int = 1080
    fps: Union[int, float, str, Fraction] = 60
    scene: str = DEFAULT_SCENE
    palette: Union[str, Palette] = DEFAULT_PALETTE
    contrast: float = 1.4        # power curve on band values; higher = more dynamic range
    intensity: float = 1.0       # overall brightness
    motion: float = 1.0          # speed of the ambient drift
    gain: Tuple[float, float, float] = (1.0, 1.0, 1.0)   # bass, mid, treble gain applied to the spectrum

    def fps_float(self) -> float:
        f = Fraction(self.fps) if not isinstance(self.fps, float) else Fraction(self.fps).limit_denominator(1001)
        return float(f)


def shape(x: float, contrast: float) -> float:
    return float(max(x, 0.0) ** contrast)


def _create_context() -> moderngl.Context:
    backend = os.environ.get("FREQFLUX_GL_BACKEND")
    attempts = [{"backend": backend}] if backend else [{"backend": "egl"}, {}]
    last: Optional[Exception] = None
    for kw in attempts:
        try:
            return moderngl.create_standalone_context(require=330, **kw)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(
        "Could not create a headless OpenGL 3.3 context. Try LIBGL_ALWAYS_SOFTWARE=1 or run under "
        "xvfb-run -a. Last error: %s" % last
    )


class FreqFluxRenderer:
    def __init__(self, params: Optional[RenderParams] = None, ctx: Optional[moderngl.Context] = None):
        self.params = params or RenderParams()
        p = self.params
        if p.width < 64 or p.height < 64 or p.width % 2 or p.height % 2:
            raise ValueError("width/height must be even and >= 64 (yuv420p needs even sizes)")
        self.ctx = ctx or _create_context()
        self._owns_ctx = ctx is None
        self.dt_default = 1.0 / p.fps_float()
        self.tex = self.ctx.texture((p.width, p.height), 3, dtype="f1")
        self.fbo = self.ctx.framebuffer([self.tex])
        self.spec_tex = self.ctx.texture((N_SPEC, N_HIST), 1, dtype="f1")
        self.spec_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.spec_tex.repeat_x = self.spec_tex.repeat_y = False
        self._zero_spec = np.zeros((N_HIST, N_SPEC), dtype=np.uint8)
        self._programs: Dict[str, tuple] = {}
        self.state = SceneState()
        self.set_scene(p.scene)
        self.set_palette(p.palette)

    def set_scene(self, name: str) -> None:
        if name not in SCENES:
            raise KeyError(f"unknown scene {name!r}; choose from {list(SCENES)}")
        self.params.scene = name
        if name not in self._programs:
            prog = self.ctx.program(vertex_shader=VERT, fragment_shader=SCENES[name].frag)
            self._programs[name] = (prog, self.ctx.vertex_array(prog, []))
        self.prog, self.vao = self._programs[name]

    def set_palette(self, palette: Union[str, Palette]) -> None:
        if isinstance(palette, Palette):
            self.palette = palette
        else:
            if palette not in PALETTES:
                raise KeyError(f"unknown palette {palette!r}; choose from {list(PALETTES)}")
            self.palette = PALETTES[palette]
        self.params.palette = self.palette

    def reset(self) -> None:
        self.state.reset()

    def _set(self, **uniforms) -> None:
        for k, v in uniforms.items():
            if k in self.prog:
                self.prog[k].value = v

    def _write_array(self, name: str, arr: np.ndarray) -> None:
        if name in self.prog:
            self.prog[name].write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())

    def render(self, c: ControlFrame, dt: Optional[float] = None) -> bytes:
        """Render one frame; returns top-down RGB24 bytes (width*height*3)."""
        if c.index == 0:
            self.reset()
        p, st = self.params, self.state
        dt = self.dt_default if dt is None else float(dt)
        st.advance(c, dt, p.motion, p.contrast)
        col = st.colors(self.palette)
        spec = c.spec_hist if c.spec_hist is not None else self._zero_spec
        self.spec_tex.write(np.ascontiguousarray(spec, dtype=np.uint8).tobytes())
        self.spec_tex.use(location=0)
        self.fbo.use()
        self.ctx.viewport = (0, 0, p.width, p.height)
        self._set(
            u_spec=0, u_flip=-1.0,
            u_res=(float(p.width), float(p.height)),
            u_time=float(st.T), u_flow=float(st.flow), u_beat=float(st.beat),
            u_bass=float(min(st.bass, 1.6)), u_mid=float(min(st.mid, 1.6)), u_treble=float(min(st.treble, 1.6)),
            u_kick=float(st.kick), u_snare=float(st.snare), u_hat=float(st.hat),
            u_intensity=float(p.intensity), u_contrast=float(p.contrast), u_frame=float(c.index % 65536),
            u_gain=tuple(float(g) for g in p.gain),
            u_c0=tuple(float(x) for x in col[0]), u_c1=tuple(float(x) for x in col[1]),
            u_c2=tuple(float(x) for x in col[2]), u_c3=tuple(float(x) for x in col[3]),
        )
        self._write_array("u_rip", st.ripple_array())
        self._write_array("u_chroma", st.chroma.astype(np.float32))
        self.vao.render(moderngl.TRIANGLES, vertices=3)
        return self.fbo.read(components=3, alignment=1)

    def render_array(self, c: ControlFrame, dt: Optional[float] = None) -> np.ndarray:
        buf = self.render(c, dt)
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.params.height, self.params.width, 3)

    def close(self) -> None:
        if self._owns_ctx:
            self.ctx.release()


# --------------------------------------------------------------------------- #
# CLI: render still frames for look-dev
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="FreqFlux renderer look-dev")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--scene", default="all")
    ap.add_argument("--palette", default=DEFAULT_PALETTE)
    ap.add_argument("--frame", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=180)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default="renders")
    a = ap.parse_args()

    from PIL import Image

    from audio_dsp import ControlParams, analyze, load_audio

    ft = analyze(load_audio(a.audio, 44100), 44100, 60)
    cp = ControlParams(onset_sensitivity=0.3)
    start = max(0, a.frame - a.warmup)
    seq = [ft.frame(i, cp) for i in range(start, a.frame + 1)]
    os.makedirs(a.out, exist_ok=True)
    names = list(SCENES) if a.scene == "all" else [a.scene]
    r = FreqFluxRenderer(RenderParams(width=a.width, height=a.height, scene=names[0], palette=a.palette))
    for name in names:
        r.set_scene(name)
        r.reset()
        for cf in seq:
            img = r.render_array(cf)
        path = os.path.join(a.out, f"{name.lower().replace(' ', '_')}_f{a.frame}.png")
        Image.fromarray(img).save(path)
        print("wrote", path)
    r.close()


if __name__ == "__main__":
    main()
