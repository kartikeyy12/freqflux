"""FreqFlux scenes: the single source of truth for how the visuals look.

The GLSL below is compiled by BOTH the headless Python renderer (MP4 export,
GLSL 330 core) and the browser (WebGL2, GLSL ES 300). app.py serves it to the
page via to_webgl(), so the live player and the exported video match.

Everything that moves is tied to something in the music:

    kick (low-band hit)   -> a ring launches and travels outward; the pattern gets a push
    mid-band hit / note   -> a thinner, faster ring; a smaller push
    hat (high-band hit)   -> the motes twinkle
    pitch (12 notes)      -> the note ring around the orb lights the sounding pitch classes
    spectrum              -> in Aurora Veil and Soft Waves, horizontal position IS frequency:
                             bass on the left, treble on the right
    beat position         -> sway and scroll locked to the tracked tempo
    onsets (any)          -> a slow drift between two tones of the palette

No effect is a global scale pulse. The per-frame motion logic (SceneState) is
small and mirrored in web/index.html; its constants live in STYLE.
"""

from __future__ import annotations

import colorsys
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from audio_dsp import CHROMA_FMAX, CHROMA_FMIN, N_HIST, N_SPEC, SPEC_EDGES

# --------------------------------------------------------------------------- #
# Shared constants (also served to the browser)
# --------------------------------------------------------------------------- #
STYLE: Dict[str, float] = {
    "band_attack_s": 0.03,        # loudness smoothing: short, so hits stay visible
    "band_release_s": 0.20,
    "kick_attack_s": 0.012, "kick_release_s": 0.24,
    "snare_attack_s": 0.012, "snare_release_s": 0.18,
    "hat_attack_s": 0.008, "hat_release_s": 0.10,
    "chroma_attack_s": 0.04, "chroma_release_s": 0.28,
    "brightness_tau_s": 1.2,      # hue smoothing for the Spectrum palette
    "flow_base": 0.02,            # ambient drift of the flowing patterns
    "flow_level": 0.10,           # extra drift while the music is loud
    "flow_kick": 0.55,            # velocity added by a kick
    "flow_snare": 0.22,           # velocity added by a mid-band hit
    "flow_decay_s": 0.55,         # how quickly a push glides to a stop
    "ripple_life_s": 2.2,
    "ripple_max": 6,
    "shift_min_onset": 0.35,      # onset strength that may trigger a tone drift
    "shift_min_gap_s": 1.0,
    "shift_tau_s": 1.6,
    "auto_hue_base": 0.50,        # Spectrum palette sweeps teal -> violet only
    "auto_hue_span": 0.32,
}


# --------------------------------------------------------------------------- #
# Palettes: two tones (A and B) of four roles: background, body, secondary, highlight
# --------------------------------------------------------------------------- #
def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        raise ValueError(f"bad hex color: {h!r}")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    return "#%02x%02x%02x" % tuple(int(round(min(max(c, 0.0), 1.0) * 255)) for c in rgb)


def _lin(rgb: Sequence[float]) -> np.ndarray:
    return np.power(np.asarray(rgb, dtype=np.float64), 2.2)


@dataclass(frozen=True)
class Palette:
    name: str
    a: Tuple[str, ...] = ()
    b: Tuple[str, ...] = ()
    auto: bool = False

    @staticmethod
    def custom(name: str, hexes: Sequence[str]) -> "Palette":
        """Four user colors (background, body, secondary, highlight). Tone B is
        the same colors rotated 25 degrees in hue."""
        if len(hexes) != 4:
            raise ValueError("a custom palette needs exactly 4 colors")
        a = [_hex_to_rgb(h) for h in hexes]
        b = []
        for r, g, bl in a:
            h, s, v = colorsys.rgb_to_hsv(r, g, bl)
            b.append(_rgb_to_hex(colorsys.hsv_to_rgb((h + 0.07) % 1.0, s, v)))
        return Palette(name, tuple(hexes), tuple(b))

    def resolved(self, brightness: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Linear-light role colors for tone A and tone B."""
        if self.auto:
            h = STYLE["auto_hue_base"] + STYLE["auto_hue_span"] * min(max(brightness, 0.0), 1.0)

            def tone(dh: float) -> List[np.ndarray]:
                specs = [(0.0, 0.55, 0.10), (0.0, 0.45, 0.80), (0.07, 0.35, 0.88), (0.13, 0.20, 1.0)]
                return [_lin(colorsys.hsv_to_rgb((h + dh + o) % 1.0, s, v)) for o, s, v in specs]

            return tone(0.0), tone(0.06)
        return [_lin(_hex_to_rgb(c)) for c in self.a], [_lin(_hex_to_rgb(c)) for c in self.b]

    def to_json(self) -> dict:
        return {"name": self.name, "auto": self.auto, "a": list(self.a), "b": list(self.b)}


PALETTES: Dict[str, Palette] = {
    p.name: p
    for p in (
        Palette("Deep Ocean", ("#04101c", "#2a6f97", "#61a5c2", "#cfe8f3"), ("#06141a", "#2f8f9d", "#8ecae6", "#e9f6fb")),
        Palette("Dusk", ("#120b1c", "#7b5ea7", "#c08bb0", "#f2d6e0"), ("#170c16", "#a45c8a", "#e0a1a4", "#ffe9d6")),
        Palette("Moss", ("#07120d", "#3a7d5c", "#8bb89a", "#e3f1e0"), ("#0a130e", "#5b8f6a", "#b5cfa0", "#f1f4dc")),
        Palette("Ember", ("#140806", "#a8553a", "#e09f7d", "#fbe5d0"), ("#160a08", "#c47a45", "#f0c08a", "#fff0d9")),
        Palette("Mono Mist", ("#0a0b0d", "#6b7684", "#aeb7c2", "#eef1f4"), ("#0b0c0f", "#7c8794", "#c6cdd6", "#ffffff")),
        Palette("Spectrum (auto)", auto=True),
    )
}
DEFAULT_PALETTE = "Deep Ocean"


# --------------------------------------------------------------------------- #
# GLSL
# --------------------------------------------------------------------------- #
VERT = """#version 330
uniform float u_flip;
out vec2 v_uv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = p;
    gl_Position = vec4((p * 2.0 - 1.0) * vec2(1.0, u_flip), 0.0, 1.0);
}
"""

_COMMON = """in vec2 v_uv;
out vec4 fragColor;
uniform vec2  u_res;
uniform float u_time;       // slow ambient clock
uniform float u_bass;       // smoothed loudness per band group
uniform float u_mid;
uniform float u_treble;
uniform float u_kick;       // fast pulses that jump on a hit and decay
uniform float u_snare;
uniform float u_hat;
uniform float u_flow;       // position of the flowing pattern (pushed by hits)
uniform float u_beat;       // beats since the start of the track (continuous)
uniform float u_intensity;
uniform float u_contrast;
uniform float u_frame;
uniform vec3  u_gain;       // bass, mid, treble gain applied to the spectrum
uniform vec3  u_rip[8];     // ripples: (age in seconds, strength, kind 0=kick 1=mid), strength 0 = unused
uniform float u_chroma[12]; // sounding strength of each pitch class, C..B
uniform sampler2D u_spec;   // 48 bands x 48 frames of history, row 0 = now
uniform vec3  u_c0;
uniform vec3  u_c1;
uniform vec3  u_c2;
uniform vec3  u_c3;

const float NB = 48.0;
const float NH = 48.0;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0)), d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    for (int i = 0; i < 4; i++) { v += a * vnoise(p); p = m * p; a *= 0.5; }
    return v;
}
vec2 aspectP() {
    return (v_uv - 0.5) * vec2(u_res.x / u_res.y, 1.0) * 2.0;
}
// Spectrum energy at horizontal position x (0 = lowest frequency, 1 = highest),
// 'row' frames ago. Gains apply per region and the contrast curve sharpens peaks.
float specAt(float x, float row) {
    float xc = clamp(x, 0.0, 1.0);
    float v = texture(u_spec, vec2(xc * (NB - 1.0) / NB + 0.5 / NB, (row + 0.5) / NH)).r;
    float wl = 1.0 - smoothstep(0.12, 0.38, xc);
    float wh = smoothstep(0.62, 0.88, xc);
    float wm = 1.0 - wl - wh;
    v *= wl * u_gain.x + wm * u_gain.y + wh * u_gain.z;
    return pow(clamp(v, 0.0, 1.6), u_contrast);
}
float dustLayer(vec2 p, float scale, float speed, float seed) {
    vec2 q = p * scale + vec2(seed * 3.1, u_time * speed + seed * 7.7);
    vec2 cell = floor(q);
    vec2 f = fract(q) - 0.5;
    float h = hash21(cell + seed);
    vec2 o = (vec2(hash21(cell + 1.7), hash21(cell + 9.1)) - 0.5) * 0.6;
    float d = length(f - o);
    float tw = 0.55 + 0.45 * sin(u_time * (0.35 + h) + h * 40.0);
    return step(0.90, h) * exp(-d * d * 70.0) * tw;
}
vec3 finish(vec3 c) {
    vec2 d = v_uv - 0.5;
    c *= u_intensity;
    c *= 1.0 - 0.9 * dot(d, d);
    c = 1.0 - exp(-1.15 * c);
    c = pow(c, vec3(1.0 / 2.2));
    float n = hash12(gl_FragCoord.xy + u_frame * vec2(37.0, 17.0))
            + hash12(gl_FragCoord.xy * 1.7 + u_frame * vec2(11.0, 29.0)) - 1.0;
    return clamp(c + n * (1.6 / 255.0), 0.0, 1.0);
}
"""

_ORB = """
void main() {
    vec2 p = aspectP();
    float r = length(p);
    vec3 col = u_c0 * (0.55 + 0.45 * (1.0 - r * 0.5));

    // the orb itself only nudges with the kick; the motion is in the rings and notes
    float R = 0.17 + 0.045 * u_kick + 0.015 * u_bass;
    float body = smoothstep(R + 0.02, R - 0.20, r);
    col += u_c1 * body * body * (0.26 + 0.40 * u_kick + 0.18 * u_bass);
    col += u_c2 * exp(-max(r - R, 0.0) * 6.0) * (0.09 + 0.34 * u_kick + 0.08 * u_bass);
    col += u_c3 * exp(-r * r * 14.0) * (0.05 + 0.25 * u_snare + 0.12 * u_kick);

    // ripples: every low hit and every mid-band hit (snare, new note) launches a ring
    // that really travels outward, so you can see each beat leave the centre
    for (int i = 0; i < 8; i++) {
        vec3 rp = u_rip[i];
        if (rp.y > 0.0) {
            float speed = mix(0.62, 0.95, rp.z);
            float rad = R + 0.02 + rp.x * speed;
            float w = mix(0.024, 0.010, rp.z) * (1.0 + rp.x * 0.4);
            float d = r - rad;
            float fade = exp(-rp.x * mix(1.5, 2.6, rp.z));
            col += mix(u_c2, u_c3, rp.z) * exp(-d * d / (w * w)) * fade * rp.y * mix(0.55, 0.28, rp.z);
        }
    }

    // notes: the 12 pitch classes sit around a circle (C at the top, clockwise);
    // whatever is sounding lights its place, so a melody walks around the ring
    float Rn = 0.62;
    col += u_c2 * exp(-pow((r - Rn) * 90.0, 2.0)) * 0.035;
    for (int i = 0; i < 12; i++) {
        float a = float(i) / 12.0 * 6.2831853;
        vec2 bp = Rn * vec2(sin(a), cos(a));
        float e = u_chroma[i];
        float d2 = dot(p - bp, p - bp);
        col += u_c3 * exp(-d2 * 1500.0) * (0.14 + 2.4 * e * e);
        col += u_c2 * exp(-d2 * 110.0) * e * e * e * 0.70;
    }

    float dust = dustLayer(p, 5.0, 0.020, 1.0) + 0.7 * dustLayer(p, 9.0, 0.032, 2.0);
    col += u_c3 * dust * (0.05 + 0.55 * u_hat + 0.15 * u_treble);

    fragColor = vec4(finish(col), 1.0);
}
"""

_AURORA = """
void main() {
    vec2 p = aspectP();
    float x01 = v_uv.x;

    // the veil is pushed along by every hit and sways once per two beats, locked to the tempo
    float sway = sin(u_beat * 3.14159265) * 0.10;
    float t = u_time * 0.04;
    vec2 q = vec2(p.x * 0.85 + u_flow * 0.9 + sway, p.y * 0.38);
    float n1 = fbm(q + vec2(t * 0.7, t * 0.3));
    float n2 = fbm(q * 1.6 + vec2(-t, 5.0) + n1 * (1.2 + 0.9 * u_mid));
    float veil = smoothstep(0.34, 0.80, n2);

    // frequency map: left is bass, right is treble. Sounding notes light their column.
    float e0 = specAt(x01, 0.0);
    float e1 = specAt(x01, 5.0);
    float column = 0.16 + 1.3 * e0 + 0.5 * e1;

    float v = 0.5 + 0.5 * p.y;
    float vert = smoothstep(0.02, 0.62, v) * (1.0 - 0.55 * smoothstep(0.70, 1.05, v));

    vec3 col = u_c0 * (0.5 + 0.5 * v);
    col += u_c1 * veil * vert * column * 0.55;
    col += u_c2 * veil * n1 * vert * (0.08 + 0.75 * e0 * e0 + 0.25 * u_kick);
    col += u_c3 * pow(veil, 3.0) * vert * (0.03 + 0.25 * e0 * e0);

    float dust = dustLayer(p, 6.0, 0.015, 3.0);
    col += u_c3 * dust * (0.04 + 0.55 * u_hat + 0.12 * u_treble);

    fragColor = vec4(finish(col), 1.0);
}
"""

_WAVES = """
void main() {
    vec2 p = aspectP();
    vec3 col = u_c0 * (0.55 + 0.45 * (1.0 - length(p) * 0.45));
    float x01 = clamp(v_uv.x, 0.0, 1.0);
    float edge = smoothstep(0.0, 0.07, x01) * (1.0 - smoothstep(0.93, 1.0, x01));
    float dxw = 0.02 * u_res.x / u_res.y * 2.0;

    // each line is the spectrum from a moment ago (front line = now, back = 0.6 s ago):
    // a kick makes a bump on the left that rolls back through the lines, and every
    // note is a peak at its own frequency
    for (int i = 0; i < 7; i++) {
        float fi = float(i) / 6.0;
        float row = fi * 36.0;
        float amp = 0.34 - 0.10 * fi;
        float e = 0.5 * (specAt(x01 - 0.006, row) + specAt(x01 + 0.006, row));
        float slope = (specAt(x01 + 0.01, row) - specAt(x01 - 0.01, row)) * amp / dxw;
        float y = -0.62 + fi * 1.12 + e * amp;
        float d = abs(p.y - y) / sqrt(1.0 + slope * slope);
        float line = exp(-d * d * 3600.0);
        float halo = exp(-d * 16.0) * 0.10;
        vec3 lc = mix(u_c1, u_c2, fi);
        col += lc * (line * 0.9 + halo) * edge * (1.0 - 0.55 * fi) * (0.50 + 0.45 * e);
    }
    col += u_c3 * exp(-dot(p, p) * 8.0) * (0.02 + 0.10 * u_kick);
    float dust = dustLayer(p, 6.0, 0.012, 5.0);
    col += u_c3 * dust * (0.03 + 0.4 * u_hat);
    fragColor = vec4(finish(col), 1.0);
}
"""



_RESONANCE = """
// six clean figures (orders 1..4); the two neighbouring pitch classes of each pair
// use the same figure with opposite phase
const vec2 MODES[6] = vec2[6](vec2(1.0, 2.0), vec2(1.0, 3.0), vec2(2.0, 3.0), vec2(1.0, 4.0), vec2(2.0, 4.0), vec2(3.0, 4.0));

// one standing-wave mode of a vibrating plate
float plateMode(vec2 q, vec2 nm, float sgn) {
    vec2 a = cos(3.14159265 * nm.x * q);
    vec2 b = cos(3.14159265 * nm.y * q);
    return a.x * b.y + sgn * b.x * a.y;
}

void main() {
    vec2 p = aspectP();
    float r = length(p);

    // every kick turns the plate a little, then it glides to a stop
    float ang = u_flow * 0.16 + u_time * 0.012;
    float cs = cos(ang), sn = sin(ang);
    vec2 q = vec2(cs * p.x - sn * p.y, sn * p.x + cs * p.y) * 0.64;

    // each sounding note excites its own pattern; a chord is their sum.
    // a faint idle pattern keeps silence from being empty.
    float f = 0.07 * plateMode(q, vec2(2.0, 3.0), 1.0);
    float wsum = 0.07;
    for (int i = 0; i < 12; i++) {
        float e = u_chroma[i];
        if (e > 0.10) {
            float w = e * e * e;                 // the strongest notes dominate the figure
            f += w * plateMode(q, MODES[i / 2], (i % 2 == 0) ? 1.0 : -1.0);
            wsum += w;
        }
    }

    // strikes: a wave leaves the centre and bends the nodal lines as it passes
    for (int i = 0; i < 8; i++) {
        vec3 rp = u_rip[i];
        if (rp.y > 0.0) {
            float rad = rp.x * mix(0.75, 1.0, rp.z);
            float env = exp(-pow((r - rad) / 0.20, 2.0)) * exp(-rp.x * 1.4);
            f += wsum * 0.34 * rp.y * env * sin((r - rad) * 12.0);
        }
    }

    // nodal lines (f = 0), drawn at a constant pixel width via a distance estimate
    float g = max(length(vec2(dFdx(f), dFdy(f))), 1e-5);
    float dpx = abs(f) / g;
    float px = u_res.y / 1080.0;
    float exc = clamp(0.18 + 1.1 * u_bass + 0.6 * u_mid + 0.4 * u_kick, 0.0, 1.0);
    float core = exp(-pow(dpx / (1.3 * px), 2.0));
    float glow = exp(-dpx / (9.0 * px));

    // quiet: the sand barely clings to the lines. loud: they are solid.
    float grain = hash21(floor(gl_FragCoord.xy / (1.6 * px)));
    float sand = mix(step(0.45, grain), 1.0, smoothstep(0.25, 0.8, exc));
    float edge = 1.0 - 0.5 * smoothstep(1.0, 1.9, r);
    float tw = 1.0 + 0.6 * u_hat * hash21(floor(gl_FragCoord.xy / (3.0 * px)));

    vec3 col = u_c0 * (0.5 + 0.5 * (1.0 - r * 0.3));
    col += u_c1 * glow * (0.04 + 0.16 * exc) * edge;
    col += mix(u_c2, u_c3, exc) * core * sand * (0.50 + 0.85 * exc + 0.8 * u_kick) * edge * tw;
    fragColor = vec4(finish(col), 1.0);
}
"""

_STRING = """
// the chord as a wave: every sounding note contributes a sine whose frequency
// ratio is the real musical ratio (C = 1 ... B = 1.89), so the shape of the line
// is the interval structure of what is playing
float chordWave(float x, float lag) {
    float y = 0.0, a = 0.0;
    for (int i = 0; i < 12; i++) {
        float e = u_chroma[i];
        if (e > 0.03) {
            float fr = exp2(float(i) / 12.0);
            y += e * e * sin(6.2831853 * fr * (x * 0.55 - (u_beat - lag) * 0.125) + float(i) * 1.7);
            a += e * e;
        }
    }
    y += 0.12 * sin(6.2831853 * (x * 0.55 - (u_beat - lag) * 0.125));
    return y / max(1.0, a + 0.12);
}

void main() {
    vec2 p = aspectP();
    float px = u_res.y / 1080.0;
    float level = clamp(0.5 * u_bass + 0.5 * u_mid, 0.0, 1.2);
    float amp = 0.07 + 0.26 * level;

    vec3 col = u_c0 * (0.5 + 0.5 * (1.0 - length(p) * 0.25));
    float edge = 1.0 - smoothstep(1.2, 1.95, abs(p.x));

    // five trails of the same string, each a little earlier in the beat: they fan out
    // into a ribbon so you can see which way the wave travels
    for (int k = 0; k < 5; k++) {
        float fk = float(k);
        float lag = fk * 0.30;
        float y = chordWave(p.x, lag);
        // a kick plucks the string at the centre; the pluck runs outward both ways
        for (int i = 0; i < 8; i++) {
            vec3 rp = u_rip[i];
            if (rp.y > 0.0) {
                float rad = max(rp.x - lag * 0.25, 0.0) * mix(0.9, 1.3, rp.z);
                float d = abs(p.x) - rad;
                y += 0.55 * rp.y * exp(-pow(d / 0.16, 2.0)) * exp(-rp.x * 1.5) * cos(d * 22.0);
            }
        }
        y *= amp;
        float slope = dFdx(y) / max(dFdx(p.x), 1e-6);
        float d = abs(p.y - y) / sqrt(1.0 + slope * slope);
        float dpx = d / (2.0 / u_res.y);
        float core = exp(-pow(dpx / (1.25 * px), 2.0));
        float glow = exp(-dpx / (8.0 * px));
        float fade = exp(-fk * 0.75);
        vec3 lc = mix(u_c1, u_c2, clamp(0.5 + 0.5 * p.x, 0.0, 1.0));
        col += (lc * (core * (0.55 + 0.6 * u_kick) + glow * 0.07) + u_c3 * core * 0.35 * u_kick) * fade * edge;
    }
    fragColor = vec4(finish(col), 1.0);
}
"""


@dataclass(frozen=True)
class SceneDef:
    name: str
    frag: str          # GLSL 330 source


def _frag(body: str) -> str:
    return "#version 330\n" + _COMMON + body


SCENES: Dict[str, SceneDef] = {
    s.name: s
    for s in (
        SceneDef("Resonance", _frag(_RESONANCE)),
        SceneDef("Harmonic String", _frag(_STRING)),
        SceneDef("Ripple Orb", _frag(_ORB)),
        SceneDef("Aurora Veil", _frag(_AURORA)),
        SceneDef("Soft Waves", _frag(_WAVES)),
    )
}
DEFAULT_SCENE = "Resonance"


def to_webgl(src: str) -> str:
    """GLSL 330 core -> GLSL ES 300 (WebGL2). Only the header differs."""
    return re.sub(r"^#version 330\s*\n", "#version 300 es\nprecision highp float;\nprecision highp int;\n", src, count=1)


def web_config() -> dict:
    return {
        "vert": to_webgl(VERT),
        "scenes": [{"name": s.name, "frag": to_webgl(s.frag)} for s in SCENES.values()],
        "palettes": [p.to_json() for p in PALETTES.values()],
        "style": STYLE,
        "defaults": {"scene": DEFAULT_SCENE, "palette": DEFAULT_PALETTE},
        "audio": {"n_spec": N_SPEC, "n_hist": N_HIST, "spec_edges": [float(x) for x in SPEC_EDGES],
                  "chroma_fmin": CHROMA_FMIN, "chroma_fmax": CHROMA_FMAX},
    }


# --------------------------------------------------------------------------- #
# Per-frame motion state (mirrored in web/index.html)
# --------------------------------------------------------------------------- #
def _smooth_step(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _follow(y: float, x: float, dt: float, attack_s: float, release_s: float) -> float:
    tau = attack_s if x > y else release_s
    return y + (x - y) * (1.0 - math.exp(-dt / max(tau, 1e-4)))


class SceneState:
    """Everything that evolves frame to frame. Call advance() once per frame."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.bass = self.mid = self.treble = 0.0
        self.kick = self.snare = self.hat = 0.0
        self.bright = 0.4
        self.T = 0.0
        self.flow = 0.0
        self.vel = 0.0
        self.beat = 0.0
        self.shift = 0.0
        self.shift_target = 0.0
        self.last_shift = -1e9
        self.ripples: List[List[float]] = []          # [age, strength, kind]
        self.chroma = np.zeros(12, dtype=np.float64)

    def advance(self, c, dt: float, motion: float = 1.0, contrast: float = 1.4) -> None:
        S = STYLE
        shape = lambda v: float(max(v, 0.0) ** contrast)   # noqa: E731
        a, r = S["band_attack_s"], S["band_release_s"]
        self.bass = _follow(self.bass, shape(c.sub_bass), dt, a, r)
        self.mid = _follow(self.mid, shape(c.mid), dt, a, r)
        self.treble = _follow(self.treble, shape(c.treble), dt, a, r)
        self.kick = _follow(self.kick, c.kick, dt, S["kick_attack_s"], S["kick_release_s"])
        self.snare = _follow(self.snare, c.snare, dt, S["snare_attack_s"], S["snare_release_s"])
        self.hat = _follow(self.hat, c.hat, dt, S["hat_attack_s"], S["hat_release_s"])
        self.bright += (c.brightness - self.bright) * (1.0 - math.exp(-dt / S["brightness_tau_s"]))
        self.beat = float(c.beat)
        self.T += dt * motion

        # ripples: age everything, then launch new ones on this frame's hits
        for rp in self.ripples:
            rp[0] += dt
        self.ripples = [rp for rp in self.ripples if rp[0] < S["ripple_life_s"]]
        if c.kick > 0.0:
            self.ripples.append([0.0, float(c.kick), 0.0])
        if c.snare >= 0.5:
            self.ripples.append([0.0, float(c.snare), 1.0])
        self.ripples = self.ripples[-int(S["ripple_max"]):]

        # flow: every hit pushes the pattern forward, then it glides to a stop
        self.vel += S["flow_kick"] * c.kick + S["flow_snare"] * c.snare
        self.vel *= math.exp(-dt / S["flow_decay_s"])
        self.flow += dt * motion * (S["flow_base"] + S["flow_level"] * self.bass) + self.vel * dt * motion

        # notes
        target = c.chroma if c.chroma is not None else np.zeros(12)
        att = 1.0 - math.exp(-dt / S["chroma_attack_s"])
        rel = 1.0 - math.exp(-dt / S["chroma_release_s"])
        self.chroma += (target - self.chroma) * np.where(target > self.chroma, att, rel)

        # slow palette drift on any onset
        if c.onset >= S["shift_min_onset"] and c.time - self.last_shift >= S["shift_min_gap_s"]:
            self.shift_target = 1.0 - self.shift_target
            self.last_shift = c.time
        self.shift += (self.shift_target - self.shift) * (1.0 - math.exp(-dt / S["shift_tau_s"]))

    def ripple_array(self) -> np.ndarray:
        out = np.zeros((8, 3), dtype=np.float32)
        for i, rp in enumerate(self.ripples[-8:]):
            out[i] = rp
        return out

    def colors(self, palette: Palette) -> List[np.ndarray]:
        a, b = palette.resolved(self.bright)
        m = _smooth_step(self.shift)
        return [a[k] * (1.0 - m) + b[k] * m for k in range(4)]
