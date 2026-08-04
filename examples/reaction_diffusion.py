"""Evolving heatmap: a Gray-Scott reaction-diffusion pattern growing over time.

Two virtual chemicals U and V diffuse and react on a grid following the
Gray-Scott model:

    U_t = Du * lap(U) - U*V^2 + F*(1 - U)
    V_t = Dv * lap(V) + U*V^2 - (F + k)*V

Seeded from a small blob, V self-organizes into spots / mazes / coral textures.
We snapshot the V field every few steps and feed the stack to
``heatmap_evolving`` so every cell's color is keyframed -- a whole pattern that
blooms and spreads inside the comp.

The nuanced cousin of the gallery ``heatmap_evolving`` demo: real PDE data,
periodic boundaries, and a color scale locked across the whole animation.
"""

RENDERING_TIME = "VERY_LONG"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np

THEME = "Aurora Veil"

# --- Gray-Scott parameters (coral-growth regime) ---
N = 24
Du, Dv = 0.16, 0.08
F, k = 0.060, 0.062
rng = np.random.default_rng(3)


def laplacian(a):
    return (-4 * a
            + np.roll(a, 1, 0) + np.roll(a, -1, 0)
            + np.roll(a, 1, 1) + np.roll(a, -1, 1))


U = np.ones((N, N))
V = np.zeros((N, N))
# Seed a few noisy squares of chemical V.
for _ in range(8):
    cx, cy = rng.integers(8, N - 8, size=2)
    U[cy - 4:cy + 4, cx - 4:cx + 4] = 0.50
    V[cy - 4:cy + 4, cx - 4:cx + 4] = 0.25
V += 0.02 * rng.random((N, N))

n_frames = 100
steps_per_frame = 18
frames = []
for f in range(n_frames):
    frames.append(V.copy())
    for _ in range(steps_per_frame):
        uvv = U * V * V
        U = U + Du * laplacian(U) - uvv + F * (1 - U)
        V = V + Dv * laplacian(V) + uvv - (F + k) * V

frames = np.array(frames)        # shape (n_frames, N, N)

# --- Build the evolving heatmap ---
rd = (
    AEGraph(theme=THEME, comp_name="Reaction Diffusion", fps=60,
            cinematic_effects=True, plot_frame=True,
            xaxis_location="bottom", yaxis_location="left")
    .set_title("Gray-Scott Reaction-Diffusion")
    .set_subtitle("chemical V self-organizing over time")
)
rd.heatmap_evolving(
    frames,
    gradient=([8, 18, 26], [255, 120, 200]),    # dark teal -> magenta
    vmin=0.0, vmax=0.45,
    frame_duration=0.06,
    colorbar_label="concentration of V",
    fade_in=0.5,
)
rd.set_xticks().set_yticks()

rd.render()
