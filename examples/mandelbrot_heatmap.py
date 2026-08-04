"""Static heatmap: the Mandelbrot set as escape-time "squares of heat".

For every point c on a grid in the complex plane we iterate z -> z^2 + c and
record how many steps it takes |z| to escape past 2. That escape count maps onto
a color gradient, so the famous fractal filigree appears as a field of colored
cells. Points that never escape (the set itself) are masked with ``np.nan`` so
they render as empty cells -- a built-in "black body" interior.

This is the nuanced cousin of the gallery ``heatmap`` demo: same method, real
fractal data, a masked interior, and a tuned color range.
"""

RENDERING_TIME = "SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np

THEME = "Plasma Field"

# --- Sample the complex plane ---
NX, NY = 22*1, 16*1
max_iter = 80
xs = np.linspace(-2.2, 0.8, NX)
ys = np.linspace(-1.25, 1.25, NY)
C = xs[None, :] + 1j * ys[:, None]      # shape (NY, NX)

Z = np.zeros_like(C)
escape = np.full(C.shape, np.nan)        # NaN == still bounded (in the set)
alive = np.ones(C.shape, dtype=bool)
for n in range(max_iter):
    Z[alive] = Z[alive] ** 2 + C[alive]
    escaped = alive & (np.abs(Z) > 2.0)
    # Smooth escape value for a continuous gradient instead of integer banding.
    escape[escaped] = n + 1 - np.log(np.log(np.abs(Z[escaped]))) / np.log(2)
    alive &= ~escaped

# --- Build the heatmap ---
mandel = (
    AEGraph(theme=THEME, comp_name="Mandelbrot Set", drop_shadow=False,
            cinematic_effects=True, plot_frame=True,
            xaxis_location="bottom", yaxis_location="left")
    .set_title("The Mandelbrot Set")
    .set_subtitle("escape time on the complex plane")
    .set_xlabel("Re(c)")
    .set_ylabel("Im(c)")
)
mandel.heatmap(
    escape, x=xs, y=ys,
    gradient=([13, 8, 30], [244, 226, 84]),   # deep indigo -> bright yellow
    vmin=0, vmax=45,
    colorbar_label="escape iterations",
    animate=2.0,
    reveal="diagonal"
)
mandel.set_xticks().set_yticks()

mandel.render()
