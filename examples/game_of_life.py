"""Evolving heatmap: Conway's Game of Life playing out on a grid.

The classic cellular automaton. Every step, each cell lives or dies from its 8
neighbors:

    * a live cell with 2 or 3 live neighbors survives,
    * a dead cell with exactly 3 live neighbors is born,
    * everything else dies (under- or over-population).

Instead of a flat black/white board, we track each live cell's *age* (how many
steps it has survived) and color by it, so long-lived structures glow hot while
freshly born cells start cool -- gliders streak, oscillators pulse, and still
lifes burn steadily. The age field is fed to ``heatmap_evolving`` so every
cell's color is keyframed.
"""

RENDERING_TIME = "VERY_LONG"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np

THEME = "Retro Terminal"

N = 23                      # grid is N x N cells
n_frames = 130
max_age = 12                 # age at which a cell is fully "hot"
rng = np.random.default_rng(7)

# --- Seed: a random soup plus a couple of gliders for guaranteed motion ---
alive = (rng.random((N, N)) < 0.18)

glider = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 1, 1]], dtype=bool)
alive[2:5, 2:5] = glider
alive[10:13, 20:23] = glider[::-1]      # a second glider, flipped


def step(a):
    """One Game of Life update with wraparound (toroidal) edges."""
    n = sum(np.roll(np.roll(a, dr, 0), dc, 1)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0))
    return (a & ((n == 2) | (n == 3))) | (~a & (n == 3))


age = np.where(alive, 1, 0).astype(float)
frames = [np.minimum(age, max_age) / max_age]
for _ in range(n_frames - 1):
    alive = step(alive)
    age = np.where(alive, age + 1, 0)        # survivors age, dead reset to 0
    frames.append(np.minimum(age, max_age) / max_age)

frames = np.array(frames)        # shape (n_frames, N, N), values in [0, 1]

# --- Build the evolving heatmap ---
life = (
    AEGraph(theme=THEME, comp_name="Game of Life", fps=60,
            cinematic_effects=True, plot_frame=True,
            xaxis_location="bottom", yaxis_location="left")
    .set_title("Conway's Game of Life")
    .set_subtitle("cells colored by how long they've survived")
)
life.heatmap_evolving(
    frames,
    gradient=([10, 16, 10], [255, 255, 120]),   # dead/background -> long-lived
    vmin=0.0, vmax=1.0,
    gap=0.12,                                    # a little grout between cells
    frame_duration=0.07,
    colorbar_label="cell age",
    fade_in=0.4,
    reveal="diagonal"
)
life.set_xticks().set_yticks()

life.render()
