"""Evolving line: the 1D heat equation diffusing over time (Ember Blueprint theme).

We numerically integrate the heat equation

    du/dt = alpha * d^2u/dx^2

starting from a couple of sharp spikes, and capture a snapshot of u(x) every few
steps. Each snapshot becomes one keyframe on the line's *Path* property, so After
Effects morphs the curve from spiky to smooth right in the comp.

Toggle the morph style with hold_keyframes:
    - False (default): LINEAR keyframes -> the curve smoothly morphs between frames.
    - True:            HOLD keyframes  -> the curve snaps from one frame to the next.
"""

RENDERING_TIME = "MODERATE"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
from aegraph_config import config
import numpy as np

THEME = "Ember Blueprint"

# --- Simulate the heat equation ---
N = 400                      # spatial samples (vertex count, constant across frames)
x = np.linspace(0.0, 1.0, N)
dx = x[1] - x[0]
alpha = 0.6
dt = 0.4 * dx * dx / alpha    # stable explicit time step

u = np.zeros(N)
u[N // 4] = 1.0               # two initial spikes

for i in range(10):
    u[3 * N // 4 + i] = 0.8

frames = []
steps_per_frame = 60
n_frames = 50
for f in range(n_frames):
    frames.append(u.copy())
    for _ in range(steps_per_frame):
        lap = np.zeros_like(u)
        lap[1:-1] = u[2:] - 2.0 * u[1:-1] + u[:-2]
        u = u + alpha * dt / (dx * dx) * lap
        u[0] = u[-1] = 0.0    # fixed (Dirichlet) ends

y_frames = np.array(frames)   # shape (n_frames, N)

# --- Build the evolving graph ---
heat = (
    AEGraph(theme=THEME, comp_name="Heat Equation", drop_shadow=True, cinematic_effects=True)
    .set_title("1D Heat Equation")
    .set_subtitle("u_t = alpha * u_xx")
    .set_xlabel("x")
    .set_ylabel("u(x, t)")
    .grid()
    .add_legend(legend_pos="top_right")
)
heat.plot_evolving(
    x, y_frames,
    frame_duration=0.18,      # seconds between snapshots
    hold_keyframes=False,     # smooth morph; set True for a snapping "stop-motion" look
    color=config.object_color_1,
    label="u(x, t)",
    linewidth=5,
    drop_shadow=True,
)
heat.set_xlim(0, 1).set_ylim(0, 0.2).set_xticks().set_yticks()

heat.render()
