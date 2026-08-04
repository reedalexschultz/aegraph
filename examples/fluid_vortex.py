"""Fluid field: a swirling vortex (quiver) with tracer particles spiraling in.

The background arrows are a quiver plot of a vortex + sink velocity field:

    u = -S * y / r^2  -  D * x / r       (rotation + inward pull)
    v =  S * x / r^2  -  D * y / r

Arrows are drawn uniform-length and colored by flow speed. On top, a cloud of
massless tracer particles is released and advected by the very same field, so
they wind inward exactly along the streamlines -- the "moving parts" inside the
flow.
"""

RENDERING_TIME = "LONG"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np

THEME = "Coral Reef"
S, D = 1.6, 0.55          # swirl strength, inward (sink) strength


def velocity(x, y):
    r2 = x * x + y * y + 0.25       # soften the singularity at the center
    r = np.sqrt(r2)
    u = -S * y / r2 - D * x / r
    v = S * x / r2 - D * y / r
    return u, v


# --- Quiver field on a grid ---
g = np.linspace(-3.0, 3.0, 15)
X, Y = np.meshgrid(g, g)
U, V = velocity(X, Y)
speed = np.hypot(U, V)

# --- Advect tracer particles through the field ---
rng = np.random.default_rng(7)
n_particles = 90
n_frames = 110
dt = 0.05
ang = rng.uniform(0, 2 * np.pi, n_particles)
rad = rng.uniform(1.2, 2.9, n_particles)
px = rad * np.cos(ang)
py = rad * np.sin(ang)

x_frames = np.zeros((n_frames, n_particles))
y_frames = np.zeros((n_frames, n_particles))
spd_frames = np.zeros((n_frames, n_particles))
for f in range(n_frames):
    x_frames[f], y_frames[f] = px, py
    u1, v1 = velocity(px, py)
    spd_frames[f] = np.hypot(u1, v1)
    # RK2 step
    u2, v2 = velocity(px + 0.5 * dt * u1, py + 0.5 * dt * v1)
    px = px + dt * u2
    py = py + dt * v2

vortex = (
    AEGraph(theme=THEME, comp_name="Vortex Flow", fps=60,
            drop_shadow=True, cinematic_effects=True,
            xaxis_location="bottom", yaxis_location="left", plot_frame=True)
    .set_title("Vortex Flow")
    .set_subtitle("a quiver velocity field with tracer particles")
    .set_xlabel("x")
    .set_ylabel("y")
    .grid()
)

vortex.quiver(
    X, Y, U, V,
    scale=0.15, normalize=False,           # uniform arrows: show direction
    c=True, gradient=([90, 120, 230], [255, 120, 215]),   # color by speed
    width=3.5, headwidth=3.0, headlength=10.0,
    alpha=0.85, animate=1.4,
)

vortex.scatter_evolving(
    x_frames, y_frames,
    c_frames=spd_frames,
    gradient=([120, 210, 255], [255, 245, 170]),
    radius=6,
    frame_duration=0.09,
    hold_keyframes=False,
    alpha=0.95,
    fade_in=0.4,
    drop_shadow=True,
)

vortex.set_xlim(-3.2, 3.2).set_ylim(-3.2, 3.2)
vortex.set_xticks().set_yticks()

vortex.render()
