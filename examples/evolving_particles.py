"""Evolving scatter: a particle cloud advected by a 2D dynamical system.

Each particle follows the spiral-sink ODE

    dx/dt = -damp * x - omega * y
    dy/dt =  omega * x - damp * y

which winds every particle inward toward the origin. For each particle we keyframe
three things over time, all on the same dot:

    - position  -> the swirling motion (Position keyframes)
    - size      -> radius mapped to current speed (Ellipse Size keyframes)
    - color     -> mapped to current speed via the theme gradient (Fill Color keyframes)

Set hold_keyframes=True to make every dot snap between frames instead of gliding.
"""

RENDERING_TIME = "LONG"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np

THEME = "Royal Static"

# --- Simulate the particle cloud ---
rng = np.random.default_rng(11)
n_particles = 60
n_frames = 90
dt = 0.08
damp = 0.18
omega = 1.6

# Seed particles on a ring so the inward spiral reads clearly.
angles = rng.uniform(0, 2 * np.pi, n_particles)
radii = rng.uniform(3.0, 6.0, n_particles)
px = radii * np.cos(angles)
py = radii * np.sin(angles)

x_frames = np.zeros((n_frames, n_particles))
y_frames = np.zeros((n_frames, n_particles))
speed_frames = np.zeros((n_frames, n_particles))

for f in range(n_frames):
    x_frames[f] = px
    y_frames[f] = py
    dxdt = -damp * px - omega * py
    dydt = omega * px - damp * py
    speed_frames[f] = np.sqrt(dxdt ** 2 + dydt ** 2)
    px = px + dxdt * dt
    py = py + dydt * dt

# Map speed -> bubble radius (faster = bigger).
s_min, s_max = speed_frames.min(), speed_frames.max()
radius_frames = 4.0 + 14.0 * (speed_frames - s_min) / (s_max - s_min)

# --- Build the evolving scatter graph ---
particles = (
    AEGraph(theme=THEME, comp_name="Spiral Particles", drop_shadow=True, cinematic_effects=True, xaxis_location="bottom", yaxis_location="left", plot_frame=True)
    .set_title("Particles in a Spiral Field")
    .set_subtitle("size & color follow speed")
    .set_xlabel("x")
    .set_ylabel("y")
    .grid()
)
particles.scatter_evolving(
    x_frames, y_frames,
    radius_frames=radius_frames,
    c_frames=speed_frames,        # drives per-dot color via the theme gradient
    frame_duration=0.12,
    hold_keyframes=False,         # smooth glide; True = snapping stop-motion
    alpha=0.9,
    drop_shadow=True,
)
particles.set_xticks().set_yticks()

particles.render()
