"""The three-body problem: the stable figure-eight choreography (Midnight Neon theme).

Three equal masses chase each other around a single figure-eight orbit -- a real
periodic solution of Newtonian gravity (Chenciner & Montgomery, 2000). We
integrate the equations of motion with RK4, then animate it two ways:

    - a faint static guide line tracing the full figure-eight, and
    - an evolving scatter of the three bodies whose Position keyframes carry the
      orbit and whose Ellipse-Size keyframes pulse with each body's speed.

Because the orbit is periodic, the animation loops seamlessly. Flip
hold_keyframes=True for a stop-motion feel.
"""

RENDERING_TIME = "VERY_SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np

THEME = "Obsidian Sakura"

# --- Equations of motion (G = 1, unit masses) ---
G = 1.0
m = np.array([1.0, 1.0, 1.0])


def accelerations(pos):
    acc = np.zeros_like(pos)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            d = pos[j] - pos[i]
            r = np.hypot(d[0], d[1])
            acc[i] += G * m[j] * d / (r ** 3)
    return acc


def derivative(state):
    pos = state[:6].reshape(3, 2)
    vel = state[6:].reshape(3, 2)
    return np.concatenate([vel.ravel(), accelerations(pos).ravel()])


# Classic figure-eight initial conditions.
pos0 = np.array([[-0.97000436, 0.24308753],
                 [0.97000436, -0.24308753],
                 [0.0, 0.0]])
v3 = np.array([-0.93240737, -0.86473146])
v1 = -v3 / 2.0
vel0 = np.array([v1, v1, v3])
state = np.concatenate([pos0.ravel(), vel0.ravel()])

# --- Integrate one full period with RK4 ---
period = 6.32591398
n_steps = 4000
dt = period / n_steps

states = np.empty((n_steps + 1, 12))
for s in range(n_steps + 1):
    states[s] = state
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

pos_all = states[:, :6].reshape(-1, 3, 2)   # (steps, body, xy)
vel_all = states[:, 6:].reshape(-1, 3, 2)

# Sample evenly spaced frames for the animated bodies.
n_frames = 90
idx = np.linspace(0, n_steps, n_frames).astype(int)
x_frames = pos_all[idx, :, 0]               # (n_frames, 3)
y_frames = pos_all[idx, :, 1]

speed = np.hypot(vel_all[:, :, 0], vel_all[:, :, 1])
speed_frames = speed[idx]                   # (n_frames, 3)
s_min, s_max = speed_frames.min(), speed_frames.max()
radius_frames = 9.0 + 12.0 * (speed_frames - s_min) / (s_max - s_min)

# One color per body, constant across time.
body_colors = [[255, 210, 80], [70, 210, 255], [255, 90, 160]]
color_frames = [body_colors for _ in range(n_frames)]

# Full figure-eight path for the guide line (all three bodies share this curve).
guide = pos_all[::15, 0, :]
guide_x = guide[:, 0]
guide_y = guide[:, 1]

# --- Build the graph ---
orbit = (
    AEGraph(theme=THEME, comp_name="Three-Body Figure Eight", fps=60,
            drop_shadow=True, cinematic_effects=True, xaxis_location="bottom", yaxis_location="left", plot_frame=True)
    .set_title("Three-Body Problem")
    .set_subtitle("the figure-eight choreography")
    .set_xlabel("x")
    .set_ylabel("y")
    .grid()
)

# Faint static trajectory (animate=0 -> drawn instantly, no trim-on).
orbit.plot(guide_x, guide_y, color=[120, 130, 150], linewidth=2, animate=0)

# The three moving bodies.
orbit.scatter_evolving(
    x_frames, y_frames,
    radius_frames=radius_frames,
    color_frames=color_frames,
    frame_duration=0.05,
    hold_keyframes=False,
    alpha=1.0,
    drop_shadow=True,
)
orbit.set_xticks().set_yticks()

orbit.render()
