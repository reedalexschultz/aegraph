"""A chaotic three-body system that nonetheless stays inside the frame.

We start from the famous stable figure-eight choreography, then give the bodies a
tiny velocity kick. That small perturbation is enough to break the delicate
periodic orbit: the three bodies wander off the figure-eight and into genuinely
chaotic motion (sensitive dependence on initial conditions).

To keep everything on screen we (a) soften gravity slightly so near-collisions
don't fling a body to infinity, and (b) lock the axes to the trajectory's true
bounding box. A faint guide line traces the *original* figure-eight so you can
watch the real orbit peel away from the ideal one.

Per body we keyframe Position (the chaotic dance) and Ellipse-Size (a pulse tied
to speed). Flip hold_keyframes=True for a stepped look.
"""

RENDERING_TIME = "VERY_SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np

THEME = "Blood Moon"

G = 1.0
m = np.array([1.0, 1.0, 1.0])
SOFTENING = 0.04          # gravitational softening keeps close approaches finite


def accelerations(pos):
    acc = np.zeros_like(pos)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            d = pos[j] - pos[i]
            r2 = d[0] ** 2 + d[1] ** 2 + SOFTENING ** 2
            acc[i] += G * m[j] * d / (r2 ** 1.5)
    return acc


def derivative(state):
    pos = state[:6].reshape(3, 2)
    vel = state[6:].reshape(3, 2)
    return np.concatenate([vel.ravel(), accelerations(pos).ravel()])


# Figure-eight initial conditions ...
pos0 = np.array([[-0.97000436, 0.24308753],
                 [0.97000436, -0.24308753],
                 [0.0, 0.0]])
v3 = np.array([-0.93240737, -0.86473146])
v1 = -v3 / 2.0
vel0 = np.array([v1, v1, v3])

# ... plus a tiny kick that tips the system into chaos.
rng = np.random.default_rng(4)
vel0 = vel0 + rng.normal(0, 0.018, size=vel0.shape)

state = np.concatenate([pos0.ravel(), vel0.ravel()])

# --- Integrate with RK4 ---
period = 6.32591398
n_periods = 3.0
n_steps = 9000
dt = (period * n_periods) / n_steps

states = np.empty((n_steps + 1, 12))
for s in range(n_steps + 1):
    states[s] = state
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

pos_all = states[:, :6].reshape(-1, 3, 2)
vel_all = states[:, 6:].reshape(-1, 3, 2)

# Sample evenly spaced animation frames.
n_frames = 120
idx = np.linspace(0, n_steps, n_frames).astype(int)
x_frames = pos_all[idx, :, 0]
y_frames = pos_all[idx, :, 1]

speed = np.hypot(vel_all[:, :, 0], vel_all[:, :, 1])
speed_frames = speed[idx]
s_min, s_max = speed_frames.min(), speed_frames.max()
radius_frames = 9.0 + 13.0 * (speed_frames - s_min) / (s_max - s_min)

# One color per body, constant over time.
body_colors = [[255, 196, 120], [120, 200, 255], [255, 110, 130]]
color_frames = [body_colors for _ in range(n_frames)]

# Faint guide line: the *unperturbed* figure-eight, for contrast.
guide_t = np.linspace(0, 2 * np.pi, 240)
guide_x = np.sin(guide_t)
guide_y = np.sin(guide_t) * np.cos(guide_t)

# --- Build the graph ---
chaos = (
    AEGraph(theme=THEME, comp_name="Chaotic Three-Body", fps=60,
            drop_shadow=True, cinematic_effects=True, xaxis_location="bottom", yaxis_location="left",
            plot_frame=True)
    .set_title("Chaotic Three-Body Problem")
    .set_subtitle("size = velocity")
    .set_xlabel("x")
    .set_ylabel("y")
    .grid()
)

# chaos.plot(guide_x, guide_y, color=[110, 90, 110], linewidth=2, animate=0)

chaos.scatter_evolving(
    x_frames, y_frames,
    radius_frames=radius_frames,
    color_frames=color_frames,
    frame_duration=0.08,          # 120 * 0.08 s ~= 9.6 s
    hold_keyframes=False,
    alpha=1.0,
    drop_shadow=True,
)

# Lock the frame to the full extent of the chaotic motion so nothing drifts off.
xs = x_frames
ys = y_frames
pad_x = 0.1 * (xs.max() - xs.min())
pad_y = 0.1 * (ys.max() - ys.min())
chaos.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
chaos.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
chaos.set_xticks().set_yticks()

chaos.render()
