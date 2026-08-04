"""A cool differential equation: the Lorenz attractor blooming over ~10 seconds.

The Lorenz system

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

is the textbook example of deterministic chaos. We release a cloud of points from
nearly-identical starting positions and integrate them all with RK4. Projected onto
the classic (x, z) plane, the swarm sweeps out the butterfly wings, and because the
system is chaotic the initially-tight cloud smears across both lobes -- sensitive
dependence on initial conditions, made visible.

Per point we keyframe:
    - position -> the trajectory through the attractor (projected onto x, z)
    - size  -> mapped to the Lorenz y coordinate, the axis pointing out of the
               (x, z) projection plane. Points swinging toward the viewer (high y)
               grow; points falling away (low y) shrink, faking depth/perspective.
    - color -> hue follows instantaneous speed via the theme gradient, then the
               whole color is dimmed by depth (far/low y = darker) so receding
               points fall into shadow, reinforcing the perspective.

Runs for ~10 seconds. Set hold_keyframes=True for a strobe-like stepped look.
"""

RENDERING_TIME = "LONG"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np

THEME = "Infrared Night"

# --- Lorenz parameters (the canonical chaotic regime) ---
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0


def lorenz(state):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    return np.stack([sigma * (y - x),
                     x * (rho - z) - y,
                     x * y - beta * z], axis=-1)


# --- Seed a tight cloud of points near one starting location ---
n_points = 50
rng = np.random.default_rng(7)
state = np.array([1.0, 1.0, 20.0]) + rng.normal(0, 10, size=(n_points, 3))

# Integrate with RK4. We sub-step between captured frames so the motion is smooth.

# high load
n_frames = 600
sub_steps = 12
dt = 0.002

# regular
n_frames = 150
sub_steps = 12
dt = 0.005

xz = np.zeros((n_frames, n_points, 2))     # projection onto (x, z)
speed = np.zeros((n_frames, n_points))
depth = np.zeros((n_frames, n_points))     # Lorenz y -> out-of-plane depth

for f in range(n_frames):
    xz[f, :, 0] = state[:, 0]
    xz[f, :, 1] = state[:, 2]
    depth[f] = state[:, 1]
    v = lorenz(state)
    speed[f] = np.linalg.norm(v, axis=1)
    for _ in range(sub_steps):
        k1 = lorenz(state)
        k2 = lorenz(state + 0.5 * dt * k1)
        k3 = lorenz(state + 0.5 * dt * k2)
        k4 = lorenz(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

x_frames = xz[:, :, 0]
y_frames = xz[:, :, 1]

# Depth (Lorenz y) -> bubble radius. Points swinging toward the viewer (high y)
# grow; points falling away (low y) shrink, giving the swarm a sense of perspective.
d_min, d_max = depth.min(), depth.max()
norm_depth = (depth - d_min) / (d_max - d_min)   # 0 = far, 1 = near
radius_frames = 3.0 + 11.0 * norm_depth

# --- Build the evolving scatter graph ---
lorenz_graph = (
    AEGraph(theme=THEME, comp_name="Lorenz Attractor", fps=60,
            drop_shadow=True, cinematic_effects=True,
            xaxis_location="bottom", yaxis_location="left",
            plot_frame=True)
    .set_title("Lorenz Attractor")
    .set_subtitle("30 points over 10 s")
    .set_xlabel("x")
    .set_ylabel("z")
    .grid()
)

# Build explicit per-point colors: hue from speed along the theme gradient, then
# dimmed by depth so receding points (low y) fall into shadow.
low, high = (np.array(c) for c in lorenz_graph._gradient_endpoints(None))
norm_speed = np.sqrt((speed - speed.min()) / (speed.max() - speed.min()))
base_color = low + norm_speed[..., None] * (high - low)   # (n_frames, n_points, 3)
DEPTH_DIM = 0.4                                            # darkest factor for far points
brightness = DEPTH_DIM + (1.0 - DEPTH_DIM) * norm_depth   # (n_frames, n_points)
color_frames = (base_color * brightness[..., None]).tolist()

lorenz_graph.scatter_evolving(
    x_frames, y_frames,
    radius_frames=radius_frames,
    color_frames=color_frames,
    frame_duration=0.1,
    hold_keyframes=False,
    alpha=0.88,
    fade_in=0.6,
    drop_shadow=True,
)

# Lock limits to the attractor's true extent so the butterfly fills the frame.
pad_x = 0.08 * (x_frames.max() - x_frames.min())
pad_y = 0.08 * (y_frames.max() - y_frames.min())
lorenz_graph.set_xlim(x_frames.min() - pad_x, x_frames.max() + pad_x)
lorenz_graph.set_ylim(y_frames.min() - pad_y, y_frames.max() + pad_y)
lorenz_graph.set_xticks().set_yticks()

lorenz_graph.render()
