"""Fluid field: the oscillating double gyre -- the canonical chaotic-mixing flow.

The double gyre is the textbook time-dependent velocity field used to study ocean
mixing and Lagrangian coherent structures:

    f(x,t) = a(t) x^2 + b(t) x,   a = eps*sin(wt),  b = 1 - 2*eps*sin(wt)
    u = -pi A sin(pi f) cos(pi y)
    v =  pi A cos(pi f) sin(pi y) * df/dx

Two counter-rotating gyres breathe back and forth. The quiver shows a snapshot of
the field; the tracer particles are advected by the full *time-varying* field, so
the cloud stretches and folds along the dividing streamline -- chaotic mixing made
visible.
"""

RENDERING_TIME = "LONG"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np

THEME = "Deep Ocean"
A = 0.1
EPS = 0.25
OMEGA = 2 * np.pi / 10.0


def velocity(x, y, t):
    a = EPS * np.sin(OMEGA * t)
    b = 1.0 - 2.0 * EPS * np.sin(OMEGA * t)
    f = a * x ** 2 + b * x
    dfdx = 2.0 * a * x + b
    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y)
    v = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * dfdx
    return u, v


# --- Quiver snapshot at t = 0 ---
gx = np.linspace(0.02, 1.98, 17)
gy = np.linspace(0.04, 0.96, 9)
X, Y = np.meshgrid(gx, gy)
U, V = velocity(X, Y, 0.0)

# --- Advect tracers through the time-varying field ---
rng = np.random.default_rng(3)
n_particles = 120
n_frames = 120
dt = 0.08
px = rng.uniform(0.1, 1.9, n_particles)
py = rng.uniform(0.1, 0.9, n_particles)

x_frames = np.zeros((n_frames, n_particles))
y_frames = np.zeros((n_frames, n_particles))
spd_frames = np.zeros((n_frames, n_particles))
t = 0.0
for f in range(n_frames):
    x_frames[f], y_frames[f] = px, py
    u1, v1 = velocity(px, py, t)
    spd_frames[f] = np.hypot(u1, v1)
    u2, v2 = velocity(px + 0.5 * dt * u1, py + 0.5 * dt * v1, t + 0.5 * dt)
    px = np.clip(px + dt * u2, 0.0, 2.0)
    py = np.clip(py + dt * v2, 0.0, 1.0)
    t += dt

gyre = (
    AEGraph(theme=THEME, comp_name="Double Gyre", fps=60,
            drop_shadow=True, cinematic_effects=True,
            xaxis_location="bottom", yaxis_location="left", plot_frame=True)
    .set_title("The Double Gyre")
    .set_subtitle("a flow folds a cloud of tracers")
    .set_xlabel("x")
    .set_ylabel("y")
    .grid()
)

gyre.quiver(
    X, Y, U, V,
    scale=0.03, normalize=True,
    c=True, gradient=([40, 90, 150], [120, 230, 240]),
    width=3.0, headwidth=3.0, headlength=8.0,
    alpha=0.8, animate=1.4,
)

gyre.scatter_evolving(
    x_frames, y_frames,
    c_frames=spd_frames,
    gradient=([90, 200, 220], [255, 240, 150]),
    radius=5,
    frame_duration=0.08,
    hold_keyframes=False,
    alpha=0.95,
    fade_in=0.4,
    drop_shadow=True,
)

gyre.set_xlim(0.0, 2.0).set_ylim(0.0, 1.0)
gyre.set_xticks().set_yticks()

gyre.render()
