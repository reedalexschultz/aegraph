"""Two synchronized views of one dynamical system, side by side.

A damped pendulum  theta'' + b*theta' + sin(theta) = 0  is rewritten as a
first-order system in state (theta, omega):

    d(theta)/dt = omega
    d(omega)/dt = -sin(theta) - b*omega

Left panel: the **phase plane** -- a quiver plot of that vector field with the
trajectory spiraling into the origin drawn on top.
Right panel: the **same trajectory with respect to time** -- theta(t) and
omega(t) drawn progressively.

Both panels share the simulation clock: trajectories are drawn with
``plot_evolving`` keyframes at the actual integration times (not a trim-path
easy-ease), so the curve grows in sync with the pendulum's motion.
"""

RENDERING_TIME = "MODERATE"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

import numpy as np
from aegraph import subplots

THEME = "Cyber Grape"
B = 0.30                        # damping coefficient


def deriv(theta, omega):
    return omega, -np.sin(theta) - B * omega


def growing_trace(series):
    """Return (n_frames, n) arrays whose rows reveal ``series`` left-to-right.

    Future samples are pinned to the current tip so every row has the same
    vertex count (required by ``plot_evolving``) while the visible trace grows.
    """
    n = series.size
    out = np.empty((n, n))
    for f in range(n):
        out[f, : f + 1] = series[: f + 1]
        out[f, f + 1 :] = series[f]
    return out


# --- Integrate the trajectory (RK4) ---
# One sample per displayed frame so keyframe times match simulation time exactly.
t = np.linspace(0.0, 14.0, 121)
dt = t[1] - t[0]
theta = np.zeros_like(t)
omega = np.zeros_like(t)
theta[0], omega[0] = 2.7, 0.0
for i in range(t.size - 1):
    k1t, k1o = deriv(theta[i], omega[i])
    k2t, k2o = deriv(theta[i] + 0.5 * dt * k1t, omega[i] + 0.5 * dt * k1o)
    k3t, k3o = deriv(theta[i] + 0.5 * dt * k2t, omega[i] + 0.5 * dt * k2o)
    k4t, k4o = deriv(theta[i] + dt * k3t, omega[i] + dt * k3o)
    theta[i + 1] = theta[i] + (dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
    omega[i + 1] = omega[i] + (dt / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)

# Growing traces keyed to simulation time (not trim-path easy-ease).
trace_th = growing_trace(theta)
trace_om = growing_trace(omega)
theta_vs_t = growing_trace(theta)
omega_vs_t = growing_trace(omega)
frame_times = t.tolist()

# --- Vector field for the phase-plane quiver ---
gth = np.linspace(-3.4, 3.4, 17)
gom = np.linspace(-3.2, 3.2, 15)
TH, OM = np.meshgrid(gth, gom)
U, V = deriv(TH, OM)

THETA_COLOR = [120, 210, 255]   # angle
OMEGA_COLOR = [255, 130, 210]   # angular rate
TRAJ_COLOR = [255, 235, 150]

fig, axes = subplots(
    1, 2,
    comp_name="Pendulum Two Views",
    theme=THEME,
    wspace=0.24,
    plot_frame=True,
)

# --- Left: phase plane ---
phase = axes[0]
phase.quiver(
    TH, OM, U, V,
    normalize=True, c=True,                      # uniform arrows, color by speed
    gradient=([90, 120, 230], [255, 120, 215]),
    scale=0.20, width=3.0, headwidth=3.0, headlength=9.0,
    alpha=0.7, animate=1.2,
)
phase.plot_evolving(
    trace_th, trace_om,
    frame_times=frame_times,
    hold_keyframes=True,
    color=TRAJ_COLOR, linewidth=5,
    drop_shadow=True, fade_in=0.0, label="trajectory",
)
phase.set_title("Phase Plane").set_xlabel("theta").set_ylabel("omega")
phase.set_xlim(-3.4, 3.4).set_ylim(-3.2, 3.2)
phase.set_xticks().set_yticks().grid().add_legend()

# --- Right: same trajectory vs time ---
ts = axes[1]
ts.plot_evolving(
    t, theta_vs_t,
    frame_times=frame_times,
    hold_keyframes=True,
    color=THETA_COLOR, linewidth=5,
    drop_shadow=True, fade_in=0.0, label="theta (angle)",
)
ts.plot_evolving(
    t, omega_vs_t,
    frame_times=frame_times,
    hold_keyframes=True,
    color=OMEGA_COLOR, linewidth=5,
    drop_shadow=True, fade_in=0.0, label="omega (rate)",
)
ts.set_title("With Respect to Time").set_xlabel("t").set_ylabel("value")
ts.set_xticks().set_yticks().grid().add_legend()

fig.suptitle("Damped Pendulum: Two Views of One Motion")
fig.render()
