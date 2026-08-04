"""Sensitive dependence on initial conditions, the classic way: a fan of double
pendulums released from almost-identical angles.

Every pendulum starts within a few thousandths of a radian of its neighbor, so
they track together at first and then diverge into completely different motions
-- the hallmark of deterministic chaos. Each pendulum is one evolving 3-point
line (pivot -> bob 1 -> bob 2) tinted along a rainbow gradient, and all the
lower bobs ride along as a single evolving scatter.
"""

RENDERING_TIME = "VERY_LONG"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
import colorsys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

import numpy as np
from aegraph import AEGraph

THEME = "Midnight Neon"
N = 10                      # number of pendulums in the ensemble
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.4
G = 9.81


def gradient_colors(n):
    """A vivid rainbow sweep as [r, g, b] triplets (0-255)."""
    cols = []
    for i in range(n):
        h = 0.58 + 0.42 * (i / max(n - 1, 1))      # cyan -> blue -> magenta
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.85, 1.0)
        cols.append([int(r * 255), int(g * 255), int(b * 255)])
    return cols


def derivs(state):
    """Vectorized over the ensemble: state is (4, N)."""
    th1, th2, w1, w2 = state
    d = th2 - th1
    cd, sd = np.cos(d), np.sin(d)
    den1 = (M1 + M2) * L1 - M2 * L1 * cd * cd
    dw1 = (M2 * L1 * w1 * w1 * sd * cd
           + M2 * G * np.sin(th2) * cd
           + M2 * L2 * w2 * w2 * sd
           - (M1 + M2) * G * np.sin(th1)) / den1
    den2 = (L2 / L1) * den1
    dw2 = (-M2 * L2 * w2 * w2 * sd * cd
           + (M1 + M2) * G * np.sin(th1) * cd
           - (M1 + M2) * L1 * w1 * w1 * sd
           - (M1 + M2) * G * np.sin(th2)) / den2
    return np.array([w1, w2, dw1, dw2])


def rk4_step(s, dt):
    k1 = derivs(s)
    k2 = derivs(s + 0.5 * dt * k1)
    k3 = derivs(s + 0.5 * dt * k2)
    k4 = derivs(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# --- Simulate the whole ensemble at once ---
n_frames = 300
frame_dt = 0.04
substeps = 6
dt = frame_dt / substeps

# Tiny spread of starting angles -> identical-looking start, chaotic divergence.
theta1_0 = np.radians(130.0) + np.linspace(0.0, 0.06, N)
theta2_0 = np.radians(130.0) * np.ones(N)
state = np.array([theta1_0, theta2_0, np.zeros(N), np.zeros(N)])

x1_all = np.zeros((n_frames, N)); y1_all = np.zeros((n_frames, N))
x2_all = np.zeros((n_frames, N)); y2_all = np.zeros((n_frames, N))
for f in range(n_frames):
    th1, th2 = state[0], state[1]
    x1_all[f] = L1 * np.sin(th1)
    y1_all[f] = -L1 * np.cos(th1)
    x2_all[f] = x1_all[f] + L2 * np.sin(th2)
    y2_all[f] = y1_all[f] - L2 * np.cos(th2)
    for _ in range(substeps):
        state = rk4_step(state, dt)

colors = gradient_colors(N)

dp = AEGraph(
    theme=THEME, comp_name="Double Pendulum Chaos", fps=60,
    comp_width=1080, comp_height=1080,          # square so arms stay rigid
    drop_shadow=True, cinematic_effects=True,
)
dp.set_title("Double Pendulum Chaos").set_subtitle("a few thousandths of a radian apart")

# Each pendulum: one evolving articulated line pivot -> bob 1 -> bob 2.
for i in range(N):
    poly_x = np.column_stack([np.zeros(n_frames), x1_all[:, i], x2_all[:, i]])
    poly_y = np.column_stack([np.zeros(n_frames), y1_all[:, i], y2_all[:, i]])
    dp.plot_evolving(poly_x, poly_y, frame_duration=frame_dt,
                     hold_keyframes=False, color=colors[i], linewidth=4,
                     fade_in=0.0)

# All lower bobs in one evolving scatter, each tinted to match its pendulum.
color_frames = [list(colors) for _ in range(n_frames)]
dp.scatter_evolving(x2_all, y2_all, color_frames=color_frames,
                    frame_duration=frame_dt, hold_keyframes=False,
                    radius=12, drop_shadow=True, fade_in=0.0)

# Fixed pivot.
dp.scatter([0], [0], color=[230, 235, 250], radius=9)

reach = L1 + L2 + 0.25
dp.set_xlim(-reach, reach).set_ylim(-reach, reach)

dp.render()
