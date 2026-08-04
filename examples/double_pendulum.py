"""A chaotic double pendulum, built entirely from evolving plot primitives.

The two rigid arms are ``plot_evolving`` lines whose endpoints move every frame
(arm 1: pivot -> bob 1, arm 2: bob 1 -> bob 2), and the masses at the bottom of
each arm are ``scatter_evolving`` bobs that ride along. A fading trace of the
lower bob draws the signature chaotic curve.

Physics: the standard double-pendulum equations of motion, integrated with RK4.
"""

RENDERING_TIME = "MODERATE"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

import numpy as np
from aegraph import AEGraph

THEME = "Duke Light"
L1, L2 = 1.0, 1.0          # arm lengths
M1, M2 = 1.0, 1.4          # bob masses
G = 9.81

ROD_COLOR = [225, 230, 245]
BOB1_COLOR = [120, 210, 255]
BOB2_COLOR = [255, 120, 200]


def derivs(state):
    """[theta1, theta2, omega1, omega2] -> their time derivatives."""
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


# --- Simulate ---
n_frames = 250
frame_dt = 0.02                 # seconds between rendered frames
substeps = 6                    # integration substeps per frame (stability)
dt = frame_dt / substeps

state = np.array([np.radians(125.0), np.radians(-10.0), 0.0, 0.0])
x1 = np.zeros(n_frames); y1 = np.zeros(n_frames)
x2 = np.zeros(n_frames); y2 = np.zeros(n_frames)
for f in range(n_frames):
    th1, th2 = state[0], state[1]
    x1[f] = L1 * np.sin(th1)
    y1[f] = -L1 * np.cos(th1)
    x2[f] = x1[f] + L2 * np.sin(th2)
    y2[f] = y1[f] - L2 * np.cos(th2)
    for _ in range(substeps):
        state = rk4_step(state, dt)

# --- Evolving rods: 2-point lines whose ends move each frame ---
arm1_x = np.column_stack([np.zeros(n_frames), x1])   # pivot -> bob 1
arm1_y = np.column_stack([np.zeros(n_frames), y1])
arm2_x = np.column_stack([x1, x2])                   # bob 1 -> bob 2
arm2_y = np.column_stack([y1, y2])

# --- Growing trace of the lower bob (future points held at the current bob) ---
trace_x = np.empty((n_frames, n_frames))
trace_y = np.empty((n_frames, n_frames))
for f in range(n_frames):
    trace_x[f, :f + 1] = x2[:f + 1]
    trace_x[f, f + 1:] = x2[f]
    trace_y[f, :f + 1] = y2[:f + 1]
    trace_y[f, f + 1:] = y2[f]

dp = AEGraph(
    theme=THEME, comp_name="Double Pendulum", fps=60,
    comp_width=1300, comp_height=1300,          # square so the arms stay rigid
    width=1000,height=1000,
    film_style=True,
)
dp.set_title("Double Pendulum").set_subtitle("deterministic chaos")

dp.film_style_parameters(
        roughen_border=0.1,                     # any numeric knob
        element_blur=True,                      # bool: toggle a whole piece on/off
        element_roughen_kinds={"bar": False},    # dict: per-kind override (merges into the existing map)
        edge_blur_margin=0.03,
        light_leak=True,                        # turn off any whole piece: paper/light_leak/temporal/vignette/edge_blur/zoom
        zoom_amount=3
    )

# Faint chaotic trace first (drawn behind everything else).
dp.plot_evolving(
    trace_x, trace_y, frame_duration=frame_dt, hold_keyframes=False,
    color=BOB2_COLOR, linewidth=2, fade_in=0.0,
)

# The two arms.
dp.plot_evolving(arm1_x, arm1_y, frame_duration=frame_dt, hold_keyframes=False,
                 color=ROD_COLOR, linewidth=7, drop_shadow=True, fade_in=0.0)
dp.plot_evolving(arm2_x, arm2_y, frame_duration=frame_dt, hold_keyframes=False,
                 color=ROD_COLOR, linewidth=7, drop_shadow=True, fade_in=0.0)

# The bobs at the bottom of each arm.
dp.scatter_evolving(x1.reshape(-1, 1), y1.reshape(-1, 1),
                    frame_duration=frame_dt, hold_keyframes=False,
                    color=BOB1_COLOR, radius=18, drop_shadow=True, fade_in=0.0)
dp.scatter_evolving(x2.reshape(-1, 1), y2.reshape(-1, 1),
                    frame_duration=frame_dt, hold_keyframes=False,
                    color=BOB2_COLOR, radius=24, drop_shadow=True, fade_in=0.0)

# Fixed pivot at the origin.
dp.scatter([0], [0], color=ROD_COLOR, radius=9)

reach = L1 + L2 + 0.25
dp.set_xlim(-reach, reach).set_ylim(-reach, reach)

dp.render()
