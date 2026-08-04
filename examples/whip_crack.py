"""Crack the whip: an elastic string as a tapered mass-spring chain.

We model a whip as N point masses joined by springs (a discretized elastic
string). The handle is heavy, the tip is light -- the mass tapers exponentially
along the lash. Newton's second law for each bead,

    m_i * y_i'' = K * (y_{i-1} - 2 y_i + y_{i+1}) - gamma * y_i'

is the damped wave equation with a position-dependent wave speed c_i = sqrt(K/m_i)*dx.
Because the tip is light, the pulse you inject at the handle accelerates as it
travels outward, dumping the whole string's energy into the last few grams of
lash -- the crack. We integrate the ODE system (semi-implicit Euler), then
animate the whip's shape together with beads that swell and glow with local
speed, so the energy concentrating at the tip is impossible to miss.
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

# --- Whip / chain parameters ---
N = 90                    # beads along the lash
L = 10.0                  # whip length
dx = L / (N - 1)
x = np.linspace(0.0, L, N)

alpha = 4.0               # mass taper (bigger = lighter, faster tip)
mass = np.exp(-alpha * np.arange(N) / (N - 1))
K = 120.0                 # spring coupling stiffness
gamma = 0.30              # light damping

Tf = 0.40                 # flick duration (a single handle snap)
Amp = 1.0

# --- Integrate the ODE system ---
dt = 0.0006
T_total = 7.5
nsteps = int(T_total / dt)
n_frames = 120
capture_every = max(1, nsteps // n_frames)

y = np.zeros(N)
v = np.zeros(N)
y_frames, speed_frames = [], []
t = 0.0
for s in range(nsteps):
    y[0] = Amp * np.sin(np.pi * t / Tf) if t < Tf else 0.0   # handle snap
    lap = np.zeros(N)
    lap[1:-1] = y[:-2] - 2 * y[1:-1] + y[2:]
    lap[-1] = y[-2] - y[-1]            # free tip (single spring)
    a = K * lap / mass - gamma * v
    a[0] = 0.0                          # handle driven kinematically
    v += a * dt
    v[0] = 0.0
    y += v * dt
    t += dt
    if s % capture_every == 0:
        y_frames.append(y.copy())
        speed_frames.append(np.abs(v.copy()))

y_frames = np.array(y_frames)
speed_frames = np.array(speed_frames)
n_frames = len(y_frames)
smax = speed_frames.max()

# Beads on a subsample so the "elastic" reads clearly without clutter.
idx = np.arange(0, N, 3)
bead_y = y_frames[:, idx]
bead_x = np.tile(x[idx], (n_frames, 1))
bead_speed = speed_frames[:, idx]
bead_radius = 3.0 + 13.0 * bead_speed / smax

whip = (
    AEGraph(theme=THEME, comp_name="Crack the Whip", fps=60,
            drop_shadow=True, cinematic_effects=True,
            xaxis_location=0.0, yaxis_location="left", plot_frame=True)
    .set_title("Crack the Whip")
    .set_subtitle("a tapered elastic string races its energy into the tip")
    .set_xlabel("Position along the whip")
    .set_ylabel("Displacement")
    .grid()
)

# At-rest line underneath.
whip.plot(x, np.zeros_like(x), color=[110, 120, 140], linestyle="dashed",
          linewidth=2, animate=0.01)

# The whip's shape, morphing frame to frame.
whip.plot_evolving(
    x, list(y_frames),
    frame_duration=0.07, hold_keyframes=False,
    color=[255, 110, 80], linewidth=5, drop_shadow=True, fade_in=0.0,
)

# Glowing beads: size & color follow local speed (the tip lights up on the crack).
whip.scatter_evolving(
    bead_x, bead_y,
    radius_frames=bead_radius,
    c_frames=bead_speed,
    gradient=([90, 150, 255], [255, 230, 120]),   # slow=cool -> fast=hot
    frame_duration=0.07,
    hold_keyframes=False,
    alpha=0.95,
    fade_in=0.0,
    drop_shadow=True,
)

whip.set_xlim(0.0, L).set_ylim(-3.4, 3.4)
whip.set_xticks().set_yticks()

whip.render()
