"""A double pendulum and its *living* phase field, side by side and time-locked.

Right panel: the physical double pendulum (two rigid arms + bobs), built from
evolving plot primitives.

Left panel: the phase plane of the lower bob -- angle (theta2) vs angular rate
(omega2). Instead of a pre-drawn curve, it's a **quiver field that breathes**:
at every instant the arrows show the local flow  (d theta2/dt, d omega2/dt)  of
the equations of motion, evaluated with the *live* values of the other two
state variables (theta1, omega1) at that same frame. So the field itself
evolves as the pendulum swings, and the bright dot is the system's actual state
tracing its path through that changing field.

Both panels share one frame clock, so the phase field and the pendulum advance
in lockstep -- two synchronized views of the same chaotic motion.

Physics: the standard double-pendulum equations of motion, integrated with RK4.
"""

RENDERING_TIME = "LONG"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

import numpy as np
from aegraph import subplots

THEME = "Midnight Neon"
L1, L2 = 1.0, 1.0          # arm lengths
M1, M2 = 1.0, 0.5          # bob masses
G = 9.81

ROD_COLOR = [225, 230, 245]
BOB1_COLOR = [120, 210, 255]
BOB2_COLOR = [255, 120, 200]
TRACE_COLOR = [255, 235, 150]
FIELD_LOW = [70, 90, 180]    # slow flow
FIELD_HIGH = [255, 120, 215]  # fast flow


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


def omega2_dot(th1, th2, w1, w2):
    """Vectorized d(omega2)/dt over a (theta2, omega2) grid for live (th1, w1)."""
    d = th2 - th1
    cd, sd = np.cos(d), np.sin(d)
    den1 = (M1 + M2) * L1 - M2 * L1 * cd * cd
    den2 = (L2 / L1) * den1
    return (-M2 * L2 * w2 * w2 * sd * cd
            + (M1 + M2) * G * np.sin(th1) * cd
            - (M1 + M2) * L1 * w1 * w1 * sd
            - (M1 + M2) * G * np.sin(th2)) / den2


# --- Simulate the trajectory ---
n_frames = 200
frame_dt = 0.045                # seconds between rendered frames
substeps = 6
dt = frame_dt / substeps
T_total = (n_frames - 1) * frame_dt

state = np.array([np.radians(85.0), np.radians(5.0), 0.0, 0.0])
th1 = np.zeros(n_frames); th2 = np.zeros(n_frames)
w1 = np.zeros(n_frames);  w2 = np.zeros(n_frames)
for f in range(n_frames):
    th1[f], th2[f], w1[f], w2[f] = state
    for _ in range(substeps):
        state = rk4_step(state, dt)

# Cartesian positions of the two bobs (for the physical view).
x1 = L1 * np.sin(th1); y1 = -L1 * np.cos(th1)
x2 = x1 + L2 * np.sin(th2); y2 = y1 - L2 * np.cos(th2)

# --- Phase-plane grid (theta2 vs omega2), sized to the trajectory ---
def padded(lo, hi, pad=0.18):
    span = hi - lo
    return lo - pad * span, hi + pad * span

th2_lo, th2_hi = padded(th2.min(), th2.max())
w2_lo, w2_hi = padded(w2.min(), w2.max())
gth = np.linspace(th2_lo, th2_hi, 13)
gom = np.linspace(w2_lo, w2_hi, 11)
TH, OM = np.meshgrid(gth, gom)

# The field "breathes": sample the live (theta1, omega1) at a coarser set of
# keyframe times that span the SAME wall-clock window as the pendulum, so the
# two panels stay in lockstep while keeping the arrow count manageable.
n_qframes = 48
q_idx = np.linspace(0, n_frames - 1, n_qframes).round().astype(int)
q_times = np.linspace(0.0, T_total, n_qframes)

# d(theta2)/dt = omega2 (constant in time); d(omega2)/dt evolves with the state.
U_frames = np.tile(OM.ravel(), (n_qframes, 1))
V_frames = np.empty((n_qframes, OM.size))
for k, fi in enumerate(q_idx):
    V_frames[k] = omega2_dot(th1[fi], TH, w1[fi], OM).ravel()
speed_frames = np.hypot(U_frames, V_frames)

# Growing trace of the phase point (future samples pinned at the current state).
trace_th = np.empty((n_frames, n_frames))
trace_om = np.empty((n_frames, n_frames))
for f in range(n_frames):
    trace_th[f, :f + 1] = th2[:f + 1]; trace_th[f, f + 1:] = th2[f]
    trace_om[f, :f + 1] = w2[:f + 1];  trace_om[f, f + 1:] = w2[f]

# --- Figure: two synchronized panels in one comp ---
fig, axes = subplots(
    1, 2,
    comp_name="Double Pendulum Phase Field",
    theme=THEME, fps=60,
    comp_width=1920, comp_height=1080,
    wspace=0.22, hspace=0.30,
    drop_shadow=True, cinematic_effects=True,
    plot_frame=True,
)
phase, pend = axes

# --- Left: the living phase field ---
phase.quiver_evolving(
    TH, OM, U_frames, V_frames,
    normalize=True, scale=0.032, scale_mode="comp",     # uniform direction arrows
    c_frames=speed_frames, gradient=(FIELD_LOW, FIELD_HIGH),
    width=3.0, headwidth=3.0, headlength=10.0,
    alpha=0.6, frame_times=q_times.tolist(),
    hold_keyframes=False, fade_in=0.6,
)
phase.plot_evolving(trace_th, trace_om, frame_duration=frame_dt,
                    hold_keyframes=False, color=TRACE_COLOR, linewidth=4,
                    fade_in=0.0)
phase.scatter_evolving(th2.reshape(-1, 1), w2.reshape(-1, 1),
                       frame_duration=frame_dt, hold_keyframes=False,
                       color=BOB2_COLOR, radius=14, drop_shadow=True, fade_in=0.0)
phase.set_title("Phase Field of the Lower Bob")
phase.set_xlabel("theta2 (angle)").set_ylabel("omega2 (rate)")
phase.set_xlim(th2_lo, th2_hi).set_ylim(w2_lo, w2_hi)
phase.set_xticks().set_yticks().grid()

# --- Right: the physical double pendulum, same frame clock ---
arm1_x = np.column_stack([np.zeros(n_frames), x1])   # pivot -> bob 1
arm1_y = np.column_stack([np.zeros(n_frames), y1])
arm2_x = np.column_stack([x1, x2])                   # bob 1 -> bob 2
arm2_y = np.column_stack([y1, y2])

trace2_x = np.empty((n_frames, n_frames)); trace2_y = np.empty((n_frames, n_frames))
for f in range(n_frames):
    trace2_x[f, :f + 1] = x2[:f + 1]; trace2_x[f, f + 1:] = x2[f]
    trace2_y[f, :f + 1] = y2[:f + 1]; trace2_y[f, f + 1:] = y2[f]

pend.plot_evolving(trace2_x, trace2_y, frame_duration=frame_dt,
                   hold_keyframes=False, color=BOB2_COLOR, linewidth=2, fade_in=0.0)
pend.plot_evolving(arm1_x, arm1_y, frame_duration=frame_dt, hold_keyframes=False,
                   color=ROD_COLOR, linewidth=7, drop_shadow=True, fade_in=0.0)
pend.plot_evolving(arm2_x, arm2_y, frame_duration=frame_dt, hold_keyframes=False,
                   color=ROD_COLOR, linewidth=7, drop_shadow=True, fade_in=0.0)
pend.scatter_evolving(x1.reshape(-1, 1), y1.reshape(-1, 1), frame_duration=frame_dt,
                      hold_keyframes=False, color=BOB1_COLOR, radius=18,
                      drop_shadow=True, fade_in=0.0)
pend.scatter_evolving(x2.reshape(-1, 1), y2.reshape(-1, 1), frame_duration=frame_dt,
                      hold_keyframes=False, color=BOB2_COLOR, radius=24,
                      drop_shadow=True, fade_in=0.0)
pend.scatter([0], [0], color=ROD_COLOR, radius=9)
reach = L1 + L2 + 0.25
pend.set_title("The Pendulum")
pend.set_xlim(-reach, reach).set_ylim(-reach, reach)

fig.suptitle("Double Pendulum: A Living Phase Field")
fig.render()
