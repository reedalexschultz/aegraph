"""Evolving heatmap: an infection spreading across a grid from a single cell.

A simple SIR cellular automaton. Every cell is one of three states:

    S (susceptible)  -- healthy, can catch the infection
    I (infected)     -- contagious for a few steps, then recovers
    R (recovered)    -- immune afterwards

Each step, a susceptible cell catches the infection from any infected
4-neighbor with probability ``p_infect``; infected cells stay contagious for
``infect_dur`` steps and then turn to R. We start with a SINGLE infected cell in
the middle, so you watch one cell seed an expanding wave of infection that
leaves a recovered core behind it.

For coloring we map the state to a scalar (S -> 0, R -> 0.4, I -> 1.0) and feed
the stack to ``heatmap_evolving``, so the contagious ring glows hot while the
recovered interior settles to a cooler tone.
"""

RENDERING_TIME = "LONG"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np

THEME = "Toxic Mint"

# --- Model parameters ---
N = 20                       # grid is N x N cells
p_infect = 0.30              # chance an infected neighbor infects a cell / step
infect_dur = 6               # steps a cell stays infectious before recovering
n_frames = 130
rng = np.random.default_rng(2)

S, I, R = 0, 1, 2
state = np.zeros((N, N), dtype=int)         # everyone susceptible...
timer = np.zeros((N, N), dtype=int)         # remaining infectious steps
state[N // 2, N // 2] = I                    # ...except one seed in the center
timer[N // 2, N // 2] = infect_dur

# Map a state grid to the scalar field the heatmap colors.
VALUE = {S: 0.0, R: 0.4, I: 1.0}


def to_value(s):
    v = np.zeros_like(s, dtype=float)
    v[s == R] = 0.4
    v[s == I] = 1.0
    return v


frames = [to_value(state)]
for _ in range(n_frames - 1):
    infected = state == I
    # Count infected 4-neighbors for every cell (no wraparound).
    neighbors = np.zeros((N, N), dtype=int)
    neighbors[1:, :] += infected[:-1, :]
    neighbors[:-1, :] += infected[1:, :]
    neighbors[:, 1:] += infected[:, :-1]
    neighbors[:, :-1] += infected[:, 1:]

    # Susceptible cells with >=1 infected neighbor may catch it.
    exposure = 1.0 - (1.0 - p_infect) ** neighbors      # P(infected this step)
    catches = (state == S) & (rng.random((N, N)) < exposure)

    # Advance existing infections; recover when the timer runs out.
    timer[infected] -= 1
    recovered = infected & (timer <= 0)
    state[recovered] = R

    # Apply new infections.
    state[catches] = I
    timer[catches] = infect_dur

    frames.append(to_value(state))

frames = np.array(frames)        # shape (n_frames, N, N)

# --- Build the evolving heatmap ---
epi = (
    AEGraph(theme=THEME, comp_name="Infection Spread", fps=60,
            cinematic_effects=True, plot_frame=True,
            xaxis_location="bottom", yaxis_location="left")
    .set_title("Infection Spreading")
    .set_subtitle("one cell seeds an expanding SIR wave")
)
epi.heatmap_evolving(
    frames,
    gradient=([20, 28, 44], [255, 58, 92]),    # cool susceptible -> hot infected
    vmin=0.0, vmax=1.0,
    frame_duration=0.06,
    colorbar_label="infection state",
    fade_in=0.4,
)
epi.set_xticks().set_yticks()

epi.render()
