"""Signal-processing explainer: building a square wave from sine waves.

A perfect square wave is an infinite sum of odd harmonics:

    f(x) = (4/pi) * ( sin x + (1/3) sin 3x + (1/5) sin 5x + ... )

Each animation frame adds the *next* odd harmonic to the partial sum, so the
single smooth sine wave gradually sprouts ripples and squares itself off -- the
Fourier series assembling itself in real time. The leftover wiggles near the
edges are the Gibbs phenomenon, which this makes easy to point at.

The curve is one line whose *Path* is keyframed: frame k is the sum of the first
k harmonics. hold_keyframes=False morphs smoothly between partial sums; set it to
True to snap harmonic-by-harmonic like a slideshow.
"""

RENDERING_TIME = "MODERATE"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
from aegraph_config import config
import numpy as np

THEME = "Nordic Forest"

# --- Build the partial sums ---
N = 600
x = np.linspace(-np.pi, np.pi, N)
n_harmonics = 60                     # how many odd terms we build up to

frames = []
partial = np.zeros(N)
for k in range(n_harmonics):
    harmonic = 2 * k + 1             # 1, 3, 5, 7, ...
    partial = partial + (4.0 / np.pi) * np.sin(harmonic * x) / harmonic
    frames.append(partial.copy())

y_frames = np.array(frames)          # (n_harmonics, N)

# --- Build the evolving graph ---
fourier = (
    AEGraph(theme=THEME, comp_name="Fourier Square Wave",
            drop_shadow=True, cinematic_effects=True)
    .set_title("Building a Square Wave")
    .set_subtitle("each frame adds one more odd harmonic")
    .set_xlabel("x")
    .set_ylabel("sum of sines")
    .grid()
    .add_legend(legend_pos="top_right")
)

# Faint target square wave so viewers see what we're converging toward.
square = (4.0 / np.pi) * np.sign(np.sin(x))   # scaled to match amplitude
fourier.plot(x, np.clip(square, -1.0, 1.0), color=[130, 130, 140],
             linewidth=2, linestyle="dashed", animate=1, label="target")

fourier.plot_evolving(
    x, y_frames,
    frame_duration=0.55,             # ~7.7 s build-up
    hold_keyframes=False,            # smooth morph; True = step per harmonic
    color=config.object_color_1,
    label="partial sum",
    linewidth=5,
    drop_shadow=True,
)

fourier.set_xlim(-np.pi, np.pi).set_ylim(-1.4, 1.4).set_xticks().set_yticks()

fourier.render()
