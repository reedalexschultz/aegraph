"""China's population pyramid evolving from 2026 -> 2050 -> 2100.

Each census year is a dataset. Every age band is a real bar (male on the left,
female on the right) that animates in individually and then morphs its width
from one year to the next via ``add_population_pyramid_evolving``.
"""

RENDERING_TIME = "VERY_SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np
import pandas as pd

years = [2026, 2050, 2100]
files = [f"examples/data/China-{y}.csv" for y in years]

# --- Load every year (same age bands, same order) ---
ages = None
male_frames = []
female_frames = []
for file in files:
    df = pd.read_csv(file)
    if ages is None:
        ages = df["Age"].astype(str).tolist()
    male_frames.append(df["M"].astype(float).values)
    female_frames.append(df["F"].astype(float).values)

y_positions = np.arange(len(ages))

# Hold each census year for a beat, then morph to the next.
frame_times = [2.0, 6.0, 10.0]

(
AEGraph(
    width=1000,
    height=1500,
    compwidth=1500,
    compheight=1920,
    drop_shadow=True,
    comp_name="China Population Pyramid 2026-2100",
    fps=60,
    show_all_points=True,
    animate_opacity=False,
    animate_axes=False,

    # same cool background vibe
    bg_color=[25, 25, 25],
    ui_color=[212, 218, 223],
    full_bg=True,
    cinematic_effects=True,
    distress_texture=1,

    ease_speed=0,
    ease_influence=60,
    font_scale=1,
)
.add_population_pyramid_evolving(
    ages=ages,
    male_frames=male_frames,
    female_frames=female_frames,
    mode="percent",
    frame_times=frame_times,
    hold_keyframes=False,   # smooth morph; True = snap between years
    animate=2.0,            # grow-in window (settles by frame_times[0])
    bar_duration=0.5,
    animate_downward=True,
    drop_shadow=True,

    # softer pastel blue/red
    color_male=[25, 120, 214],
    color_female=[229, 70, 101],

    show_grid=False,
)
# Year ticker that counts up as the pyramid morphs (synced to frame_times).
.evolving_text(
    years,
    location=(0, len(ages) - 1),
    frame_times=frame_times,
    hold_keyframes=False,
    fontsize=120,
    horizontal_alignment="left",
    vertical_alignment="top",
    fade_in=0.6,
)
.set_xlabel("China Population Pyramid: 2026 \u2192 2050 \u2192 2100")
.set_yticks(positions=y_positions, labels=ages)
.render()
)
