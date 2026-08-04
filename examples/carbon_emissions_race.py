"""A century of carbon: annual CO2 emissions of the biggest emitters, 1950->2022.

Each country is a horizontal bar (a special case of barh) whose length is its
annual CO2 emissions, keyframed across every year with ``barh_evolving``. Bars
are ordered by their final-year emissions, so you watch China's bar erupt from
almost nothing and rocket past a plateauing United States -- the single biggest
shift in the geography of carbon.

Data: Our World in Data (Global Carbon Project), annual CO2 in million tonnes.
    examples/data/co2_by_country.csv
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

THEME = "Ember Blueprint"

START, END, STEP = 1950, 2022, 2
years = np.arange(START, END + 1, STEP)

df = pd.read_csv("examples/data/co2_by_country.csv")
TOP = ["China", "United States", "India", "Russia", "Japan",
       "Iran", "Indonesia", "Germany", "Saudi Arabia", "South Korea"]
df = df[df["country"].isin(TOP)]
wide = (df.pivot(index="year", columns="country", values="co2")
          .reindex(range(START, END + 1)).interpolate().loc[years])

# Order bars by final-year emissions (biggest gets the top row).
order = wide.loc[END].sort_values().index.tolist()
co2_vmax = float(wide.max().max())   # shared color scale for every bar

race = (
    AEGraph(theme=THEME, comp_name="A Century of Carbon", fps=60,
            cinematic_effects=True, drop_shadow=True, show_all_points=True)
    .set_title("A Century of Carbon")
    .set_subtitle("annual CO\u2082 emissions of the top emitters, 1950 \u2192 2022")
    .set_xlabel("Annual CO\u2082 emissions (million tonnes)")
)

for y_pos, country in enumerate(order):
    width_frames = [np.array([wide.loc[yr, country]]) for yr in years]
    race.barh_evolving(
        np.array([y_pos]), width_frames,
        frame_duration=0.22,
        animate=1.6,
        bar_duration=0.6,
        hold_keyframes=False,
        # Fill color tracks annual CO₂ (same values as bar width), using the
        # theme's default object-color gradient and a shared 0..max scale.
        c_frames=width_frames,
        color_vmin=0.0,
        color_vmax=co2_vmax,
        anchor_at_y_axis=False,
        bar_height=0.74,
        drop_shadow=True,
    )

race.set_yticks(positions=list(range(len(order))), labels=order)
xmax_val = float(wide.max().max()) * 1.05
race.set_xlim(0, xmax_val)
race.set_ylim(-0.7, len(order) - 0.3)
race.set_xticks()

# Big bottom-right year ticker, matching barh_evolving's keyframe clock
# (which starts after the grow-in window: animate + k * frame_duration).
year_times = [1.6 + k * 0.22 for k in range(len(years))]
race.evolving_text(
    years.tolist(),
    location=(0, 0.6),
    frame_times=year_times,
    hold_keyframes=False,
    fontsize=130,
    horizontal_alignment="right",
    vertical_alignment="bottom",
    fade_in=0.5,
)

race.render()
