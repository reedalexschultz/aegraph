"""Half a century of population shift -- two donuts in one comp.

The world's population didn't just grow between 1952 and 2007, it
redistributed. Two side-by-side donuts make the shift toward Asia obvious.
Because both panels live in one AEFigure, they share a single background (and a
single grunge texture), not one per panel.

Data: Gapminder (142 countries).
    examples/data/gapminder.csv
"""

RENDERING_TIME = "MODERATE"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import subplots
import pandas as pd

df = pd.read_csv("examples/data/gapminder.csv")

# Keep the slice order identical across both donuts so colors stay consistent.
order = list(df.groupby("continent")["pop"].sum().sort_values(ascending=False).index)


def shares(year):
    pop = df[df["year"] == year].groupby("continent")["pop"].sum()
    return [pop[c] / 1e6 for c in order]


fig, axes = subplots(1, 2, comp_name="Population Shift 1952 to 2007",
                     theme="Slate Report", wspace=0.22,
                     suptitle="Half a Century of Population Shift")

for ax, year in zip(axes, (1952, 2007)):
    ax.pie(
        shares(year),
        labels=order,
        donut=0.55,
        show_percent=True,
        animate=1.4,
        gap=14.0,
        roundness=9.0,
        leader_lines=True,
    )
    ax.set_title(str(year))

fig.render()
