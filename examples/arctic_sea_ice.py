"""The vanishing ice: the Arctic's summer sea-ice minimum since 1979.

Every September the Arctic Ocean reaches its yearly minimum sea-ice extent.
Satellites have watched that minimum collapse from ~7 million km2 in 1979 to
under 5 million today. The line draws across the decades against the 1979
baseline so the loss is impossible to miss.

Data: NSIDC Sea Ice Index v4, September monthly extent.
    examples/data/arctic_sea_ice.csv
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

THEME = "Glacier Ink"

df = pd.read_csv("examples/data/arctic_sea_ice.csv").sort_values("year")
year = df["year"].to_numpy(dtype=float)
extent = df["extent"].to_numpy(dtype=float)

baseline = float(extent[0])
decline_pct = (extent[-1] - baseline) / baseline * 100.0

# Linear trend line for context.
slope, intercept = np.polyfit(year, extent, 1)
trend = slope * year + intercept

ice = (
    AEGraph(theme=THEME, comp_name="The Vanishing Ice", fps=60, cinematic_effects=True,
            xaxis_location="bottom", yaxis_location="left", plot_frame=True)
    .set_title("The Vanishing Ice")
    .set_subtitle("Arctic sea-ice minimum (September), 1979 \u2192 2025")
    .set_xlabel("Year")
    .set_ylabel("Sea-ice extent (million km\u00b2)")
    .grid()
)

# 1979 baseline + trend underneath, then the data line on top.
ice.plot(year, np.full_like(year, baseline), color=[150, 170, 195],
         linestyle="dashed", linewidth=2, animate=0.01)
ice.plot(year, trend, color=[120, 140, 170], linestyle="dotted", linewidth=2, animate=0.01)
ice.plot(year, extent, color=[90, 200, 235], linewidth=7, animate=6.0)

ice.annotate(f"1979 baseline \u2248 {baseline:.1f}", x=year[1], y=baseline + 0.18, fontsize=20)
ice.annotate(f"{decline_pct:.0f}% since 1979", x=year[-1] - 16, y=extent[-1] - 0.5, fontsize=24)

ice.set_xlim(year.min(), year.max())
ice.set_xticks().set_yticks()

ice.render()
