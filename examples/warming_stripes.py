"""A warming world: every year since 1880 as a bar, colored by how hot it was.

Each bar is one year's global surface-temperature anomaly relative to the
1951-1980 average. Bars below the line (cool years, early record) are blue and
fall downward; bars above (the modern surge) are red and climb. Watching them
animate in left-to-right traces the unmistakable hockey-stick of warming.

Data: NASA GISS GISTEMP v4, global land-ocean temperature index (deg C).
    examples/data/global_temp_anomaly.csv
"""

RENDERING_TIME = "SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph

import numpy as np
import pandas as pd

THEME = "Midnight Neon"

COLD = [45, 105, 215]    # deep blue for the coolest years
HOT = [220, 45, 45]      # hot red for the warmest years

df = pd.read_csv("examples/data/global_temp_anomaly.csv")
year = df["year"].to_numpy(dtype=float)
anomaly = df["anomaly_c"].to_numpy(dtype=float)

warming = (
    AEGraph(theme=THEME, comp_name="A Warming World", fps=60,
            drop_shadow=False, cinematic_effects=True,
            xaxis_location=0.0, yaxis_location="left",
            plot_frame=True)
    .set_title("A Warming World")
    .set_subtitle("Global temperature anomaly vs the 1951\u20131980 average")
    .set_xlabel("Year")
    .set_ylabel("Temperature anomaly (\u00b0C)")
    .grid()
)

warming.bar_graph(
    year, anomaly,
    bar_width=1.0,
    c=anomaly,                       # color each bar by its own warmth
    gradient=(COLD, HOT),
    animate=6.5,
    bar_duration=0.28,
    drop_shadow=False,
    alpha=0.95,
    meta_easy_ease=True,
    meta_ease_speed=0,
    meta_ease_influence=30
)

warming.set_xlim(year.min() - 1, year.max() + 1)
warming.set_ylim(anomaly.min() - 0.1, anomaly.max() + 0.15)
warming.set_xticks().set_yticks()

warming.render()
