"""Small multiples: one line panel per region, on shared axes.

``AEFigure.facet`` splits a tidy DataFrame by a column and lays out one panel
per group. With ``sharex``/``sharey`` every panel uses the same limits so the
facets are directly comparable.
"""

RENDERING_TIME = "VERY_SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

import numpy as np
import pandas as pd
from aegraph import AEFigure

rng = np.random.default_rng(7)
regions = ["North", "South", "East", "West", "Central", "Coastal"]
rows = []
for r in regions:
    level = 8 + (hash(r) % 5)
    trend = rng.uniform(-0.4, 0.6)
    for month in range(12):
        rows.append({
            "region": r,
            "month": month,
            "sales": level + trend * month + rng.normal(0, 0.8),
        })
df = pd.DataFrame(rows)

fig = AEFigure(comp_name="Sales Small Multiples", theme="Minimal Frost",
               wspace=0.28, hspace=0.40)
fig.facet(df, by="region", x="month", y="sales", kind="line",
          sharex=True, sharey=True, color="blue", animate=2.0)
fig.suptitle("Monthly Sales by Region")
fig.render()
