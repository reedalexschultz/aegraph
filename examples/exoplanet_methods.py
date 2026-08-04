"""How we find exoplanets -- discovery methods as a donut.

Nearly every confirmed planet beyond our solar system was found one of a few
ways. This donut shows the breakdown, folding the rare techniques into a single
"Other" slice. Demonstrates grouping a long tail plus the donut styling.

Data: NASA Exoplanet Archive (confirmed planets sample).
    examples/data/exoplanets.csv
"""

RENDERING_TIME = "SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import pandas as pd

df = pd.read_csv("examples/data/exoplanets.csv")
counts = df["discoverymethod"].value_counts()

# Keep the big methods; everything rarer collapses into "Other".
TOP = 4
top = counts.head(TOP)
labels = list(top.index)
values = list(top.values)
other = int(counts[TOP:].sum())
if other:
    labels.append("Other")
    values.append(other)

chart = (
    AEGraph(theme="Deep Ocean", comp_name="How We Find Exoplanets",
            cinematic_effects=True)
    .set_title("How We Find Exoplanets")
    .set_subtitle("confirmed planets by discovery method")
)

chart.pie(
    values,
    labels=labels,
    donut=0.5,
    show_percent=True,
    animate=1.6,
    gap=14.0,
    roundness=8.0,
    leader_lines=True,
)

chart.render()
