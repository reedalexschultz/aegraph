"""Who emits the world's CO2? -- top emitters' share of 2023 emissions.

A pie of the largest national CO2 emitters in 2023, with everyone else folded
into a single "Rest of world" wedge. Uses the absolute-pixel gap, rounded
corners, and color-matched leader lines.

Data: Global Carbon Project (via Our World in Data), territorial CO2 (Mt).
    examples/data/co2_by_country.csv
"""

RENDERING_TIME = "SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import pandas as pd

df = pd.read_csv("examples/data/co2_by_country.csv")
latest = df[df["year"] == df["year"].max()]

countries = ["China", "United States", "India", "Russia", "Japan", "Iran", "Germany"]
emissions = latest.set_index("country")["co2"]
world_total = float(emissions["World"])

values = [float(emissions[c]) for c in countries]
labels = list(countries)
# Everyone not broken out individually.
values.append(world_total - sum(values))
labels.append("Rest of world")

chart = (
    AEGraph(theme="Ember Blueprint", comp_name="Global CO2 Emissions 2023",
            cinematic_effects=True)
    .set_title("Who Emits the World's CO\u2082?")
    .set_subtitle("share of global emissions, 2023")
)

chart.pie(
    values,
    labels=labels,
    show_percent=True,
    animate=1.6,
    gap=14.0,
    roundness=6.0,
    leader_lines=True,
    donut = 0.5
)

chart.render()
