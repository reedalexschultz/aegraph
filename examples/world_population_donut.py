"""World population by continent (2007) -- a donut.

A clean five-slice donut of how the world's people are distributed across
continents, using the real Gapminder snapshot. Demonstrates the donut hole,
constant-width pixel gaps, and color-matched leader lines.

Data: Gapminder (142 countries), year 2007.
    examples/data/gapminder.csv
"""

RENDERING_TIME = "SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import pandas as pd

df = pd.read_csv("examples/data/gapminder.csv")
latest = df[df["year"] == df["year"].max()]

pop = latest.groupby("continent")["pop"].sum().sort_values(ascending=False)
labels = list(pop.index)
values = [v / 1e6 for v in pop.values]  # millions of people

chart = (
    AEGraph(theme="Nordic Forest", comp_name="World Population by Continent",
            cinematic_effects=True)
    .set_title("Where the World Lives")
    .set_subtitle("population by continent, 2007")
)

chart.pie(
    values,
    labels=labels,
    donut=0.55,
    show_percent=True,
    animate=1.6,
    gap=16.0,
    roundness=10.0,
    leader_lines=True,
)

chart.render()
