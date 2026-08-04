"""Population-over-time bar chart for two countries (Warm Atlas theme).

Reads the bundled CSV from examples/data/ (resolved relative to this file so it
works from any working directory). Data colors come from the theme.
"""

RENDERING_TIME = "VERY_SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
from aegraph_config import config
import pandas as pd
from pathlib import Path

THEME = "Warm Atlas"
DATA = Path(__file__).resolve().parent / "data" / "country_population.csv"

pops = pd.read_csv(DATA)
pops = (
    pops.drop(columns=["Country Code", "Indicator Name", "Indicator Code", "Unnamed: 69"])
    .set_index(pops.columns[0])
    .T
)



countries = (
    AEGraph(theme=THEME, comp_name="Indonesia vs. United States", fps=30,
            cinematic_effects=True)
    .set_title("Indonesia vs. United States")
    .set_xlabel("Year")
    .set_ylabel("Population (Millions)")
    .grid()
    .add_legend()
)

series_colors = {
    "United States": config.object_color_2,
    "Indonesia": config.object_color_1,
}
for country, color in series_colors.items():
    countries.bar_graph(
        pops.index.astype(int),
        pops[country].astype(int) / 1000000,
        color=color,
        label=country,
        alpha=0.75,
        animate=4,
        meta_easy_ease=True,
        meta_ease_speed=0,
        meta_ease_influence=50
    )
countries.set_xticks().set_yticks()

countries.render()
