"""Health & wealth of nations -- the real Gapminder bubble race, 1952 -> 2007.

Hans Rosling's famous animation, rebuilt from the actual Gapminder data: every
country is a bubble drifting up and to the right as it grows richer and lives
longer. Each continent is its own evolving-scatter layer (so the legend reads as
a continent key) and they all share one frame clock.

    x    = GDP per capita (log scale)   -> Position keyframes
    y    = life expectancy (years)      -> Position keyframes
    size = population                   -> Ellipse-Size keyframes
    color= continent

Data: Gapminder (via plotly datasets), 142 countries, 5-year steps.
    examples/data/gapminder.csv
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

THEME = "Slate Report"

CONTINENT_COLORS = {
    "Africa":   [240, 130, 60],
    "Asia":     [235, 80, 95],
    "Americas": [120, 205, 110],
    "Europe":   [110, 160, 245],
    "Oceania":  [190, 130, 240],
}

df = pd.read_csv("examples/data/gapminder.csv")
years = np.sort(df["year"].unique())
n_frames = len(years)
pop_max = df["pop"].max()

world = (
    AEGraph(theme=THEME, comp_name="Health and Wealth of Nations",
            cinematic_effects=True, xaxis_location="bottom", yaxis_location="left", plot_frame=True)
    .set_title("Health & Wealth of Nations")
    .set_subtitle("every country, 1952 \u2192 2007 (real Gapminder data)")
    .set_xlabel("GDP per capita ($/yr, log scale)")
    .set_ylabel("Life expectancy (years)")
    .set_xscale("log")
    .grid()
)

for continent, color in CONTINENT_COLORS.items():
    sub = df[df["continent"] == continent]
    countries = np.sort(sub["country"].unique())
    # Pivot to (n_frames, n_countries) matrices aligned on the shared year grid.
    gdp = sub.pivot(index="year", columns="country", values="gdpPercap").loc[years, countries].to_numpy()
    life = sub.pivot(index="year", columns="country", values="lifeExp").loc[years, countries].to_numpy()
    pop = sub.pivot(index="year", columns="country", values="pop").loc[years, countries].to_numpy()
    radius = 6.0 + 42.0 * np.sqrt(pop / pop_max)

    world.scatter_evolving(
        gdp, life,
        radius_frames=radius,
        color=color,
        label=continent,
        frame_duration=0.7,        # ~8.5 s total
        hold_keyframes=True,
        alpha=0.82,
        drop_shadow=False,
        fade_in=0.4,
    )

world.set_xlim(200, 120000).set_ylim(20, 90)
world.set_xticks().set_yticks()
world.add_legend(legend_pos="bottom_right")

# A year ticker that counts up in sync with the bubbles (matches frame_duration).
world.evolving_text(
    years,
    location=(0, 86),
    frame_duration=0.7,
    hold_keyframes=True,   # count smoothly through every year
    fontsize=84,
    horizontal_alignment="left",
    vertical_alignment="top",
    fade_in=0.4,
    outline = True
)

world.render()
