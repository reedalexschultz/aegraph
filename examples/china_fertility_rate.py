RENDERING_TIME = "VERY_SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

print("--------------")
print(os.environ["AEGRAPH_PATH"])
from aegraph import AEGraph

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("examples/data/China Fertility Rate over Time.csv")

df = df.sort_values("Year")

x = df["Year"].astype(int).tolist()
y = df["Fertility Rate"].astype(float).tolist()

policy_years = [1971, 1980, 2001, 2016, 2021]
policy_labels = {
    1971: "1971: Later, Longer, Fewer",
    1980: "1980: One-child policy",
    2001: "2001: Rural relaxation",
    2016: "2016: Two-child policy",
    2021: "2021: Three-child policy incentives",
}

graph = (
    AEGraph(
        width=2000,
        height=900,
        compwidth=2300,
        compheight=1200,
        drop_shadow=True,
        comp_name="China Fertility",
        fps=60,
        show_all_points=True,
        bg_color=[25,25,25],
        ui_color=[212,218,223],
        full_bg=True,
        cinematic_effects=True,
        distress_texture=5,
        ease_speed=0,
        ease_influence=60,
        font_scale=1
    )

    # fertility line
    .plot(
        x,
        y,
        color=[204,67,67],
        linewidth=5,
        animate=4,
        drop_shadow=True
    )

    .set_xlim(1950, 2025)
    .set_ylim(0, 12)

    .set_xticks(list(np.arange(1950, 2031, 10)))
    .set_yticks(list(np.arange(0, 12, 1)))

    .set_title("China Fertility Rate Over Time")
    .set_subtitle("Source: WorldBank", color=[134,137,140])

    .set_xlabel("Year")
    .set_ylabel("Births per Woman")

    # horizontal-only grid
    .grid(
        color=[118,135,152],
        alpha=0.25,
        hide_vertical=True,
        linestyle="--",
        dash_size=0.5
    )
)

# custom vertical policy lines and annotations
for year in policy_years:
    graph.plot(
        [year, year],
        [0, 11],
        color=[180,180,180],
        alpha=0.4,
        linestyle="--",
        dash_size=0.5,
        linewidth=2,
        animate=0.15
    )
    graph.annotate(
        policy_labels[year],
        year,
        11.5,
        fontsize=18,
        alignment="center"
    )

graph.render()