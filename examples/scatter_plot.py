"""Scatter demo: two correlated point clouds styled entirely by a theme.

Instead of hardcoding colors, comp size, grid styling, and animation timing,
this leans on the active theme (Midnight Neon) and the aegraph_config defaults.
The only data colors used come straight from the theme's object colors.
"""

RENDERING_TIME = "VERY_SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
from aegraph_config import config
import numpy as np

THEME = "Viridis Lab"

n = 25
t = np.linspace(0, 1, n)
x_a = t + np.random.normal(0, 0.08, n)
y_a = x_a + np.random.normal(0, 0.08, n)
x_b = t * 0.5 + np.random.normal(0, 0.08, n)
y_b = t + np.random.normal(0, 0.08, n)

scatter = (
    AEGraph(theme=THEME, comp_name="Scatter Demo", drop_shadow=True, cinematic_effects=True, plot_frame=True, xaxis_location="bottom", yaxis_location="left")
    .set_title("Two Correlated Clouds")
    .set_xlabel("x")
    .set_ylabel("y")
    .grid()
    .add_legend()
)
scatter.scatter(x_a, y_a, color=config.object_color_1, label="Series A", radius=10, alpha=0.85, drop_shadow=True)
scatter.scatter(x_b, y_b, color=config.object_color_2, label="Series B", radius=10, alpha=0.85, drop_shadow=True)
scatter.set_xlim(0, 1).set_ylim(0, 1).set_xticks().set_yticks()

scatter.render()
