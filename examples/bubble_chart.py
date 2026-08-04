"""Bubble chart (Velvet Gold theme).

Two channels of information beyond x/y:
  - bubble *size* via radius_size="dynamic_range" (raw magnitudes are auto-scaled
    so the largest equals config.scatter_max_radius).
  - bubble *color* via c=, which maps a value array onto the theme gradient and
    draws an automatic colorbar.
"""

RENDERING_TIME = "VERY_SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np

THEME = "Velvet Gold"

rng = np.random.default_rng(7)
n = 30
x = rng.uniform(0, 10, n)
y = rng.uniform(0, 10, n)
size = rng.uniform(1, 100, n)          # drives bubble radius
value = x + y + rng.normal(0, 1, n)    # drives bubble color

bubbles = (
    AEGraph(theme=THEME, comp_name="Bubble Chart", drop_shadow=True, cinematic_effects=True)
    .set_title("Bubbles: size = magnitude, color = value")
    .set_xlabel("x")
    .set_ylabel("y")
    .grid()
)
bubbles.scatter(
    x, y,
    radius=size,
    radius_size="dynamic_range",
    c=value,
    alpha=0.85,
    animate=1.5,
    drop_shadow=True,
)
bubbles.set_xlim(0, 10).set_ylim(0, 10).set_xticks().set_yticks()

bubbles.render()
