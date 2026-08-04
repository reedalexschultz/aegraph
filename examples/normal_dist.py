"""Two overlaid normal distributions as density histograms (Cyber Grape theme).

Each histogram borrows one of the theme's object colors; bin animation timing
falls back to the aegraph_config defaults.
"""

RENDERING_TIME = "SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
from aegraph_config import config
import numpy as np

THEME = "Lavender Archive"

narrow = np.random.normal(0, 1, 4000)
wide = np.random.normal(0, 2, 4000)

hist = (
    AEGraph(theme=THEME, comp_name="Double Histogram", width=900, height=900,
            comp_width=1200, comp_height=1080, drop_shadow=True, cinematic_effects=True)
    .set_title("Two Normal Distributions")
    .set_xlabel("Value")
    .set_ylabel("Density")
    .set_xlim(-6, 6)
    .grid()
    .add_legend()
)
hist.histogram(wide, bins=200, color=config.object_color_2, alpha=0.55,
               label="Normal (σ = 2)", density=True, animate=4.0)
hist.histogram(narrow, bins=100, color=config.object_color_1, alpha=0.55,
               label="Normal (σ = 1)", density=True, animate=4.0)
hist.set_xticks().set_yticks()

hist.render()
