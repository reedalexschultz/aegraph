"""Three chi-square distributions overlaid as density histograms (Acid Jungle theme).

Demonstrates how the same plot reads across very different degrees of freedom.
Colors are pulled from the theme (two object colors plus the UI color).
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

THEME = "Acid Jungle"

chi_4 = np.random.chisquare(df=4, size=4000)
chi_8 = np.random.chisquare(df=8, size=4000)
chi_15 = np.random.chisquare(df=15, size=4000)

plot = (
    AEGraph(theme=THEME, comp_name="Chi-Square Histogram", width=720, height=720,
            comp_width=1920, comp_height=1080, drop_shadow=True, cinematic_effects=True)
    .set_title("Chi-Square Distributions")
    .set_xlabel("Value")
    .set_ylabel("Density")
    .set_xlim(0, 20)
    .set_ylim(0, 0.25)
    .grid()
    .add_legend()
)

plot.histogram(chi_4, bins=100, color=config.object_color_1, alpha=0.55,
               label="df = 4", density=True, animate=4.0)
plot.histogram(chi_8, bins=100, color=config.object_color_2, alpha=0.55,
               label="df = 8", density=True, animate=4.0)
plot.histogram(chi_15, bins=100, color=config.ui_color, alpha=0.45,
               label="df = 15", density=True, animate=4.0)

plot.set_xticks().set_yticks()
plot.render()
