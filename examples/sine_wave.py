"""Animated sine wave rendered with the Deep Ocean theme.

Colors and grid come from the theme; animation duration is the only timing knob.
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

THEME = "Citrus Grove"

x = np.linspace(0, 10, 2500)
y = np.sin(x)

sine = (
    AEGraph(theme=THEME, comp_name="Sine Wave", drop_shadow=True, cinematic_effects=True)
    .set_title("Sine Wave")
    .set_xlabel("x")
    .set_ylabel("sin(x)")
    .grid()
    .add_legend()
)
sine.plot(x, y, color=config.object_color_1, label="sin(x)", animate=3, drop_shadow=True)
sine.set_xticks().set_yticks()

sine.render()
