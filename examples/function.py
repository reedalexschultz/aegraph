"""The Weierstrass function: continuous everywhere, differentiable nowhere.

Rendered with the Retro Terminal theme so the fractal curve glows on black.
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

THEME = "Wasabi Porcelain"

x = np.linspace(-0.4, 0.4, 10000)
y = np.zeros_like(x)
for n in range(1, 10):
    y += np.cos(3**n * np.pi * x) / (2**n)

weier = (
    AEGraph(theme=THEME, comp_name="Weierstrass Function", drop_shadow=True, cinematic_effects=True)
    .set_title("Weierstrass Function")
    .set_xlabel("x")
    .set_ylabel("W(x)")
    .grid()
    .add_legend()
)
weier.plot(x, y, color=config.object_color_1, label="W(x)", linewidth=1, animate=3)
weier.set_xlim(-0.4, 0.4).set_ylim(-2, 2).set_xticks().set_yticks()

weier.render()
