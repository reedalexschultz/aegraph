"""Log-scale line plot (Petroleum Glow theme).

Exponential curves look like straight lines on a logarithmic y-axis. This shows
set_yscale("log") with theme-driven colors and auto y-limits.
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

THEME = "Petroleum Glow"

x = np.linspace(0, 10, 400)
slow = np.exp(0.5 * x)
fast = np.exp(1.0 * x)

logp = (
    AEGraph(theme=THEME, comp_name="Log Scale Demo", drop_shadow=True, cinematic_effects=True)
    .set_title("Exponential Growth on a Log Axis")
    .set_xlabel("Time")
    .set_ylabel("Value (log scale)")
    .set_yscale("log")
    .grid()
    .add_legend()
)
logp.plot(x, slow, color=config.object_color_2, label="e^(0.5x)", animate=3, drop_shadow=True)
logp.plot(x, fast, color=config.object_color_1, label="e^(x)", animate=3, drop_shadow=True)
logp.set_xticks().set_yticks()

logp.render()
