"""A four-panel dashboard rendered into a single After Effects comp.

Demonstrates the matplotlib-style ``subplots`` wrapper: each panel is an
independent AEGraph (its own data, limits, and ticks) tiled into one shared
composition, with a figure-level title on top.
"""

RENDERING_TIME = "SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

import numpy as np
from aegraph import subplots

t = np.linspace(0, 4 * np.pi, 240)

fig, axes = subplots(
    2, 2,
    comp_name="Dashboard",
    theme="Slate Report",
    wspace=0.26, hspace=0.34,
)

# Top-left: a line
axes[0][0].plot(t, np.sin(t), color="blue", label="sin", animate=2.0)
axes[0][0].set_title("Signal").grid().add_legend()

# Top-right: a scatter with a trendline
x = np.linspace(0, 10, 60)
y = 0.6 * x + np.random.default_rng(1).normal(0, 1.2, x.size)
axes[0][1].scatter(x, y, color="orange", radius=7)
axes[0][1].add_trendline(x, y, kind="linear", color="red", label="fit")
axes[0][1].set_title("Correlation").grid().add_legend()

# Bottom-left: bars
axes[1][0].bar_graph([1, 2, 3, 4, 5], [4, 7, 3, 8, 5])
axes[1][0].set_title("Categories").grid()

# Bottom-right: an area chart
axes[1][1].area(t, np.abs(np.sin(t)) + 0.3, color="teal", alpha=0.4,
                animate=2.0)
axes[1][1].set_title("Volume").grid()

fig.suptitle("Quarterly Report")
fig.render()
