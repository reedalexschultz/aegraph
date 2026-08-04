"""Pie and donut charts side by side in one comp.

Pie charts are polar: they ignore x/y axes and the grid, center themselves in
the panel, and reveal each wedge in sequence.
"""

RENDERING_TIME = "SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import subplots

THEME = "Duke Light"

labels = ["Search", "Direct", "Social", "Referral", "Email"]
values = [42, 23, 16, 12, 7]

fig, axes = subplots(1, 2, comp_name=f"Traffic Mix {THEME}", theme=THEME,
                     wspace=0.50)

axes[0].pie(values, labels=labels, show_percent=True, animate=2.0,
            gap=20.0, roundness=6.0, leader_lines=True)
# axes[0].set_title("Pie")

axes[1].pie(values, labels=labels, donut=0.55, show_percent=True, animate=2.0,
            gap=20.0, roundness=10.0, leader_lines=True)
# axes[1].set_title("Donut")

fig.suptitle("Traffic Sources")
fig.render()
