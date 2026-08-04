"""Stacked and grouped bar charts.

``bar_stacked`` draws all layers as one element and reveals each column
bottom-to-top with the same smoothstep cumulative animation as ``pie()``.
``bar_grouped`` places series side by side with the standard bar animation.
"""

RENDERING_TIME = "VERY_SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

import numpy as np
from aegraph import subplots

quarters = [1, 2, 3, 4]
product_a = [12, 15, 14, 18]
product_b = [8, 9 , 11, 10]
product_c = [5, 6, 7, 9]

fig, axes = subplots(1, 2, comp_name="Bar Compositions",
                     theme="Copper Oxide", wspace=0.30)

axes[0].bar_stacked(
    quarters, [product_a, product_b, product_c],
    labels=["Product A", "Product B", "Product C"],
    animate=2.0,
)
axes[0].set_title("Stacked").grid().add_legend()

axes[1].bar_grouped(
    quarters, [product_a, product_b, product_c],
    labels=["Product A", "Product B", "Product C"],
    animate=2.0,
)
axes[1].set_title("Grouped").grid().add_legend()

fig.suptitle("Revenue by Quarter")
fig.render()
