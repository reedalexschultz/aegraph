"""Animated graph bounds: zoom the view window over time.

Demonstrates the ``view_keyframes`` parameter. The x/y limits shift from a
wide view into a tight window, and every coordinate-driven layer (line, grid,
ticks, tick labels, axes) re-maps as a true axis rescale (stroke widths and
font sizes stay constant).
"""

# Overall reduced frame rate so that rendering isn't so intensive

RENDERING_TIME = "MODERATE"

import numpy as np

import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph

x = np.linspace(0, 10, 200)
y = np.sin(x) * 30 + 50

graph = AEGraph(
    animate_axes=True,
    fps=12,
    compwidth=1920,
    compheight=1080,
    width=1300,
    height=750,
    comp_name="AnimatedBoundsZoom",
)
graph.plot(x, y, color="blue", linewidth=4)
graph.grid()
graph.set_xlabel("x")
graph.set_ylabel("y")
graph.set_title("Animated view bounds")
graph.set_view_keyframes({
        0.0: [(0, 10), (0, 100)],   # wide view
        3.0: [(0, 10), (0, 100)],   # wide view
        6.0: [(1, 3), (70, 90)],    # zoom into the first hump
        9.0: [(0, 10), (70, 90)],   # zoom back out horizontally
    },
    ease_influence=50
    )

if __name__ == "__main__":
    graph.render()
