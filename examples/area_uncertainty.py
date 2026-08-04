"""A forecast line with a shaded confidence band and a reference line.

Shows ``fill_between`` for uncertainty, ``plot`` for the central estimate,
``errorbar`` for sampled observations, and ``axhline`` for a target threshold.
"""

RENDERING_TIME = "VERY_SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

import numpy as np
from aegraph import AEGraph

rng = np.random.default_rng(3)
x = np.linspace(0, 12, 120)
mean = 10 + 2.2 * np.sin(x / 2) + 0.35 * x
sigma = 0.8 + 0.12 * x  # widening uncertainty

g = (
    AEGraph(theme="Deep Ocean", comp_name="Forecast with Uncertainty",
            xaxis_location="bottom", yaxis_location="left", plot_frame=True)
    .set_title("Forecast with Confidence Band")
    .set_xlabel("months ahead")
    .set_ylabel("value")
    .grid()
)

# 95% band, then the central estimate on top.
g.fill_between(x, mean - 1.96 * sigma, mean + 1.96 * sigma,
               color="blue", alpha=0.22, label="95% CI", animate=2.0)
g.plot(x, mean, color="blue", linewidth=5, label="forecast", animate=2.5)

# A few sampled observations with error bars.
obs_x = x[::18]
obs_y = mean[::18] + rng.normal(0, 1.0, obs_x.size)
g.errorbar(obs_x, obs_y, yerr=1.5, color="white", capsize=10, label="observed")

# Target threshold.
g.axhline(20, color="red", linestyle="dashed", label="target")

g.add_legend()
g.render()
