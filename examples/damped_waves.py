"""Damped oscillations (Arctic Signal theme).

Three decaying sine waves with different decay rates, line styles, and staggered
entrances (via delay) to show off line styling and animation timing together.
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

THEME = "Arctic Signal"

x = np.linspace(0, 12, 1500)


def damped(decay, freq=3.0):
    return np.exp(-decay * x) * np.sin(freq * x)


waves = (
    AEGraph(theme=THEME, comp_name="Damped Oscillations", drop_shadow=True, cinematic_effects=True)
    .set_title("Damped Oscillations")
    .set_xlabel("Time")
    .set_ylabel("Amplitude")
    .grid()
    .add_legend()
)
waves.plot(x, damped(0.15), color=config.object_color_1, label="slow decay",
           animate=3, drop_shadow=True)
waves.plot(x, damped(0.35), color=config.object_color_2, label="medium decay",
           linestyle="--", animate=3, delay=0.5, drop_shadow=True)
waves.plot(x, damped(0.60), color=config.ui_color, label="fast decay",
           linestyle=":", animate=3, delay=1.0, drop_shadow=True)
waves.set_xlim(0, 12).set_ylim(-1, 1).set_xticks().set_yticks()

waves.render()
