from aegraph import AEGraph
import numpy as np
import pandas as pd
import random

data = np.random.normal(loc=0, scale=1, size=4000)
data2 = np.random.normal(loc=0, scale=2, size=4000)

plot = (AEGraph(comp_name="Double Histogram", width=720, height=1280, drop_shadow=True, comp_height=1920, comp_width=1080, cinematic_effects=True, fps=12)
        .set_title("Histogram")
        .set_xlabel("Value")
        .set_ylabel("Frequency")
        .set_ylim(0,1)
        .set_xlim(-5,5)
        .grid(show=True, color="gray", alpha=0.3))
plot.histogram(
    data2, bins=200, color="p_red", alpha=0.5, label="Normal Distribution (σ = 2)", animate=4.0,bar_anim_times=0.50,density=True) \
    .histogram(data, bins=100, color="p_blue", alpha=0.5, label="Normal Distribution (σ = 1)", animate=4.0,bar_anim_times=0.50,density=True)
plot.set_xticks().set_yticks()
plot.reset_comp()
plot.render()