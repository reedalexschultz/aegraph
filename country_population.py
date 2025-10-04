from aegraph import AEGraph
import numpy as np
import pandas as pd
import random

pops = pd.read_csv('./country_population.csv')
pops = pops.drop(columns=['Country Code','Indicator Name', 'Indicator Code','Unnamed: 69']).set_index(pops.columns[0]).T

from aegraph import COLOR_NAMES
COLOR_NAMES.keys()

countries_hist = (
    AEGraph(width=1280, height=720, compwidth=1920, compheight=1080,
            drop_shadow=True, cinematic_effects=True, comp_name="Indonesia Vs. United States Populations", fps=24)
    .grid()
    .set_xlabel("Country")
    .set_ylabel("Population")
    .set_title("Indonesia Vs. United States Populations")
    .grid(show=True, color="gray", alpha=0.3)
    .add_legend()
)

INDONESIA_COLORS = ('p_red', (255, 140, 140))
UNITED_STATES_COLORS = ('p_blue', (140, 140, 255))

for colors, country in {UNITED_STATES_COLORS: "United States", INDONESIA_COLORS: "Indonesia"}.items():
    
    countries_hist.bar_graph(
        pops.index.astype(int),
        pops[country].astype(int),
        color=colors[0],
        label=country,
        alpha=0.75,
        animate=4,
        bar_anim_times=1.2,
        drop_shadow=True
    )
    countries_hist.gradient(
    colors[0],
    colors[1]
    ) 
countries_hist.set_xticks().set_yticks()

countries_hist.render()