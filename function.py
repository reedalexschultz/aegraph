from aegraph import AEGraph
import numpy as np
import pandas as pd
import random

math_graph = (
    AEGraph(width=1280, height=720, compwidth=1920, compheight=1080, drop_shadow=True, cinematic_effects=True,comp_name="Weierstrass Function", fps=60)
    .grid()
    .set_xlabel("x")
    .set_ylabel("y")
    .set_title("Weierstrass Function")
    .grid(show=True, color="gray", alpha=0.3)
    .add_legend()
)
x_weier = np.linspace(-0.4, 0.4, 10000)
y_weier = np.zeros_like(x_weier)
for n in range(1, 10):
    y_weier += np.cos(3**n * np.pi * x_weier) / (2**n)
math_graph.plot(x_weier, y_weier, color="navy", label="Weierstrass Function", linewidth=0.5, animate=3)
math_graph.set_ylim(-2,2).set_xlim(-0.4,0.4).set_yticks().set_xticks()

math_graph.render()