# AEGraph

**A matplotlib-inspired Python library for creating animated graphs in Adobe After Effects.**

AEGraph turns data into cinematic, animated visualizations. You describe a chart
with a familiar, matplotlib-style API in Python, and AEGraph generates an
ExtendScript (`.jsx`) file that builds and animates the whole composition inside
After Effects — axes, ticks, gridlines, legends, colorbars, easing, and more.

```python
from aegraph import AEGraph
import numpy as np

t = np.linspace(0, 2 * np.pi, 100)

(
    AEGraph(comp_name="Sine Wave")
    .plot(t, np.sin(t), color="blue", label="sin(t)", animate=2.0)
    .set_title("Sine Wave")
    .grid()
    .add_legend()
    .render()          # writes a .jsx and runs it in After Effects
)
```

## What it's good for

- **Explainer & educational content** — animate equations, simulations, and
  data stories (heat equations, Fourier series, orbital mechanics, diffusion,
  vector fields, reaction-diffusion).
- **Data-driven motion graphics** — line, scatter, histogram, bar,
  population-pyramid, quiver (vector field), and heatmap charts that animate
  themselves.
- **Graphs that evolve over time** — pass one dataset per time step and AEGraph
  keyframes the curve, scatter cloud, vector field, or heatmap so it morphs,
  moves, and recolors.
- **A polished house style** — 51 built-in themes, gradients, colorbars, drop
  shadows, and cinematic effects.
- **pandas-friendly** — pass `Series`/`DataFrame` columns directly; labels and
  titles are inferred from column names.

## Install

```bash
pip install -r requirements.txt
```

You'll also need Adobe After Effects (2020+) with scripting enabled.

Runnable scripts live in [`examples/`](examples/), including vector-field and
heatmap demos like `fluid_vortex.py`, `mandelbrot_heatmap.py`, and
`reaction_diffusion.py`. The bundled script builds batches of animated
compositions:

```bash
python scripts/generate_animated_examples.py        # render each in After Effects
python scripts/generate_animated_examples.py save   # just write the .jsx files
```

## License

AEGraph is released under the MIT License. See [LICENSE](LICENSE).
