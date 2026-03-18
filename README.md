# AEGraph

**A matplotlib-inspired Python library for creating animated graphs in Adobe After Effects**

AEGraph brings the power of data visualization to After Effects, allowing you to create professional, animated charts and graphs with cinematic effects. Whether you're creating data presentations, educational content, or motion graphics, AEGraph makes it easy to transform your data into stunning visualizations.

## Key Features

- **Native After Effects Integration** - Generates JSX scripts that run directly in AE
- **Multiple Plot Types** - Line plots, scatter plots, histograms, and bar charts
- **Rich Customization** - Colors, gradients, drop shadows, and cinematic effects
- **Smart Animation** - Easy ease interpolation, sequential animations, and trim paths
- **Automatic Layout** - Axes, ticks, grid lines, and coordinate mapping
- **Professional Effects** - Vignette, wiggle, and turbulent displace effects
- **Pandas Support** - Direct integration with pandas DataFrames and Series

## Quick Start

```python
from aegraph import AEGraph
import numpy as np

# Create data
t = np.linspace(0, 2*np.pi, 50)
y = np.sin(t)

# Create animated graph
plot = (AEGraph(width = 1280, height = 720, compwidth=1920, compheight=1080, drop_shadow=True, cinematic_effects=True)
        .plot(t, y, color="blue", animate=2.0)
        .scatter(t[::5], y[::5], color="red", radius=8)
        .set_title("Sine Wave")
        .set_xlabel("Angle (radians)")
        .set_ylabel("Amplitude")
        .grid(show=True, color="gray", alpha=0.3))

# Render in After Effects
plot.render()
```

## Installation Requirements

- **Python 3.7+** with NumPy
- **Adobe After Effects** (2020 or later)
- **macOS** (for automatic rendering via AppleScript)
- **Pandas** (optional, for DataFrame support)

## Class Constructor

### `AEGraph()`

```python
AEGraph(
    width=1920,                    # Graph logical width
    height=1080,                   # Graph logical height  
    comp_name="AEGraph_Comp",      # AE composition name
    bg_color="white",              # Background color
    comp_width=None,               # AE comp width (defaults to width)
    comp_height=None,              # AE comp height (defaults to height)
    position=None,                 # Graph center position in comp
    drop_shadow=False,             # Global drop shadow setting
    cinematic_effects=False,       # Add vignette adjustment layer
    wiggle=False,                  # Add wiggle/turbulent effects
    easy_ease=True,                # Apply easy ease to animations
    ease_speed=0,                  # Easy ease speed (0-100)
    ease_influence=33,             # Easy ease influence (0-100)
    fps=24                         # Composition frame rate
)
```

**Example:**
```python
# Create a cinematic graph with effects
graph = AEGraph(
    width=1920, 
    height=1080,
    drop_shadow=True,
    cinematic_effects=True,
    wiggle=True,
    easy_ease=True,
    ease_influence=50
)
```

## Plot Types

### `plot()` - Line Plots

Create animated line graphs with customizable styling.

```python
plot(x, y, color="blue", label=None, linewidth=4, animate=1.0, drop_shadow=False)
```

**Parameters:**
- `x, y`: Data points (lists, numpy arrays, or pandas Series)
- `color`: Color name (e.g., "blue", "red") or RGB list [r,g,b]
- `label`: Legend label
- `linewidth`: Line thickness in pixels
- `animate`: Animation duration in seconds (uses trim paths)
- `drop_shadow`: Add drop shadow effect

**Example:**
```python
import numpy as np

# Sine and cosine waves
t = np.linspace(0, 4*np.pi, 100)
sine = np.sin(t)
cosine = np.cos(t)

graph = (AEGraph()
         .plot(t, sine, color="blue", label="sin(x)", linewidth=6, animate=3.0)
         .plot(t, cosine, color="red", label="cos(x)", linewidth=6, animate=3.0)
         .set_title("Trigonometric Functions")
         .set_xlabel("x")
         .set_ylabel("y"))
```

### `scatter()` - Scatter Plots

Create animated scatter plots with sequential point animation.

```python
scatter(x, y, color="red", label=None, radius=8, animate=1.0, drop_shadow=False, bar_anim_times=None)
```

**Parameters:**
- `x, y`: Data points
- `color`: Point color
- `label`: Legend label
- `radius`: Point radius in pixels
- `animate`: Total animation duration (points appear sequentially)
- `drop_shadow`: Add drop shadow to points
- `bar_anim_times`: Custom per-point animation durations

**Example:**
```python
# Random scatter plot
np.random.seed(42)
x = np.random.randn(50)
y = np.random.randn(50)

graph = (AEGraph()
         .scatter(x, y, color="purple", radius=12, animate=4.0, drop_shadow=True)
         .set_title("Random Scatter Plot")
         .set_xlabel("X Values")
         .set_ylabel("Y Values")
         .grid(show=True))
```

### `histogram()` - Histograms

Create animated histograms with sequential bar animation.

```python
histogram(data, bins=10, color="p_blue", label=None, alpha=0.8, animate=1.0, 
          drop_shadow=False, bar_anim_times=None, density=False)
```

**Parameters:**
- `data`: 1D array-like data to bin
- `bins`: Number of bins or bin edges
- `color`: Bar color (try pastel colors like "p_blue", "p_green")
- `label`: Legend label
- `alpha`: Bar opacity (0-1)
- `animate`: Total animation duration
- `density`: If True, normalize to create probability density
- `bar_anim_times`: Custom per-bar animation durations

**Example:**
```python
# Normal distribution histogram
data = np.random.normal(0, 1, 1000)

graph = (AEGraph()
         .histogram(data, bins=20, color="p_green", alpha=0.7, animate=3.0)
         .set_title("Normal Distribution")
         .set_xlabel("Value")
         .set_ylabel("Frequency")
         .grid(show=True, alpha=0.2))
```

### `bar_graph()` - Bar Charts

Create bar charts with precise control over x-positions and heights.

```python
bar_graph(x_values, heights, bar_width=None, color="p_blue", label=None, 
          alpha=0.8, animate=1.0, drop_shadow=False, bar_anim_times=None)
```

**Parameters:**
- `x_values`: X positions for bars
- `heights`: Bar heights (must match length of x_values)
- `bar_width`: Bar width in data coordinates (auto-calculated if None)
- `color`: Bar color
- `alpha`: Bar opacity
- `animate`: Animation duration
- `bar_anim_times`: Custom per-bar timing

**Example:**
```python
# Sales data bar chart
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales = [100, 150, 120, 200, 180]
x_pos = list(range(len(months)))

graph = (AEGraph()
         .bar_graph(x_pos, sales, color="p_orange", animate=2.5)
         .set_title("Monthly Sales")
         .set_xlabel("Month")
         .set_ylabel("Sales ($)")
         .set_xticks(x_pos, months))
```

### `add_population_pyramid()` - Population Pyramids

Create animated mirrored horizontal bar charts (population pyramids) showing age/sex distribution.

```python
add_population_pyramid(
    csv_path=None,              # Path to CSV with columns: age, male, female
    ages=None,                  # List of age group labels
    male=None,                  # Male population values
    female=None,                # Female population values
    mode="percent",             # "percent" (% of total) or "counts" (raw values)
    animate=5.0,                # Total animation duration in seconds
    bar_duration=1.0,           # Duration each individual bar takes to animate
    animate_downward=True,      # Animate bars from top to bottom
    show_grid=True,             # Show vertical grid lines
    color_male=[73, 118, 222],  # Male bar color (name or RGB list)
    color_female=[168, 61, 104],# Female bar color (name or RGB list)
    label_male="Male",          # Legend label for male bars (False to hide)
    label_female="Female",      # Legend label for female bars (False to hide)
    drop_shadow=None            # Drop shadow on bars (None inherits global setting)
)
```

**Parameters:**
- `csv_path`: Path to a CSV file with three columns (age, male, female). Use this **or** pass arrays directly.
- `ages, male, female`: Arrays of age labels and population values (alternative to `csv_path`)
- `mode`: `"percent"` computes each age group's share of total population; `"counts"` uses raw numbers
- `animate` / `bar_duration`: Control overall and per-bar animation timing
- `animate_downward`: When `True`, bars animate sequentially from oldest to youngest age group
- `color_male` / `color_female`: Accept color names (`"blue"`) or RGB lists (`[73, 118, 222]`)
- `label_male` / `label_female`: Set to `False` to suppress legend entries

**Example — from a CSV file:**
```python
import numpy as np

graph = (AEGraph(
    width=800,
    height=1500,
    compwidth=1500,
    compheight=1920,
    drop_shadow=True,
    fps=60,
    bg_color="none",
    ui_color="black",
    font_scale=1.7,
).add_population_pyramid(
    csv_path="data/us/United-States-2024.csv",
    mode="percent",
    animate=1,
    bar_duration=0.5,
    animate_downward=True,
    drop_shadow=True,
    color_female=[224, 76, 157],
    color_male=[73, 118, 222],
    show_grid=False,
)
.set_xlabel("United States")
.set_xticks([-6, -4, -2, 0, 2, 4, 6])
.set_xlim(-6, 6)
.render())
```

**Example — passing arrays directly:**
```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/Bhutan-2024.csv")
ages = df["Age"].astype(str).tolist()
male = df["M"].astype(float).values
female = df["F"].astype(float).values
y_positions = np.arange(len(ages))

graph = (AEGraph(
    width=800,
    height=1500,
    compwidth=1500,
    compheight=1920,
    drop_shadow=True,
    fps=60,
    bg_color="none",
    ui_color="black",
    font_scale=1.7,
).add_population_pyramid(
    ages=ages,
    male=male,
    female=female,
    mode="percent",
    animate=1,
    bar_duration=0.5,
    animate_downward=True,
    drop_shadow=True,
    label_male=False,
    label_female=False,
    color_female=[224, 76, 157],
    color_male=[73, 118, 222],
    show_grid=False,
)
.set_xlabel("Bhutan")
.set_yticks(positions=y_positions, labels=ages)
.set_xticks([-6, -4, -2, 0, 2, 4, 6])
.set_xlim(-6, 6)
.render())
```

### `gradient()` - Color Gradients

Apply color gradients to the most recently added histogram or bar chart.

```python
gradient(color_start, color_end)
```

**Parameters:**
- `color_start`: Starting color (name, RGB list, or RGB values 0-255)
- `color_end`: Ending color

**Example:**
```python
# Gradient bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

graph = (AEGraph()
         .bar_graph(range(len(categories)), values, color="blue")
         .gradient("blue", "red")  # Blue to red gradient
         .set_title("Gradient Bar Chart")
         .set_xticks(range(len(categories)), categories))
```

## Styling & Customization

### Labels and Titles

```python
# Set plot title and axis labels
graph.set_title("My Amazing Graph")
graph.set_xlabel("X Axis Label")
graph.set_ylabel("Y Axis Label")
```

### Axis Limits

```python
# Set custom axis ranges
graph.set_xlim(0, 10)
graph.set_ylim(-1, 1)
```

### Custom Ticks

```python
# Custom tick positions and labels
graph.set_xticks([0, np.pi, 2*np.pi], ["0", "π", "2π"])
graph.set_yticks([-1, 0, 1], ["-1", "0", "1"])

# Auto-generate nice ticks
graph.set_xticks()  # Automatically chooses good tick positions
graph.set_yticks(nticks=5)  # Specify number of ticks
```

### Grid Lines

```python
# Enable grid with customization
graph.grid(show=True, color="gray", alpha=0.3)

# Disable grid
graph.grid(show=False)
```

### Hide Tick Labels

```python
# Hide all tick labels while keeping tick marks
graph.set_tick_labels(show=False)
```

## Animation & Effects

### Easy Ease Animation

AEGraph applies professional easy ease interpolation to all keyframes by default:

```python
# Configure easy ease parameters
graph = AEGraph(
    easy_ease=True,      # Enable easy ease (default)
    ease_speed=0,        # Speed percentage (0-100)
    ease_influence=33    # Influence percentage (0-100)
)
```

### Drop Shadows

```python
# Global drop shadow for all elements
graph = AEGraph(drop_shadow=True)

# Per-element drop shadows
graph.plot(x, y, drop_shadow=True)
graph.scatter(x, y, drop_shadow=True)
```

### Cinematic Effects

```python
# Add vignette and cinematic adjustment layers
graph = AEGraph(cinematic_effects=True)
```

### Wiggle Effects

```python
# Add organic movement with turbulent displace
graph = AEGraph(wiggle=True)
```

### Custom Animation Timing

```python
# Different animation durations per element
graph.plot(x, y, animate=2.0)
graph.scatter(x, y, animate=3.5)

# Custom per-point/bar timing
custom_times = [0.1, 0.2, 0.15, 0.3, 0.1]  # Per-point durations
graph.scatter(x, y, bar_anim_times=custom_times)
```

## Color System

AEGraph includes a comprehensive color system with named colors:

### Standard Colors
- `"red"`, `"green"`, `"blue"`, `"yellow"`, `"cyan"`, `"magenta"`
- `"orange"`, `"purple"`, `"gray"`, `"black"`, `"white"`
- `"brown"`, `"lime"`, `"navy"`, `"teal"`, `"gold"`

### Pastel Colors (Perfect for Charts)
- `"p_red"`, `"p_green"`, `"p_blue"`, `"p_yellow"`
- `"p_cyan"`, `"p_magenta"`, `"p_orange"`, `"p_purple"`
- `"p_pink"`, `"p_gray"`, `"p_brown"`, `"p_lime"`
- `"p_navy"`, `"p_teal"`, `"p_gold"`

### Custom RGB Colors
```python
# RGB values 0-255
graph.plot(x, y, color=[51, 153, 204])
```

## Saving & Rendering

### Save JSX Script

```python
# Save .jsx file without running
graph.save("my_graph.jsx", folder_path="./output/")
```

### Render in After Effects

```python
# Automatically save and run in After Effects (.jsx is saved in the current folder)
graph.render()

# Specify AE version and output folder
graph.render(ae_version="Adobe After Effects 2024", folder_path="./ae_scripts/")
```

### Reset Composition

```python
# Clear all layers from active composition
graph.reset_comp()
```

## Advanced Examples

### Multi-Dataset Comparison

```python
# Generate sample data
x = np.linspace(0, 10, 500)
y1 = np.sin(x) * np.exp(-x/10)
y2 = np.cos(x) * np.exp(-x/8)
y3 = np.sin(2*x) * np.exp(-x/15)

# Create comparison plot
graph = (AEGraph(comp_width=1920, comp_height=1080, width=1280, height = 720, 
                easy_ease=True, ease_influence=50,
                wiggle=True, cinematic_effects=True, drop_shadow=True, fps = 12)
         .plot(x, y1, color="blue", label="Dataset 1", linewidth=6, animate=2.0)
         .plot(x, y2, color="red", label="Dataset 2", linewidth=6, animate=2.5)
         .plot(x, y3, color="green", label="Dataset 3", linewidth=6, animate=3.0)
         .set_title("Exponentially Decaying Oscillations")
         .set_xlabel("Time (s)")
         .set_ylabel("Amplitude")
         .set_xticks()
         .set_yticks()
         .grid(show=True, color="lightgray", alpha=0.5)
         .add_legend())

graph.render()
```

### Statistical Distribution

```python
import numpy as np

# Generate different distributions
normal_data = np.random.normal(0, 1, 1000)
exponential_data = np.random.exponential(2, 1000)

graph = (AEGraph(comp_width=1920, comp_height=1080, width=1280, height = 720, 
                easy_ease=True, ease_influence=50,
                wiggle=True, cinematic_effects=True, drop_shadow=True, fps = 12)
         .histogram(exponential_data, bins=60, color="p_red", alpha=0.7,
                   label="Exponential", animate=3.5, bar_anim_times=0.75, density=True)
         .histogram(normal_data, bins=30, color="p_blue", alpha=0.7, 
                   label="Normal", animate=3.0, density=True, bar_anim_times=0.75)
         .set_title("Probability Distributions")
         .set_xlabel("Value")
         .set_ylabel("Frequency")
         .set_yticks()
         .set_xticks()
         .grid(show=True, alpha=0.3))

graph.render()
```

### Population Pyramid — Multiple Countries

```python
import pandas as pd
import numpy as np
from pathlib import Path

folder = Path("data/developing-africa")
files = sorted(folder.glob("*.csv"))

for file in files:
    df = pd.read_csv(file)
    ages = df["Age"].astype(str).tolist()
    male = df["M"].astype(float).values
    female = df["F"].astype(float).values
    y_positions = np.arange(len(ages))
    country = file.stem.removesuffix("-2024")

    (AEGraph(
        width=800,
        height=1500,
        compwidth=1500,
        compheight=1920,
        comp_name=country,
        drop_shadow=True,
        fps=60,
        bg_color="none",
        ui_color="black",
        font_scale=1.7,
    ).add_population_pyramid(
        ages=ages,
        male=male,
        female=female,
        mode="percent",
        animate=1,
        bar_duration=0.5,
        animate_downward=True,
        drop_shadow=True,
        label_male=False,
        label_female=False,
        color_female=[224, 76, 157],
        color_male=[73, 118, 222],
        show_grid=False,
    )
    .set_xlabel(country)
    .set_yticks(positions=y_positions, labels=ages)
    .set_xticks([-6, -4, -2, 0, 2, 4, 6])
    .set_xlim(-6, 6)
    .render())
```

### Scientific Data with Custom Styling

```python
# Temperature measurements
time_hours = np.arange(0, 24, 0.5)
temperature = 20 + 5*np.sin(2*np.pi*time_hours/24 - np.pi/2) + np.random.normal(0, 0.5, len(time_hours))

graph = (AEGraph(comp_width=1920, comp_height=1080, width=1280, height = 720, 
                easy_ease=True, ease_influence=50,
                wiggle=True, cinematic_effects=True, drop_shadow=True, fps = 12)
         .plot(time_hours, temperature, color="red", linewidth=5, 
               animate=4.0, drop_shadow=True)
         .scatter(time_hours[::4], temperature[::4], color="blue", 
                 radius=8, animate=4.0, drop_shadow=True)
         .set_title("24-Hour Temperature Monitoring")
         .set_xlabel("Time (hours)")
         .set_ylabel("Temperature (°C)")
         .set_yticks()
         .set_xticks([0, 6, 12, 18, 24], ["0:00", "6:00", "12:00", "18:00", "24:00"])
         .grid(show=True, color="gray", alpha=0.2))

graph.render()
```

## Technical Details

### Coordinate System
- **Data coordinates**: Your input data values
- **Graph coordinates**: Centered at (0,0) with specified width/height
- **Comp coordinates**: After Effects composition pixel coordinates
- All conversions are handled automatically

### Animation System
- **Trim Paths**: Line plots animate by revealing the path
- **Scale Animation**: Scatter points scale from 0 to 100%
- **Sequential Timing**: Multiple elements animate in sequence
- **Easy Ease**: Bezier curve interpolation

## Troubleshooting

### Common Issues

1. **"Unknown color name" error**
   - Use valid color names from the color system above
   - Or provide RGB values as lists: `[r, g, b]`

2. **After Effects script fails**
   - Ensure After Effects scripting is enabled
   - Check that the AE version matches in `render()` call
   - Verify the JSX file path is accessible

3. **Animation not smooth**
   - Adjust `easy_ease` parameters
   - Use appropriate animation durations
   - Check frame rate settings

4. **Gradient not working**
   - Call `gradient()` immediately after `histogram()` or `bar_graph()`
   - Only histograms and bar graphs support gradients

### Best Practices

- Always call methods in a chained fashion for readability

## License

AEGraph is released under the MIT License. Feel free to use it in personal and commercial projects.

---

**Happy graphing!**