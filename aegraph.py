import datetime
import os
import numpy as np
from typing import List, Tuple, Optional, Union
import csv
import math
try:
    import pandas as pd
except ImportError:
    pd = None

COLOR_NAMES = {
    "black":      [0.0, 0.0, 0.0],
    "white":      [1.0, 1.0, 1.0],
    "red":        [1.0, 0.0, 0.0],
    "green":      [0.0, 1.0, 0.0],
    "blue":       [0.0, 0.0, 1.0],
    "yellow":     [1.0, 1.0, 0.0],
    "cyan":       [0.0, 1.0, 1.0],
    "magenta":    [1.0, 0.0, 1.0],
    "orange":     [1.0, 0.5, 0.0],
    "purple":     [0.5, 0.0, 0.5],
    "gray":       [0.5, 0.5, 0.5],
    "lightgray":  [0.8, 0.8, 0.8],
    "brown":      [0.6, 0.4, 0.2],
    "lime":       [0.75, 1.0, 0.0],
    "navy":       [0.0, 0.0, 0.5],
    "teal":       [0.0, 0.5, 0.5],
    "gold":       [1.0, 0.84, 0.0],

    "p_red":      [1.0, 0.6, 0.6],
    "p_green":    [0.6, 1.0, 0.6],
    "p_blue":     [0.6, 0.8, 1.0],
    "p_yellow":   [1.0, 1.0, 0.6],
    "p_cyan":     [0.6, 1.0, 1.0],
    "p_magenta":  [1.0, 0.6, 1.0],
    "p_orange":   [1.0, 0.8, 0.6],
    "p_purple":   [0.8, 0.6, 1.0],
    "p_pink":     [1.0, 0.8, 0.9],
    "p_gray":     [0.8, 0.8, 0.8],
    "p_brown":    [0.8, 0.7, 0.6],
    "p_lime":     [0.8, 1.0, 0.6],
    "p_navy":     [0.6, 0.7, 0.9],
    "p_teal":     [0.6, 0.9, 0.9],
    "p_gold":     [1.0, 0.9, 0.6],
}

DEFAULT_DROP_SHADOW = {
    'direction': 135,
    'distance': 5,
    'softness': 10,
    'color': 'black',
    'opacity': 0.3,
}


def color_to_js(color: Union[str, List[float], Tuple[float, float, float]]) -> str:
    """Convert color name or RGB list/tuple to AE-friendly JS array [r,g,b].

    Accepts:
    - Named colors from COLOR_NAMES (0–1 floats)
    - RGB lists/tuples either in 0–1 floats or 0–255 ints
    """
    if isinstance(color, str):
        rgb = COLOR_NAMES.get(color.lower())
        if rgb is None:
            raise ValueError(f"Unknown color name: {color}")
    elif isinstance(color, (list, tuple)) and len(color) == 3:
        rgb = list(color)
        # Auto-normalize 0–255 integers to 0–1 floats for AE
        if max(rgb) > 1:
            rgb = [float(v) / 255.0 for v in rgb]
        else:
            rgb = [float(v) for v in rgb]
    else:
        raise ValueError("Color must be a name or 3-value RGB list/tuple.")
    return f"[{rgb[0]}, {rgb[1]}, {rgb[2]}]"


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class AEGraph:
    """
    AEGraph: A matplotlib-inspired graphing API for Adobe After Effects.
    
    This class provides a simple interface to create animated graphs in After Effects
    using Python. It supports line plots, scatter plots, axes, ticks, grid, and labels.
    
    Key Features:
    - Line plots with customizable colors, linewidth, and animation
    - Scatter plots with customizable colors, radius, and animation
    - Automatic axes positioning and styling
    - Custom tick marks and labels
    - Grid lines with customizable appearance
    - Title and axis labels
    - Automatic coordinate system handling for After Effects
    - Optional cinematic adjustment layer with CC Vignette effects
    - Optional distress texture layer for film-grain/grunge treatment
    - Optional wiggle adjustment layer with Turbulent Displace for organic movement
    - Easy ease animation system for smooth, professional keyframe interpolation
    
    Coordinate System:
    - Data coordinates are automatically converted to After Effects coordinates
    - Line plots and axes use shape layer coordinates (centered at comp center)
    - Scatter points use layer positioning (absolute comp coordinates)
    - All elements are automatically aligned and positioned correctly
    
    Example Usage:
        from aegraph import AEGraph
        import numpy as np
        
        # Create data
        t = np.linspace(0, 2*np.pi, 50)
        y = np.sin(t)
        
        # Create graph with drop shadows, cinematic effects, wiggle, and easy ease
        plot = (AEGraph(width=1920, height=1080, drop_shadow=True, cinematic_effects=True, wiggle=True, easy_ease=True, ease_speed=00, ease_influence=33)
                .plot(t, y, color="blue", animate=2.0, drop_shadow=True)
                .scatter(t[::5], y[::5], color="red", radius=8, drop_shadow=True)
                .set_title("Sine Wave")
                .set_xticks([0, np.pi, 2*np.pi], ["0", "π", "2π"])
                .set_yticks([-1, 0, 1])
                .grid(show=True, color="gray", alpha=0.3))
        
        # Save and render
        plot.save("my_graph.jsx")
        plot.render()
    """
    def __init__(self,
                 width=1920,
                 height=1080,
                 comp_name="AEGraph_Comp",
                 bg_color="white", comp_width=None,
                 comp_height=None, compwidth=None,
                 compheight=None, position=None,
                 show_all_points=False, drop_shadow=False,
                 full_bg=False,
                 distress_texture=None,
                 cinematic_effects=False, wiggle=False,
                 easy_ease=True,
                 ease_speed=00,
                 ease_influence=33,
                 animate_opacity=True,
                 animate_axes=True,
                 ui_color: Union[str, List[float], Tuple[float, float, float]] = "black",
                 font_scale: float = 1.0,
                 bg_stroke_width = 0,
                 bg_stroke_color = [0.15, 0.15, 0.15],
                 fps=24):
        """
        Initialize a new AEGraph instance.
        Args:
            width (int): Graph logical width (default: 1920)
            height (int): Graph logical height (default: 1080)
            comp_name (str): Name of the AE composition (default: "AEGraph_Comp")
            bg_color (str or list): Background color name or RGB list (default: "white").
                Pass "none" to omit the background rectangle entirely.
            comp_width/compheight/compwidth/compheight (int, optional): AE composition width/height (default: width/height)
            position (tuple, optional): (x, y) center of graph in comp coordinates (default: comp center)
            show_all_points (bool): Whether to plot all points or only those within bounds (default: False)
            drop_shadow (bool): Whether to add drop shadow to graph elements (default: False)
            full_bg (bool): Whether to make background cover full composition (True) or just graph area (False, default)
            distress_texture (int, optional): Distress texture preset to place above GraphBG but below
                all other graph layers. Currently supports `1` for `distress_textures/grunge1.jpg`
                (default: None)
            cinematic_effects (bool): Whether to add cinematic adjustment layer with vignette effects (default: False)
            wiggle (bool): Whether to add wiggle adjustment layer with Turbulent Displace (default: False)
            easy_ease (bool): Whether to apply easy ease to all animation keyframes (default: True)
            ease_speed (int): Easy ease speed percentage (default: 00)
            ease_influence (int): Easy ease influence percentage (default: 33)
            fps (int): Frame rate for the After Effects composition (default: 24)
            ui_color (str or list/tuple): Color for all non-data UI elements like axes, tick marks,
                all tick/label/title/legend/annotation text (default: "black"). Accepts named colors
                from COLOR_NAMES or RGB values in 0–1 floats or 0–255 ints.
            font_scale (float): Global scale multiplier for all text sizes (ticks, labels, title,
                legend, and annotations). Default 1.0.
        """
        self.width = width
        self.height = height
        # Priority: compwidth/compheight > comp_width/comp_height > width/height
        if compwidth is not None:
            self.comp_width = compwidth
        elif comp_width is not None:
            self.comp_width = comp_width
        else:
            self.comp_width = width
        if compheight is not None:
            self.comp_height = compheight
        elif comp_height is not None:
            self.comp_height = comp_height
        else:
            self.comp_height = height
        self.comp_name = comp_name
        self.bg_color = bg_color
        self.position = position  # (x, y) in comp coordinates, or None for center
        self.show_all_points = show_all_points
        self.drop_shadow = drop_shadow  # Global drop shadow setting
        self.full_bg = full_bg  # Whether background covers full composition or just graph
        self.distress_texture = distress_texture
        self.elements = []  # List of plot elements (dicts)
        self.title = None
        self.subtitle = None
        self.subtitle_color = None
        self.xlabel = None
        self.ylabel = None
        self.legend = []
        self.legend_style = 'color_only'  # 'color_only' or 'line_style'
        self.xlim = None
        self.ylim = None
        self.xticks = None  # X-axis tick positions and labels
        self.yticks = None  # Y-axis tick positions and labels
        self.show_grid = False   # Whether to show grid
        self.grid_color = "gray"  # Grid color
        self.grid_alpha = 0.3     # Grid opacity
        self.grid_linestyle = "solid"  # Grid line style
        self.grid_dash_size = 1.0  # Grid dash size scale factor
        self.hide_horizontal = False  # Hide horizontal grid lines
        self.hide_vertical = False  # Hide vertical grid lines
        self.show_tick_labels = True 
        self.cinematic_effects = cinematic_effects  # Whether to add cinematic adjustment layer
        self.wiggle = wiggle  # Whether to add wiggle adjustment layer with Turbulent Displace
        self.easy_ease = easy_ease  # Whether to apply easy ease to all animation keyframes
        self.ease_speed = ease_speed  # Easy ease speed percentage
        self.ease_influence = ease_influence  # Easy ease influence percentage
        self.animate_opacity = animate_opacity  # Enable/disable opacity fade animations
        self.animate_axes = animate_axes  # Enable/disable Trim Paths animation for axes
        self.fps = fps  # Frame rate for the After Effects composition
        self.bg_stroke_color = bg_stroke_color
        self.bg_stroke_width = bg_stroke_width
        self.ui_color = ui_color
        self.font_scale = float(font_scale) if font_scale is not None else 1.0
        # Tick formatting flags
        self.percent_tick_labels = False  # When True, x-ticks render as absolute percentages

    def _get_dash_values(self, linestyle: str, dash_size: float = 1.0) -> Tuple[float, float]:
        """
        Get dash and gap values based on linestyle and dash_size scale factor.
        Returns (dash_length, gap_length).
        
        Supported styles:
        - 'solid' or '-': (0, 0) - no dashes
        - 'dashed' or '--': (20, 10) * dash_size
        - 'dotted' or ':': (2, 4) * dash_size
        - 'dashdot' or '-.': uses multiple dashes via multiple setValue calls
        """
        linestyle = linestyle.lower() if isinstance(linestyle, str) else 'solid'
        dash_size = float(dash_size) if dash_size else 1.0
        
        if linestyle in ['solid', '-']:
            return 0, 0
        elif linestyle in ['dashed', '--']:
            return int(20 * dash_size), int(10 * dash_size)
        elif linestyle in ['dotted', ':']:
            return int(2 * dash_size), int(4 * dash_size)
        elif linestyle in ['dashdot', '-.']:
            return None, None
        else:
            # Default to solid
            return 0, 0

    def _filter_points(self, x, y):
        """
        Filter points based on show_all_points setting and current limits.
        Returns filtered x, y arrays.
        """
        if self.show_all_points:
            return x, y
        
        # Get current bounds - use actual graph limits, not data min/max
        if self.xlim:
            xmin, xmax = self.xlim
        else:
            xmin, xmax = float('-inf'), float('inf')
            
        if self.ylim:
            ymin, ymax = self.ylim
        else:
            ymin, ymax = float('-inf'), float('inf')
        
        filtered_x, filtered_y = [], []
        for xi, yi in zip(x, y):
            if xmin <= xi <= xmax and ymin <= yi <= ymax:
                filtered_x.append(xi)
                filtered_y.append(yi)
        
        return filtered_x, filtered_y

    def plot(self, x, y, color="blue", label=None, linewidth=4, linestyle="solid", dash_size=1.0, animate=1.0, delay = 0.0, drop_shadow=False, ease_speed=None, ease_influence=None, **kwargs):
        """
        Add a line plot to the graph.
        x, y: Data points (list, tuple, numpy array, or pandas Series).
        color: Color name or RGB list.
        label: Legend label.
        linewidth: Stroke width.
        linestyle: Line style - 'solid' (default), 'dashed'/'--', 'dotted'/':', or 'dashdot'/'-.'.
        dash_size: Scale factor for dash lengths (default: 1.0). Only applies to non-solid linestyles.
        animate: Animation duration in seconds.
        drop_shadow: Whether to add drop shadow effect (default: False).
        ease_speed: Optional per-element easy ease speed override. Uses AEGraph default when None.
        ease_influence: Optional per-element easy ease influence override. Uses AEGraph default when None.
        """
        if pd is not None:
            if isinstance(x, pd.Series):
                x = x.values
            if isinstance(y, pd.Series):
                y = y.values
        
        x, y = self._filter_points(x, y)
        
        self.elements.append({
            "type": "line",
            "x": list(x),
            "y": list(y),
            "color": color,
            "label": label,
            "linewidth": linewidth,
            "linestyle": linestyle,
            "dash_size": dash_size,
            "animate": animate,
            "drop_shadow": drop_shadow,
            "delay": delay,
            "ease_speed": ease_speed,
            "ease_influence": ease_influence,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def scatter(self, x, y, color="red", label=None, radius=8, delay = 0.0, animate=1.0, drop_shadow=False, bar_anim_times=None, ease_speed=None, ease_influence=None, **kwargs):
        """
        Add a scatter plot to the graph.
        x, y: Data points (list, tuple, numpy array, or pandas Series).
        color: Color name or RGB list.
        label: Legend label.
        radius: Point radius.
        animate: Total animation duration in seconds (points animate sequentially).
        drop_shadow: Whether to add drop shadow effect (default: False).
        bar_anim_times: Optional list of per-point animation durations (overrides sequential timing).
        ease_speed: Optional per-element easy ease speed override. Uses AEGraph default when None.
        ease_influence: Optional per-element easy ease influence override. Uses AEGraph default when None.
        """
        # Support pandas Series
        if pd is not None:
            if isinstance(x, pd.Series):
                x = x.values
            if isinstance(y, pd.Series):
                y = y.values
        
        # Filter points if needed
        x, y = self._filter_points(x, y)
        
        self.elements.append({
            "type": "scatter",
            "x": list(x),
            "y": list(y),
            "color": color,
            "label": label,
            "radius": radius,
            "animate": animate,
            "drop_shadow": drop_shadow,
            "bar_anim_times": bar_anim_times,
            "delay": delay,
            "ease_speed": ease_speed,
            "ease_influence": ease_influence,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def histogram(self, data, bins=10, color="p_blue", label=None, delay = 0.0, alpha=0.8, animate=1.0, drop_shadow=False, bar_anim_times=None, density=False, ease_speed=None, ease_influence=None, **kwargs):
        """
        Add a histogram to the graph.
        data: 1D array-like data to bin.
        bins: Number of bins or bin edges.
        color: Bar color.
        label: Legend label.
        alpha: Bar opacity (0-1).
        animate: Total animation duration in seconds (bars animate sequentially).
        drop_shadow: Whether to add drop shadow effect.
        bar_anim_times: Optional list of per-bar animation durations (overrides sequential timing).
        density: If True, normalize heights so the area under the histogram is 1 (PDF style); if False, heights are counts.
        ease_speed: Optional per-element easy ease speed override. Uses AEGraph default when None.
        ease_influence: Optional per-element easy ease influence override. Uses AEGraph default when None.
        """
        # Support pandas Series
        if pd is not None and isinstance(data, pd.Series):
            data = data.values
        data = np.asarray(data)
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_left = bin_edges[:-1]
        bin_right = bin_edges[1:]
        bin_centers = (bin_left + bin_right) / 2
        heights = counts
        if density:
            bin_widths = bin_right - bin_left
            total_area = np.sum(counts * bin_widths)
            if total_area > 0:
                heights = counts / total_area
            else:
                heights = counts
        self.elements.append({
            "type": "histogram",
            "bin_left": list(bin_left),
            "bin_right": list(bin_right),
            "bin_centers": list(bin_centers),
            "heights": list(heights),
            "color": color,
            "label": label,
            "alpha": alpha,
            "animate": animate,
            "drop_shadow": drop_shadow,
            "bar_anim_times": bar_anim_times,
            "density": density,
            "delay": delay,
            "ease_speed": ease_speed,
            "ease_influence": ease_influence,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def bar_graph(self, x_values, heights, bar_width=None, color="p_blue", label=None, alpha=0.8, animate=1.0, drop_shadow=False, bar_anim_times=None, ease_speed=None, ease_influence=None, **kwargs):
        """
        Add a bar graph to the plot using x values and corresponding heights.
        x_values: 1D array-like x positions for bars.
        heights: 1D array-like heights for bars (must match length of x_values).
        bar_width: Width of each bar in data coordinates. If None, auto-calculated from spacing.
        color: Bar color.
        label: Legend label.
        alpha: Bar opacity (0-1).
        animate: Total animation duration in seconds (bars animate sequentially).
        drop_shadow: Whether to add drop shadow effect.
        bar_anim_times: Optional list of per-bar animation durations (overrides sequential timing).
        ease_speed: Optional per-element easy ease speed override. Uses AEGraph default when None.
        ease_influence: Optional per-element easy ease influence override. Uses AEGraph default when None.
        """
        if pd is not None:
            if isinstance(x_values, pd.Series):
                x_values = x_values.values
            if isinstance(heights, pd.Series):
                heights = heights.values
        
        x_values = np.asarray(x_values)
        heights = np.asarray(heights)
        
        if len(x_values) != len(heights):
            raise ValueError("x_values and heights must have the same length")
        
        if bar_width is None:
            if len(x_values) > 1:
                # Use 80% of the minimum spacing between consecutive x values
                spacings = np.diff(np.sort(x_values))
                min_spacing = np.min(spacings[spacings > 0]) if len(spacings) > 0 and np.any(spacings > 0) else 1.0
                bar_width = 0.8 * min_spacing
            else:
                bar_width = 1.0  # Default for single bar
        
        # calculate bin edges for each bar (left and right edges)
        half_width = bar_width / 2
        bin_left = x_values - half_width
        bin_right = x_values + half_width
        bin_centers = x_values.copy()  # x_values are already the centers
        
        self.elements.append({
            "type": "bar_graph",
            "bin_left": list(bin_left),
            "bin_right": list(bin_right),
            "bin_centers": list(bin_centers),
            "heights": list(heights),
            "color": color,
            "label": label,
            "alpha": alpha,
            "animate": animate,
            "drop_shadow": drop_shadow,
            "bar_anim_times": bar_anim_times,
            "ease_speed": ease_speed,
            "ease_influence": ease_influence,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def barh(self, y_values, widths, bar_height=None, color="p_blue", label=None, delay=0.0, alpha=0.8, animate=1.0, drop_shadow=False, bar_duration=None, bar_anim_times=None, animate_downward: bool = False, anchor_at_y_axis: bool = False, ease_speed=None, ease_influence=None, **kwargs):
        """
        Add a horizontal bar graph to the plot (similar to matplotlib's barh).
        
        y_values: 1D array-like y positions for bars (vertical positions).
        widths: 1D array-like widths for bars (horizontal lengths, must match length of y_values).
        bar_height: Height (thickness) of each bar in data coordinates. If None, auto-calculated from spacing.
        color: Bar color.
        label: Legend label.
        alpha: Bar opacity (0-1).
        animate: Total animation duration in seconds (all bars complete within this time).
        drop_shadow: Whether to add drop shadow effect.
        bar_duration: Duration for each individual bar animation in seconds. If None, bars animate 
                      sequentially without overlap (each gets animate/n_bars seconds). If specified,
                      bars will overlap - e.g., animate=5.0 with bar_duration=1.0 creates smooth 
                      overlapping animations where each bar takes 1 second but spreads over 5 seconds total.
        bar_anim_times: (Deprecated, use bar_duration) Optional list of per-bar animation durations.
        animate_downward (bool): If True, bars animate from top y to bottom y
                     (downward). Default animates bottom-to-top.
        ease_speed: Optional per-element easy ease speed override. Uses AEGraph default when None.
        ease_influence: Optional per-element easy ease influence override. Uses AEGraph default when None.
        
        Example:
            # Smooth overlapping animations
            plot = AEGraph().barh(y_positions, values, animate=5.0, bar_duration=1.0)
            
            # Sequential animations (no overlap)
            plot = AEGraph().barh(y_positions, values, animate=5.0)
        """
        if pd is not None:
            if isinstance(y_values, pd.Series):
                y_values = y_values.values
            if isinstance(widths, pd.Series):
                widths = widths.values
        
        y_values = np.asarray(y_values)
        widths = np.asarray(widths)
        
        if len(y_values) != len(widths):
            raise ValueError("y_values and widths must have the same length")
        
        # auto-calculate bar height if not specified
        if bar_height is None:
            if len(y_values) > 1:
                # use 80% of the minimum spacing between consecutive y values
                spacings = np.diff(np.sort(y_values))
                min_spacing = np.min(spacings[spacings > 0]) if len(spacings) > 0 and np.any(spacings > 0) else 1.0
                bar_height = 0.8 * min_spacing
            else:
                bar_height = 1.0 # default
        
        # calculate bin edges for each bar (bottom and top edges)
        half_height = bar_height / 2
        bin_bottom = y_values - half_height
        bin_top = y_values + half_height
        bin_centers = y_values.copy()  # y_values are already the centers
        
        # use bar_duration if specified, otherwise fall back to bar_anim_times
        if bar_duration is not None:
            bar_anim_times = bar_duration
        
        self.elements.append({
            "type": "barh",
            "bin_bottom": list(bin_bottom),
            "bin_top": list(bin_top),
            "bin_centers": list(bin_centers),
            "widths": list(widths),
            "color": color,
            "label": label,
            "alpha": alpha,
            "animate": animate,
            "drop_shadow": drop_shadow,
            "bar_anim_times": bar_anim_times,
            "animate_downward": animate_downward,
            "anchor_at_y_axis": anchor_at_y_axis,
            "delay": delay,
            "ease_speed": ease_speed,
            "ease_influence": ease_influence,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def add_population_pyramid(self,
                               csv_path: Optional[str] = None,
                               ages: Optional[List[str]] = None,
                               male: Optional[List[float]] = None,
                               female: Optional[List[float]] = None,
                               mode: str = "percent",
                               animate: float = 5.0,
                               bar_duration: Optional[float] = 1.0,
                               animate_downward: bool = True,
                               show_grid: bool = True,
                               color_male: Union[str, List[float], Tuple[float, float, float]] = [73, 118, 222],
                               color_female: Union[str, List[float], Tuple[float, float, float]] = [168, 61, 104],
                               label_male: Optional[str] = "Male",
                               label_female: Optional[str] = "Female",
                               drop_shadow: Optional[bool] = None,
                               ):
        """
        Add a mirrored horizontal bar chart (population pyramid) to the current graph.

        Provide either `csv_path` to a file with three columns (age, male, female)
        or pass `ages`, `male`, and `female` arrays directly.

        mode: 'percent' to compute per-age-group share of total population (%),
              'counts' to use raw counts (auto-scaled if very large).

        Returns self for chaining.

        drop_shadow:
            - If True, apply the same Drop Shadow effect used for axes to the pyramid bars.
            - If False, do not add drop shadows to bars.
            - If None (default), inherit from the global `self.drop_shadow` setting.
        """
        if csv_path is not None:
            df = None
            if pd is not None:
                # Try common separators
                for sep, kwargs in [("\t", {}), (",", {}), (None, {"delim_whitespace": True})]:
                    try:
                        dft = pd.read_csv(csv_path, sep=sep, header=None, **kwargs)
                        if dft.shape[1] >= 3:
                            df = dft.iloc[:, :3]
                            df.columns = ["age", "male", "female"]
                            break
                    except Exception:
                        continue
            if df is None:
                rows = []
                with open(csv_path, "r", newline="") as f:
                    content = f.read().strip().splitlines()
                for line in content:
                    if not line.strip():
                        continue

                    for delim in ["\t", ","]:
                        if delim in line:
                            parts = [p.strip() for p in line.split(delim)]
                            break
                    else:
                        parts = line.split()
                    if len(parts) < 3:
                        continue
                    rows.append((parts[0], float(parts[1]), float(parts[2])))
                ages = [r[0] for r in rows]
                male = [r[1] for r in rows]
                female = [r[2] for r in rows]
            else:
                ages = df["age"].astype(str).tolist()
                male = df["male"].astype(float).tolist()
                female = df["female"].astype(float).tolist()
        else:
            if ages is None or male is None or female is None:
                raise ValueError("Provide either csv_path or ages/male/female arrays.")

        # Convert to numpy arrays
        male_arr = np.asarray(male, dtype=float)
        female_arr = np.asarray(female, dtype=float)
        n = len(ages)
        y_positions = np.arange(n)

        # Compute widths
        if mode == "counts":
            # scale down if its extremely large to keep x-range manageable
            max_val = float(max(np.max(male_arr), np.max(female_arr)))
            scale = 1.0
            if max_val > 1_000_000:
                scale = 1_000.0
            male_w = -male_arr / scale
            female_w = female_arr / scale
            xlabel = "Population (scaled)"
        else:
            total = float(male_arr.sum() + female_arr.sum())
            if total <= 0:
                raise ValueError("Total population is zero; cannot compute percentages")
            male_pct = (male_arr / total) * 100.0
            female_pct = (female_arr / total) * 100.0
            male_w = -male_pct
            female_w = female_pct
            xlabel = "Population (%)"

        max_abs = float(max(np.max(np.abs(male_w)), np.max(np.abs(female_w))))
        xpad = max_abs * 0.1
        xmin, xmax = -(max_abs + xpad), (max_abs + xpad)

        # Determine whether to add drop shadow to bars
        bar_shadow = self.drop_shadow if drop_shadow is None else bool(drop_shadow)

        # Add bars
        self.barh(
            y_positions,
            male_w,
            animate=animate,
            bar_duration=bar_duration,
            alpha=0.9,
            color=color_male,
            label=label_male,
            animate_downward=animate_downward,
            anchor_at_y_axis=True,
            drop_shadow=bar_shadow,
        )
        self.barh(
            y_positions,
            female_w,
            animate=animate,
            bar_duration=bar_duration,
            alpha=0.9,
            color=color_female,
            label=label_female,
            animate_downward=animate_downward,
            anchor_at_y_axis=True,
            drop_shadow=bar_shadow,
        )

        # Axes and labels
        self.set_xlabel(xlabel)
        self.set_yticks(positions=y_positions, labels=ages)
        self.set_xlim(xmin, xmax)
        # Ensure percent tick labels by default for pyramids in percent mode
        if mode == "percent":
            self.percent_tick_labels = True
        # Format x-ticks as absolute percent labels for population pyramids
        positions = self._nice_ticks(xmin, xmax, nticks=7)
        step = positions[1] - positions[0] if len(positions) > 1 else 1.0
        if abs(step) < 1:
            labels = [f"{abs(p):.1f}%" for p in positions]
        else:
            labels = [f"{abs(p):.0f}%" for p in positions]
        self.set_xticks(positions=positions, labels=labels)
        if show_grid:
            self.grid(show=True, color="lightgray", alpha=0.25)

        return self

    def gradient(self, color_start, color_end):
        """
        Apply a color gradient to the most recently added histogram, bar_graph, or barh.
        Supports named colors, 0–1 floats, or 0–255 ints.
        """
        if not self.elements:
            raise ValueError("No elements to apply gradient to. Add a histogram, bar_graph, or barh first.")

        # Get the last element
        last_elem = self.elements[-1]
        if last_elem["type"] not in ["histogram", "bar_graph", "barh"]:
            raise ValueError("Gradient can only be applied to histogram, bar_graph, or barh elements.")

        def normalize_color(c):
            if isinstance(c, str):
                rgb = COLOR_NAMES.get(c.lower())
                if rgb is None:
                    raise ValueError(f"Unknown color name: {c}")
                return rgb
            elif isinstance(c, (list, tuple)) and len(c) == 3:
                c = list(c)
                if max(c) > 1:  # assume 0–255 ints
                    return [v/255.0 for v in c]
                return c
            else:
                raise ValueError("Color must be a name or 3-value RGB list/tuple.")

        start_rgb = normalize_color(color_start)
        end_rgb = normalize_color(color_end)

        # Calculate number of bars
        if last_elem["type"] == "barh":
            n_bars = len(last_elem["widths"])
        else:
            n_bars = len(last_elem["heights"])
        if n_bars <= 1:
            last_elem["gradient_colors"] = [start_rgb]
        else:
            gradient_colors = []
            for i in range(n_bars):
                t = i / (n_bars - 1)  # interpolation factor
                interpolated_rgb = [
                    start_rgb[0] + t * (end_rgb[0] - start_rgb[0]),
                    start_rgb[1] + t * (end_rgb[1] - start_rgb[1]),
                    start_rgb[2] + t * (end_rgb[2] - start_rgb[2])
                ]
                gradient_colors.append(interpolated_rgb)
            last_elem["gradient_colors"] = gradient_colors

        return self


    def set_title(self, title: str):
        """Set the plot title."""
        self.title = title
        return self

    def set_subtitle(self, subtitle: str, color: Union[str, List[float], Tuple[float, float, float]] = None):
        """Set a subtitle shown below the title. Useful for sources or notes."""
        self.subtitle = subtitle
        self.subtitle_color = color
        return self

    def set_xlabel(self, label: str):
        """Set the x-axis label."""
        self.xlabel = label
        return self

    def set_ylabel(self, label: str):
        """Set the y-axis label."""
        self.ylabel = label
        return self

    def annotate(self, text: str, x: float, y: float, fontsize: int = 12, font: str = "Helvetica", alignment: str = "left"):
        """
        Add text annotation at specific coordinates on the graph.
        
        Args:
            text (str): The annotation text to display
            x (float): X coordinate in data space
            y (float): Y coordinate in data space
            fontsize (int): Font size (default: 12)
            font (str): Font name (default: "Helvetica")
            alignment (str): Text alignment - "left", "center", or "right" (default: "left")
        
        Returns:
            self: For method chaining
        """
        # Validate alignment parameter
        valid_alignments = ["left", "center", "right"]
        if alignment.lower() not in valid_alignments:
            raise ValueError(f"alignment must be one of {valid_alignments}, got '{alignment}'")
        
        # Store annotation in elements list for processing during JSX generation
        self.elements.append({
            "type": "annotation",
            "text": text,
            "x": x,
            "y": y,
            "fontsize": fontsize,
            "font": font,
            "alignment": alignment.lower()
        })
        return self

    def set_xlim(self, xmin, xmax):
        """Set the x-axis limits."""
        self.xlim = (xmin, xmax)
        return self

    def set_ylim(self, ymin, ymax):
        """Set the y-axis limits."""
        self.ylim = (ymin, ymax)
        return self

    def add_legend(self, style='color_only'):
        """
        Add a legend to the plot (auto from labels).
        
        Parameters:
        style (str): Legend display style. Options:
            - 'color_only': Show solid color swatches (default)
            - 'line_style': Show line samples with actual line styles (dashes, dots, etc.)
                           Useful for distinguishing lines with same color but different styles.
        """
        self.legend_style = style
        return self

    def set_xticks(self, positions=None, labels=None, nticks=7):
        """
        Set X-axis tick positions and labels.
        """
        if positions is None:
            # Collect x-like data from all elements
            all_x = []
            for elem in self.elements:
                if elem["type"] in ["line", "scatter"]:
                    all_x.extend(elem.get("x", []))
                elif elem["type"] in ["histogram", "bar_graph"]:
                    all_x.extend(elem.get("bin_left", []))
                    all_x.extend(elem.get("bin_right", []))
                    all_x.extend(elem.get("bin_centers", []))
                elif elem["type"] == "barh":
                    all_x.extend(elem.get("widths", []))
                    all_x.append(0)  # include baseline
            if not all_x:
                all_x = [0, 1]  # fallback if nothing plotted

            if self.xlim:
                xmin, xmax = self.xlim
            else:
                xmin, xmax = min(all_x), max(all_x)

            padding = 0.1
            x_range = xmax - xmin
            xmin -= x_range * padding
            xmax += x_range * padding
            positions = self._nice_ticks(xmin, xmax, nticks)

        if labels is None:
            if self.percent_tick_labels:
                step = positions[1] - positions[0] if len(positions) > 1 else 1.0
                if abs(step) < 1:
                    labels = [f"{abs(pos):.1f}%" for pos in positions]
                else:
                    labels = [f"{abs(pos):.0f}%" for pos in positions]
            else:
                labels = [str(pos) for pos in positions]

        self.xticks = list(zip(positions, labels))
        return self

    def set_yticks(self, positions=None, labels=None, nticks=7):
        """
        Set Y-axis tick positions and labels.
        """
        if positions is None:
            # Collect y-like data from all elements
            all_y = []
            for elem in self.elements:
                if elem["type"] in ["line", "scatter"]:
                    all_y.extend(elem.get("y", []))
                elif elem["type"] in ["histogram", "bar_graph"]:
                    all_y.extend(elem.get("heights", []))
                elif elem["type"] == "barh":
                    all_y.extend(elem.get("bin_bottom", []))
                    all_y.extend(elem.get("bin_top", []))
                    all_y.extend(elem.get("bin_centers", []))
            if not all_y:
                all_y = [0, 1]  # fallback if nothing plotted

            if self.ylim:
                ymin, ymax = self.ylim
            else:
                ymin, ymax = min(all_y), max(all_y)

            padding = 0.1
            y_range = ymax - ymin
            ymin -= y_range * padding
            ymax += y_range * padding
            positions = self._nice_ticks(ymin, ymax, nticks)

        if labels is None:
            labels = [str(pos) for pos in positions]

        self.yticks = list(zip(positions, labels))
        return self

    def grid(self, show=True, color="gray", alpha=0.3, linestyle="solid", dash_size=1.0, hide_horizontal=False, hide_vertical=False):
        """
        Enable or disable grid lines with customizable appearance.
        
        Args:
            show (bool): Whether to show grid (default: True)
            color (str or list): Grid color (default: "gray")
            alpha (float): Grid opacity 0-1 (default: 0.3)
            linestyle (str): Line style - 'solid' (default), 'dashed'/'--', 'dotted'/':', 
                           or 'dashdot'/'-.'. (default: "solid")
            dash_size (float): Scale factor for dash lengths (default: 1.0). Only applies to non-solid linestyles.
            hide_horizontal (bool): Hide horizontal grid lines (default: False)
            hide_vertical (bool): Hide vertical grid lines (default: False)
        """
        self.show_grid = show
        self.grid_color = color
        self.grid_alpha = alpha
        self.grid_linestyle = linestyle
        self.grid_dash_size = dash_size
        self.hide_horizontal = hide_horizontal
        self.hide_vertical = hide_vertical
        return self

    def set_tick_labels(self, show: bool = True):
        """Enable or disable tick labels entirely."""
        self.show_tick_labels = show
        return self

    def reset_comp(self):
        """
        Delete all layers in the active composition to start fresh.
        Returns self for method chaining.
        """
        script = []
        script.append("app.beginUndoGroup(\"Reset Composition\");\n")
        script.append("var comp = app.project.activeItem;\n")
        script.append("if (comp && comp instanceof CompItem) {\n")
        script.append("    while (comp.numLayers > 0) {\n")
        script.append("        comp.layer(1).remove();\n")
        script.append("    }\n")
        script.append("}\n")
        script.append("app.endUndoGroup();\n")
        # Execute the reset script immediately
        import subprocess
        import tempfile
        import os
        
        reset_jsx = "".join(script)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsx', delete=False) as f:
            f.write(reset_jsx)
            temp_file = f.name
        
        try:
            apple_script = f'''
            tell application "Adobe After Effects 2025"
                DoScriptFile "{temp_file}"
            end tell
            '''
            subprocess.run(["osascript", "-e", apple_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            os.unlink(temp_file)
        
        return self

    def _graph_to_comp(self, x, y):
        """
        Map graph logical coordinates (centered at 0,0, width/height) to comp pixel coordinates (centered in comp).
        """
        # Only translate, do not scale
        comp_cx = self.comp_width / 2
        comp_cy = self.comp_height / 2
        comp_x = x + comp_cx
        comp_y = y + comp_cy
        return comp_x, comp_y

    def _data_to_canvas(self, x, y, xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Convert data coordinates to graph logical pixel coordinates (width/height), then to comp pixel coordinates.
        """
        # Use provided limits or global limits
        if xmin is None or xmax is None:
            if self.xlim:
                xmin, xmax = self.xlim
            else:
                xmin, xmax = min(x), max(x)
        if ymin is None or ymax is None:
            if self.ylim:
                ymin, ymax = self.ylim
            else:
                ymin, ymax = min(y), max(y)
        # No padding
        xrange = xmax - xmin
        if xrange == 0:
            xrange = 1e-8
        yrange = ymax - ymin
        if yrange == 0:
            yrange = 1e-8
        # Map data to graph logical coordinates (origin at top-left)
        px = [self.width * (xi - xmin) / xrange for xi in x]
        py = [self.height - self.height * (yi - ymin) / yrange for yi in y]
        # Shift to graph center (origin at 0,0)
        px = [p - self.width/2 for p in px]
        py = [p - self.height/2 for p in py]
        # Map to comp coordinates
        comp_px, comp_py = zip(*[self._graph_to_comp(xg, yg) for xg, yg in zip(px, py)])
        return list(comp_px), list(comp_py)

    def _data_to_shape(self, x, y, xmin, xmax, ymin, ymax):
        """
        Convert data (x, y) to graph-local coordinates (centered at 0,0, width/height).
        """
        sx = self.width * (x - xmin) / (xmax - xmin) - self.width/2
        sy = self.height - self.height * (y - ymin) / (ymax - ymin) - self.height/2
        return sx, sy

    def _nice_ticks(self, vmin, vmax, nticks=7):
        """Generate nice tick positions between vmin and vmax."""
        if vmin == vmax:
            return [vmin]
        raw_step = (vmax - vmin) / max(1, nticks-1)
        mag = 10 ** math.floor(math.log10(abs(raw_step)))
        norm = raw_step / mag
        if norm < 1.5:
            step = 1 * mag
        elif norm < 3:
            step = 2 * mag
        elif norm < 7:
            step = 5 * mag
        else:
            step = 10 * mag
        tick_start = math.ceil(vmin / step) * step
        tick_end = math.floor(vmax / step) * step
        ticks = []
        v = tick_start
        while v <= tick_end + 1e-8:
            ticks.append(round(v, 10))
            v += step
        return ticks

    def _generate_drop_shadow_jsx(self, layer_name: str, index: str = "") -> str:
        """Generate JSX for drop shadow effect."""
        if not self.drop_shadow:
            return ""
        
        jsx = []
        jsx.append(f"var fx{index} = {layer_name}.property('Effects').addProperty('ADBE Drop Shadow');\n")
        jsx.append(f"fx{index}.property('Direction').setValue({DEFAULT_DROP_SHADOW['direction']});\n")
        jsx.append(f"fx{index}.property('Distance').setValue({DEFAULT_DROP_SHADOW['distance']});\n")
        jsx.append(f"fx{index}.property('Softness').setValue({DEFAULT_DROP_SHADOW['softness']});\n")
        jsx.append(f"fx{index}.property('Shadow Color').setValue({color_to_js(DEFAULT_DROP_SHADOW['color'])});\n")
        jsx.append(f"fx{index}.property('Opacity').setValue({255 * DEFAULT_DROP_SHADOW['opacity']});\n")
        
        return "".join(jsx)

    def _generate_axes_jsx(self, script, center_x, center_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad, ANIM_DURATION, has_barh=False):
        """Helper function to generate axes and ticks JSX code."""
        # Axes: y-axis at x=0 if in range, else at left; x-axis at y=0 if in range, else at bottom
        # For barh charts, always put x-axis at bottom (ymin_pad) so it doesn't intersect bars
        y_axis_x = 0 if xmin_pad <= 0 <= xmax_pad else xmin_pad
        if has_barh:
            x_axis_y = ymin_pad  # Always at bottom for horizontal bar charts
        else:
            x_axis_y = 0 if ymin_pad <= 0 <= ymax_pad else ymin_pad
        # Clamp axis reference values to visible range for tick placement
        y_axis_x = min(max(y_axis_x, xmin_pad), xmax_pad)
        x_axis_y = min(max(x_axis_y, ymin_pad), ymax_pad)
        # Axes endpoints in data coordinates
        x_axis_start = (xmin_pad, x_axis_y)
        x_axis_end = (xmax_pad, x_axis_y)
        y_axis_start = (y_axis_x, ymin_pad)
        y_axis_end = (y_axis_x, ymax_pad)
        # Convert to shape coordinates
        x0s, y0s = self._data_to_shape(*x_axis_start, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
        x1s, y1s = self._data_to_shape(*x_axis_end, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
        yx0s, yy0s = self._data_to_shape(*y_axis_start, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
        yx1s, yy1s = self._data_to_shape(*y_axis_end, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
        # X axis
        script.append("var axesLayer = comp.layers.addShape();\n")
        script.append("axesLayer.name = \"Axes\";\n")
        script.append(f"axesLayer.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
        script.append(f"axesLayer.parent = PlotAnchor;\n")
        script.append("var axesContents = axesLayer.property('ADBE Root Vectors Group');\n")
        script.append("var axesPathGroup = axesContents.addProperty('ADBE Vector Shape - Group');\n")
        script.append("var axesPath = axesPathGroup.property('ADBE Vector Shape');\n")
        script.append("var axesShape = new Shape();\n")
        script.append(f"axesShape.vertices = [[{x0s}, {y0s}], [{x1s}, {y1s}]];\n")
        script.append("axesShape.closed = false;\n")
        script.append("axesPath.setValue(axesShape);\n")
        script.append("var axesStroke = axesContents.addProperty('ADBE Vector Graphic - Stroke');\n")
        script.append(f"axesStroke.property('ADBE Vector Stroke Color').setValue({color_to_js(self.ui_color)});\n")
        script.append("axesStroke.property('ADBE Vector Stroke Width').setValue(3);\n")
        script.append("axesStroke.property('ADBE Vector Stroke Opacity').setValue(100);\n")
        # Animate axes with Trim Paths (optional)
        if self.animate_axes:
            script.append("var axesTrim = axesContents.addProperty('ADBE Vector Filter - Trim');\n")
            script.append(f"var axesTrimEnd = axesTrim.property('ADBE Vector Trim End');\n")
            script.append(f"axesTrimEnd.setValueAtTime(0, 0);\n")
            script.append(f"axesTrimEnd.setValueAtTime({ANIM_DURATION}, 100);\n")
            # Apply easy ease to axes trim path keyframes
            if self.easy_ease:
                script.append(f"applyEasyEase(axesTrimEnd, {self.ease_speed}, {self.ease_influence});\n")
        # Y axis
        script.append("var yAxisLayer = comp.layers.addShape();\n")
        script.append("yAxisLayer.name = \"Y-Axis\";\n")
        script.append(f"yAxisLayer.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
        script.append(f"yAxisLayer.parent = PlotAnchor;\n")
        script.append("var yAxisContents = yAxisLayer.property('ADBE Root Vectors Group');\n")
        script.append("var yAxisPathGroup = yAxisContents.addProperty('ADBE Vector Shape - Group');\n")
        script.append("var yAxisPath = yAxisPathGroup.property('ADBE Vector Shape');\n")
        script.append("var yAxisShape = new Shape();\n")
        script.append(f"yAxisShape.vertices = [[{yx0s}, {yy0s}], [{yx1s}, {yy1s}]];\n")
        script.append("yAxisShape.closed = false;\n")
        script.append("yAxisPath.setValue(yAxisShape);\n")
        script.append("var yAxisStroke = yAxisContents.addProperty('ADBE Vector Graphic - Stroke');\n")
        script.append(f"yAxisStroke.property('ADBE Vector Stroke Color').setValue({color_to_js(self.ui_color)});\n")
        script.append("yAxisStroke.property('ADBE Vector Stroke Width').setValue(3);\n")
        script.append("yAxisStroke.property('ADBE Vector Stroke Opacity').setValue(100);\n")
        # Animate y axis with Trim Paths (optional)
        if self.animate_axes:
            script.append("var yAxisTrim = yAxisContents.addProperty('ADBE Vector Filter - Trim');\n")
            script.append(f"var yAxisTrimEnd = yAxisTrim.property('ADBE Vector Trim End');\n")
            script.append(f"yAxisTrimEnd.setValueAtTime(0, 0);\n")
            script.append(f"yAxisTrimEnd.setValueAtTime({ANIM_DURATION}, 100);\n")
            # Apply easy ease to Y axis trim path keyframes
            if self.easy_ease:
                script.append(f"applyEasyEase(yAxisTrimEnd, {self.ease_speed}, {self.ease_influence});\n")
        
        # Add drop shadow to axes if specified
        if self.drop_shadow:
            script.append(self._generate_drop_shadow_jsx("axesLayer", "Axes"))
            script.append(self._generate_drop_shadow_jsx("yAxisLayer", "YAxis"))

        # X Ticks
        if self.xticks:
            for idx, (pos, label) in enumerate(self.xticks):
                pos_var = str(pos).replace('-', 'm').replace('.', '_')
                # Get shape coordinates for tick mark
                tick_xs, tick_ys0 = self._data_to_shape(pos, x_axis_y-8*(ymax_pad-ymin_pad)/self.height, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                _, tick_ys1 = self._data_to_shape(pos, x_axis_y+8*(ymax_pad-ymin_pad)/self.height, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                # Draw tick mark as vertical line in shape coordinates
                script.append(f"var xtickLayer{pos_var} = comp.layers.addShape();\n")
                script.append(f"xtickLayer{pos_var}.name = \"XTick_{pos}\";\n")
                script.append(f"xtickLayer{pos_var}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"xtickLayer{pos_var}.parent = PlotAnchor;\n")
                script.append(f"var xtickContents{pos_var} = xtickLayer{pos_var}.property('ADBE Root Vectors Group');\n")
                script.append(f"var xtickPathGroup{pos_var} = xtickContents{pos_var}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var xtickPath{pos_var} = xtickPathGroup{pos_var}.property('ADBE Vector Shape');\n")
                script.append(f"var xtickShape{pos_var} = new Shape();\n")
                script.append(f"xtickShape{pos_var}.vertices = [[{tick_xs}, {tick_ys0}], [{tick_xs}, {tick_ys1}]];\n")
                script.append(f"xtickShape{pos_var}.closed = false;\n")
                script.append(f"xtickPath{pos_var}.setValue(xtickShape{pos_var});\n")
                script.append(f"var xtickStroke{pos_var} = xtickContents{pos_var}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"xtickStroke{pos_var}.property('ADBE Vector Stroke Color').setValue({color_to_js(self.ui_color)});\n")
                script.append(f"xtickStroke{pos_var}.property('ADBE Vector Stroke Width').setValue(2);\n")
                # Opacity animation for tick mark (constant duration or disabled)
                if self.animate_opacity:
                    script.append(f"xtickStroke{pos_var}.property('ADBE Vector Stroke Opacity').setValueAtTime(0, 0);\n")
                    script.append(f"xtickStroke{pos_var}.property('ADBE Vector Stroke Opacity').setValueAtTime({ANIM_DURATION * 0.8}, 100);\n")
                    if self.easy_ease:
                        script.append(f"applyEasyEase(xtickStroke{pos_var}.property('ADBE Vector Stroke Opacity'), {self.ease_speed}, {self.ease_influence});\n")
                else:
                    script.append(f"xtickStroke{pos_var}.property('ADBE Vector Stroke Opacity').setValue(100);\n")
                # Tick label: just below the tick, in graph-local coordinates
                if self.show_tick_labels:
                    lx, ly = self._data_to_shape(pos, x_axis_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    ly += 30
                    script.append(f"var xtickLabel{pos_var} = comp.layers.addText(\"{label}\");\n")
                    script.append(f"xtickLabel{pos_var}.property('Transform').property('Position').setValue([{center_x + lx}, {center_y + ly}]);\n")
                    script.append(f"xtickLabel{pos_var}.parent = PlotAnchor;\n")
                    script.append(f"var xtickLabelProp{pos_var} = xtickLabel{pos_var}.property('Source Text');\n")
                    script.append(f"var xtickLabelDoc{pos_var} = xtickLabelProp{pos_var}.value;\n")
                    script.append(f"xtickLabelDoc{pos_var}.fontSize = {int(27 * self.font_scale)};\n")
                    script.append(f"xtickLabelDoc{pos_var}.fillColor = {color_to_js(self.ui_color)};\n")
                    script.append(f"xtickLabelDoc{pos_var}.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
                    script.append(f"xtickLabelProp{pos_var}.setValue(xtickLabelDoc{pos_var});\n")
                    script.append(f"var xsr{pos_var} = xtickLabel{pos_var}.sourceRectAtTime(0, false);\n")
                    script.append(f"var xap{pos_var} = xtickLabel{pos_var}.property('Transform').property('Anchor Point');\n")
                    script.append(f"xap{pos_var}.setValue([xsr{pos_var}.left + xsr{pos_var}.width/2, xsr{pos_var}.top]);\n")
                    # Opacity animation for label (constant duration or disabled)
                    if self.animate_opacity:
                        script.append(f"xtickLabel{pos_var}.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                        script.append(f"xtickLabel{pos_var}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 0.9}, 100);\n")
                        if self.easy_ease:
                            script.append(f"applyEasyEase(xtickLabel{pos_var}.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
                    else:
                        script.append(f"xtickLabel{pos_var}.property('Transform').property('Opacity').setValue(100);\n")
        # Y Ticks
        if self.yticks:
            for idx, (pos, label) in enumerate(self.yticks):
                pos_var = str(pos).replace('-', 'm').replace('.', '_')
                # Get shape coordinates for tick mark
                tick_xs0, tick_ys = self._data_to_shape(y_axis_x-8*(xmax_pad-xmin_pad)/self.width, pos, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                tick_xs1, _ = self._data_to_shape(y_axis_x+8*(xmax_pad-xmin_pad)/self.width, pos, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                # Draw tick mark as horizontal line in shape coordinates
                script.append(f"var ytickLayer{pos_var} = comp.layers.addShape();\n")
                script.append(f"ytickLayer{pos_var}.name = \"YTick_{pos}\";\n")
                script.append(f"ytickLayer{pos_var}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"ytickLayer{pos_var}.parent = PlotAnchor;\n")
                script.append(f"var ytickContents{pos_var} = ytickLayer{pos_var}.property('ADBE Root Vectors Group');\n")
                script.append(f"var ytickPathGroup{pos_var} = ytickContents{pos_var}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var ytickPath{pos_var} = ytickPathGroup{pos_var}.property('ADBE Vector Shape');\n")
                script.append(f"var ytickShape{pos_var} = new Shape();\n")
                script.append(f"ytickShape{pos_var}.vertices = [[{tick_xs0}, {tick_ys}], [{tick_xs1}, {tick_ys}]];\n")
                script.append(f"ytickShape{pos_var}.closed = false;\n")
                script.append(f"ytickPath{pos_var}.setValue(ytickShape{pos_var});\n")
                script.append(f"var ytickStroke{pos_var} = ytickContents{pos_var}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"ytickStroke{pos_var}.property('ADBE Vector Stroke Color').setValue({color_to_js(self.ui_color)});\n")
                script.append(f"ytickStroke{pos_var}.property('ADBE Vector Stroke Width').setValue(2);\n")
                # Opacity animation for tick mark (constant duration or disabled)
                if self.animate_opacity:
                    script.append(f"ytickStroke{pos_var}.property('ADBE Vector Stroke Opacity').setValueAtTime(0, 0);\n")
                    script.append(f"ytickStroke{pos_var}.property('ADBE Vector Stroke Opacity').setValueAtTime({ANIM_DURATION * 0.8}, 100);\n")
                    if self.easy_ease:
                        script.append(f"applyEasyEase(ytickStroke{pos_var}.property('ADBE Vector Stroke Opacity'), {self.ease_speed}, {self.ease_influence});\n")
                else:
                    script.append(f"ytickStroke{pos_var}.property('ADBE Vector Stroke Opacity').setValue(100);\n")
                # Tick label: just left of the tick, in graph-local coordinates
                if self.show_tick_labels:
                    lx, ly = self._data_to_shape(y_axis_x, pos, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    lx -= 30
                    script.append(f"var ytickLabel{pos_var} = comp.layers.addText(\"{label}\");\n")
                    script.append(f"ytickLabel{pos_var}.property('Transform').property('Position').setValue([{center_x + lx}, {center_y + ly}]);\n")
                    script.append(f"ytickLabel{pos_var}.parent = PlotAnchor;\n")
                    script.append(f"var ytickLabelProp{pos_var} = ytickLabel{pos_var}.property('Source Text');\n")
                    script.append(f"var ytickLabelDoc{pos_var} = ytickLabelProp{pos_var}.value;\n")
                    script.append(f"ytickLabelDoc{pos_var}.fontSize = {int(27 * self.font_scale)};\n")
                    script.append(f"ytickLabelDoc{pos_var}.fillColor = {color_to_js(self.ui_color)};\n")
                    script.append(f"ytickLabelDoc{pos_var}.justification = ParagraphJustification.RIGHT_JUSTIFY;\n")
                    script.append(f"ytickLabelProp{pos_var}.setValue(ytickLabelDoc{pos_var});\n")
                    script.append(f"var ysr{pos_var} = ytickLabel{pos_var}.sourceRectAtTime(0, false);\n")
                    script.append(f"var yap{pos_var} = ytickLabel{pos_var}.property('Transform').property('Anchor Point');\n")
                    script.append(f"yap{pos_var}.setValue([ysr{pos_var}.left + ysr{pos_var}.width, ysr{pos_var}.top + ysr{pos_var}.height/2]);\n")
                    # Opacity animation for label (constant duration or disabled)
                    if self.animate_opacity:
                        script.append(f"ytickLabel{pos_var}.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                        script.append(f"ytickLabel{pos_var}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 0.9}, 100);\n")
                        if self.easy_ease:
                            script.append(f"applyEasyEase(ytickLabel{pos_var}.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
                    else:
                        script.append(f"ytickLabel{pos_var}.property('Transform').property('Opacity').setValue(100);\n")

    def _generate_jsx(self) -> str:
        ANIM_DURATION = 3 * (1/2)  # seconds for default animation
        script = []
        script.append("app.beginUndoGroup(\"AEGraph Render\");\n")
        
        # Add Easy Ease Helper Function
        if self.easy_ease:
            script.append("""
// --- Helper Function: Apply Easy Ease to All Keyframes of a Property ---
function applyEasyEase(prop, easeSpeed, easeInfluence) {
    if (!prop || !prop.numKeys || prop.numKeys < 2) return;
    
    // Determine the number of dimensions
    var dim = prop.value.length ? prop.value.length : 1;
    
    // Create KeyframeEase array per dimension
    var easeInArray = [];
    var easeOutArray = [];
    for (var d = 0; d < dim; d++) {
        easeInArray.push(new KeyframeEase(easeSpeed, easeInfluence));
        easeOutArray.push(new KeyframeEase(easeSpeed, easeInfluence));
    }
    
    // Apply to all keyframes
    for (var k = 1; k <= prop.numKeys; k++) {
        prop.setTemporalEaseAtKey(k, easeInArray, easeOutArray);
    }
}

""")
        
        script.append(
            f"""
var comp = app.project.activeItem;
if (!comp || !(comp instanceof CompItem) || comp.name != '{self.comp_name}') {{
    comp = app.project.items.addComp('{self.comp_name}', {self.comp_width}, {self.comp_height}, 1, 10, {self.fps});
    comp.openInViewer();
}}
"""
        )
        # Remove comp background color setting - let it be transparent
        # script.append(f"comp.bgColor = {color_to_js(self.bg_color)};\n")

        # --- PlotAnchor Null Layer ---
        # Always place anchor at comp center by default, or user-specified position
        anchor_x = self.position[0] if self.position is not None else self.comp_width / 2
        anchor_y = self.position[1] if self.position is not None else self.comp_height / 2
        script.append(f"var PlotAnchor = comp.layers.addNull();\n")
        script.append(f"PlotAnchor.name = 'PlotAnchor';\n")
        script.append(f"PlotAnchor.property('Transform').property('Position').setValue([{anchor_x}, {anchor_y}]);\n")
        script.append(f"PlotAnchor.moveToBeginning();\n")

        # All graph elements (background, axes, grid, etc.) are drawn at the comp center
        center_x = self.comp_width / 2
        center_y = self.comp_height / 2
        distress_texture_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "distress_textures", "grunge1.jpg")
        ).replace("\\", "/").replace('"', '\\"')
        # Optional background rectangle (skipped when bg_color == 'none')
        if not (isinstance(self.bg_color, str) and self.bg_color.lower() == "none"):
            script.append(f"var bgLayer = comp.layers.addShape();\n")
            script.append(f"bgLayer.name = 'GraphBG';\n")
            script.append(f"var bgContents = bgLayer.property('ADBE Root Vectors Group');\n")
            script.append(f"var bgRect = bgContents.addProperty('ADBE Vector Shape - Rect');\n")
            # Use full composition size if full_bg is True, otherwise use graph size
            bg_width = self.comp_width if self.full_bg else self.width
            bg_height = self.comp_height if self.full_bg else self.height
            script.append(f"bgRect.property('ADBE Vector Rect Size').setValue([{bg_width}, {bg_height}]);\n")
            script.append(f"bgRect.property('ADBE Vector Rect Position').setValue([0, 0]);\n")
            script.append(f"var bgFill = bgContents.addProperty('ADBE Vector Graphic - Fill');\n")
            script.append(f"bgFill.property('ADBE Vector Fill Color').setValue({color_to_js(self.bg_color)});\n")
            # Add stroke to background rectangle
            script.append(f"var bgStroke = bgContents.addProperty('ADBE Vector Graphic - Stroke');\n")
            script.append(f"bgStroke.property('ADBE Vector Stroke Color').setValue({color_to_js(self.bg_stroke_color)});\n")
            script.append(f"bgStroke.property('ADBE Vector Stroke Width').setValue({self.bg_stroke_width});\n")
            script.append(f"bgLayer.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
            script.append(f"bgLayer.parent = PlotAnchor;\n")

        if self.distress_texture == 1:
            script.append(f"var distressFile = new File(\"{distress_texture_path}\");\n")
            script.append("if (distressFile.exists) {\n")
            script.append("    var distressFootage = null;\n")
            script.append("    var distressPath = distressFile.fsName.toLowerCase();\n")
            script.append("    for (var i = 1; i <= app.project.numItems; i++) {\n")
            script.append("        var projectItem = app.project.item(i);\n")
            script.append("        if (projectItem instanceof FootageItem && projectItem.mainSource && projectItem.mainSource.file) {\n")
            script.append("            if (projectItem.mainSource.file.fsName.toLowerCase() === distressPath) {\n")
            script.append("                distressFootage = projectItem;\n")
            script.append("                break;\n")
            script.append("            }\n")
            script.append("        }\n")
            script.append("    }\n")
            script.append("    if (!distressFootage) {\n")
            script.append("        var distressImportOptions = new ImportOptions(distressFile);\n")
            script.append("        distressFootage = app.project.importFile(distressImportOptions);\n")
            script.append("    }\n")
            script.append("    var distressLayer = comp.layers.add(distressFootage);\n")
            script.append("    distressLayer.name = 'grunge1.jpg';\n")
            script.append(f"    distressLayer.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
            script.append("    if (distressLayer.source && distressLayer.source.width > 0 && distressLayer.source.height > 0) {\n")
            script.append(f"        var distressScale = Math.max(({self.comp_width} / distressLayer.source.width) * 100, ({self.comp_height} / distressLayer.source.height) * 100);\n")
            script.append("        distressLayer.property('Transform').property('Scale').setValue([distressScale, distressScale]);\n")
            script.append("    }\n")
            script.append("    distressLayer.property('Transform').property('Opacity').setValue(10);\n")
            script.append("    distressLayer.blendingMode = BlendingMode.OVERLAY;\n")
            script.append("    if (typeof bgLayer !== 'undefined') { distressLayer.moveBefore(bgLayer); }\n")
            script.append("    distressLayer.parent = PlotAnchor;\n")
            script.append("}\n")

        # In _generate_jsx, calculate global data limits with NO padding
        all_x = []
        all_y = []
        for elem in self.elements:
            if elem["type"] in ["line", "scatter"]:
                all_x.extend(elem["x"])
                all_y.extend(elem["y"])
            elif elem["type"] in ["histogram", "bar_graph"]:
                # Use bin edges for x, heights for y
                all_x.extend(elem["bin_centers"])
                all_x.extend(elem["bin_left"])
                all_x.extend(elem["bin_right"])
                all_y.extend(elem["heights"])
                all_y.append(0)  # include baseline
            elif elem["type"] == "barh":
                # Use widths for x, bin edges for y
                all_x.extend(elem["widths"])
                all_x.append(0)  # include baseline
                all_y.extend(elem["bin_centers"])
                all_y.extend(elem["bin_bottom"])
                all_y.extend(elem["bin_top"])
                # Add padding equal to the gap between bars
                if len(elem["bin_bottom"]) > 1:
                    # Calculate the gap between bars from the bar dimensions
                    centers = sorted(elem["bin_centers"])
                    spacing = centers[1] - centers[0]  # distance between consecutive bar centers
                    bar_height = elem["bin_top"][0] - elem["bin_bottom"][0]  # height of one bar
                    gap = spacing - bar_height  # gap between bars
                    # Add padding equal to one gap
                    min_y = min(elem["bin_bottom"])
                    max_y = max(elem["bin_top"])
                    all_y.append(min_y - gap)
                    all_y.append(max_y + gap)
        if self.xlim:
            xmin, xmax = self.xlim
        else:
            xmin, xmax = min(all_x), max(all_x)
        if self.ylim:
            ymin, ymax = self.ylim
        else:
            ymin, ymax = min(all_y), max(all_y)
        # No padding
        xmin_pad = xmin
        xmax_pad = xmax
        ymin_pad = ymin
        ymax_pad = ymax
        # Check what plot types we have to decide on axes placement
        has_scatter = any(elem["type"] == "scatter" for elem in self.elements)
        has_bars_or_hist = any(elem["type"] in ["histogram", "bar_graph", "barh"] for elem in self.elements)
        has_barh = any(elem["type"] == "barh" for elem in self.elements)
        
        # For grid lines, fade in with opacity (GENERATED FIRST so grid appears behind plot elements)
        if self.show_grid:
            grid_color_js = color_to_js(self.grid_color)
            
            # Vertical grid lines
            if not self.hide_vertical:
                for idx, i in enumerate(self._nice_ticks(xmin_pad, xmax_pad)):
                    xg, yg0 = self._data_to_shape(i, ymin_pad, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    xg, yg1 = self._data_to_shape(i, ymax_pad, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    grid_var = str(i).replace('-', 'm').replace('.', '_');
                    script.append(f"var gridV{grid_var} = comp.layers.addShape();\n")
                    script.append(f"gridV{grid_var}.name = \"Grid_V_{i}\";\n")
                    script.append(f"gridV{grid_var}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                    script.append(f"gridV{grid_var}.parent = PlotAnchor;\n")
                    script.append(f"var gridVContents{grid_var} = gridV{grid_var}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var gridVPathGroup{grid_var} = gridVContents{grid_var}.addProperty('ADBE Vector Shape - Group');\n")
                    script.append(f"var gridVPath{grid_var} = gridVPathGroup{grid_var}.property('ADBE Vector Shape');\n")
                    script.append(f"var gridVShape{grid_var} = new Shape();\n")
                    script.append(f"gridVShape{grid_var}.vertices = [[{xg}, {yg0}], [{xg}, {yg1}]];\n")
                    script.append(f"gridVShape{grid_var}.closed = false;\n")
                    script.append(f"gridVPath{grid_var}.setValue(gridVShape{grid_var});\n")
                    script.append(f"var gridVStroke{grid_var} = gridVContents{grid_var}.addProperty('ADBE Vector Graphic - Stroke');\n")
                    script.append(f"gridVStroke{grid_var}.property('ADBE Vector Stroke Color').setValue({grid_color_js});\n")
                    script.append(f"gridVStroke{grid_var}.property('ADBE Vector Stroke Width').setValue(1);\n")
                    # Apply linestyle (dashes) to vertical grid with dash_size scaling
                    if self.grid_linestyle in ["dashed", "--"]:
                        dash_val, gap_val = self._get_dash_values(self.grid_linestyle, self.grid_dash_size)
                        script.append(f"gridVStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({dash_val});\n")
                        script.append(f"gridVStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({gap_val});\n")
                    elif self.grid_linestyle in ["dotted", ":"]:
                        dash_val, gap_val = self._get_dash_values(self.grid_linestyle, self.grid_dash_size)
                        script.append(f"gridVStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({dash_val});\n")
                        script.append(f"gridVStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({gap_val});\n")
                    elif self.grid_linestyle in ["dashdot", "-."]:
                        dash_val, gap_val = self._get_dash_values("dashed", self.grid_dash_size)
                        dot_val, dot_gap = self._get_dash_values("dotted", self.grid_dash_size)
                        script.append(f"gridVStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({dash_val});\n")
                        script.append(f"gridVStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({gap_val});\n")
                        script.append(f"gridVStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Dash 2').setValue({dot_val});\n")
                        script.append(f"gridVStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Gap 2').setValue({dot_gap});\n")
                    script.append(f"gridVStroke{grid_var}.property('ADBE Vector Stroke Opacity').setValue(0);\n")
                    # Animate opacity in
                    script.append(f"gridVStroke{grid_var}.property('ADBE Vector Stroke Opacity').setValueAtTime(0, 0);\n")
                    script.append(f"gridVStroke{grid_var}.property('ADBE Vector Stroke Opacity').setValueAtTime({ANIM_DURATION * (0.5 + 0.5 * idx / 10)}, {int(self.grid_alpha * 100)});\n")
                    # Apply easy ease to grid V opacity keyframes
                    if self.easy_ease:
                        script.append(f"applyEasyEase(gridVStroke{grid_var}.property('ADBE Vector Stroke Opacity'), {self.ease_speed}, {self.ease_influence});\n")
            
            # Horizontal grid lines
            if not self.hide_horizontal:
                for idx, i in enumerate(self._nice_ticks(ymin_pad, ymax_pad)):
                    xg0, yg = self._data_to_shape(xmin_pad, i, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    xg1, yg = self._data_to_shape(xmax_pad, i, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    grid_var = str(i).replace('-', 'm').replace('.', '_');
                    script.append(f"var gridH{grid_var} = comp.layers.addShape();\n")
                    script.append(f"gridH{grid_var}.name = \"Grid_H_{i}\";\n")
                    script.append(f"gridH{grid_var}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                    script.append(f"gridH{grid_var}.parent = PlotAnchor;\n")
                    script.append(f"var gridHContents{grid_var} = gridH{grid_var}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var gridHPathGroup{grid_var} = gridHContents{grid_var}.addProperty('ADBE Vector Shape - Group');\n")
                    script.append(f"var gridHPath{grid_var} = gridHPathGroup{grid_var}.property('ADBE Vector Shape');\n")
                    script.append(f"var gridHShape{grid_var} = new Shape();\n")
                    script.append(f"gridHShape{grid_var}.vertices = [[{xg0}, {yg}], [{xg1}, {yg}]];\n")
                    script.append(f"gridHShape{grid_var}.closed = false;\n")
                    script.append(f"gridHPath{grid_var}.setValue(gridHShape{grid_var});\n")
                    script.append(f"var gridHStroke{grid_var} = gridHContents{grid_var}.addProperty('ADBE Vector Graphic - Stroke');\n")
                    script.append(f"gridHStroke{grid_var}.property('ADBE Vector Stroke Color').setValue({grid_color_js});\n")
                    script.append(f"gridHStroke{grid_var}.property('ADBE Vector Stroke Width').setValue(1);\n")
                    # Apply linestyle (dashes) to horizontal grid with dash_size scaling
                    if self.grid_linestyle in ["dashed", "--"]:
                        dash_val, gap_val = self._get_dash_values(self.grid_linestyle, self.grid_dash_size)
                        script.append(f"gridHStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({dash_val});\n")
                        script.append(f"gridHStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({gap_val});\n")
                    elif self.grid_linestyle in ["dotted", ":"]:
                        dash_val, gap_val = self._get_dash_values(self.grid_linestyle, self.grid_dash_size)
                        script.append(f"gridHStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({dash_val});\n")
                        script.append(f"gridHStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({gap_val});\n")
                    elif self.grid_linestyle in ["dashdot", "-."]:
                        dash_val, gap_val = self._get_dash_values("dashed", self.grid_dash_size)
                        dot_val, dot_gap = self._get_dash_values("dotted", self.grid_dash_size)
                        script.append(f"gridHStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({dash_val});\n")
                        script.append(f"gridHStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({gap_val});\n")
                        script.append(f"gridHStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Dash 2').setValue({dot_val});\n")
                        script.append(f"gridHStroke{grid_var}.property('Dashes').addProperty('ADBE Vector Stroke Gap 2').setValue({dot_gap});\n")
                    script.append(f"gridHStroke{grid_var}.property('ADBE Vector Stroke Opacity').setValue(0);\n")
                    # Animate opacity in
                    script.append(f"gridHStroke{grid_var}.property('ADBE Vector Stroke Opacity').setValueAtTime(0, 0);\n")
                    script.append(f"gridHStroke{grid_var}.property('ADBE Vector Stroke Opacity').setValueAtTime({ANIM_DURATION * (0.5 + 0.5 * idx / 10)}, {int(self.grid_alpha * 100)});\n")
                    # Apply easy ease to grid H opacity keyframes
                    if self.easy_ease:
                        script.append(f"applyEasyEase(gridHStroke{grid_var}.property('ADBE Vector Stroke Opacity'), {self.ease_speed}, {self.ease_influence});\n")
        
        # Generate axes BEFORE plot elements ONLY for scatter plots (so axes appear behind points)
        # For histograms and bar graphs, axes will be generated AFTER plot elements (so they appear on top)
        if has_scatter and not has_bars_or_hist:
            self._generate_axes_jsx(script, center_x, center_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad, ANIM_DURATION, has_barh)

        # Plot elements
        for i, elem in enumerate(self.elements):
            delay = elem.get("delay", 0.0)
            # Use the same data limits for all mapping
            if elem["type"] == "line":
                px, py = elem["x"], elem["y"]
                shape_px, shape_py = zip(*[self._data_to_shape(x, y, xmin_pad, xmax_pad, ymin_pad, ymax_pad) for x, y in zip(px, py)])
                points_js = ",".join(f"[{x},{y}]" for x, y in zip(shape_px, shape_py))
                color_js = color_to_js(elem["color"])
                elem_ease_speed = self.ease_speed if elem.get("ease_speed") is None else elem.get("ease_speed")
                elem_ease_influence = self.ease_influence if elem.get("ease_influence") is None else elem.get("ease_influence")
                script.append(f"var lineLayer{i} = comp.layers.addShape();\n")
                script.append(f"lineLayer{i}.name = \"Line_{i}\";\n")
                script.append(f"lineLayer{i}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"lineLayer{i}.parent = PlotAnchor;\n")
                script.append(f"var contents{i} = lineLayer{i}.property('ADBE Root Vectors Group');\n")
                script.append(f"var pathGroup{i} = contents{i}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var path{i} = pathGroup{i}.property('ADBE Vector Shape');\n")
                script.append(f"var shape{i} = new Shape();\n")
                script.append(f"shape{i}.vertices = [{points_js}];\n")
                script.append(f"shape{i}.closed = false;\n")
                script.append(f"path{i}.setValue(shape{i});\n")
                script.append(f"var stroke{i} = contents{i}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"stroke{i}.property('ADBE Vector Stroke Color').setValue({color_js});\n")
                script.append(f"stroke{i}.property('ADBE Vector Stroke Width').setValue({elem['linewidth']});\n")
                # Apply linestyle (dashes) with dash_size scaling
                linestyle = elem.get("linestyle", "solid")
                dash_size = elem.get("dash_size", 1.0)
                if linestyle in ["dashed", "--"]:
                    dash_val, gap_val = self._get_dash_values(linestyle, dash_size)
                    script.append(f"stroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({dash_val});\n")
                    script.append(f"stroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({gap_val});\n")
                elif linestyle in ["dotted", ":"]:
                    dash_val, gap_val = self._get_dash_values(linestyle, dash_size)
                    script.append(f"stroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({dash_val});\n")
                    script.append(f"stroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({gap_val});\n")
                elif linestyle in ["dashdot", "-."]:
                    dash_val, gap_val = self._get_dash_values("dashed", dash_size)
                    dot_val, dot_gap = self._get_dash_values("dotted", dash_size)
                    script.append(f"stroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({dash_val});\n")
                    script.append(f"stroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({gap_val});\n")
                    script.append(f"stroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 2').setValue({dot_val});\n")
                    script.append(f"stroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 2').setValue({dot_gap});\n")
                # Animation using Trim Paths
                if elem["animate"] and elem["animate"] > 0:
                    script.append(f"var trim{i} = contents{i}.addProperty('ADBE Vector Filter - Trim');\n")
                    script.append(f"var endProp{i} = trim{i}.property('ADBE Vector Trim End');\n")
                    script.append(f"endProp{i}.setValueAtTime({delay}, 0);\n")
                    script.append(f"endProp{i}.setValueAtTime({delay + elem['animate']}, 100);\n")
                    # Apply easy ease to trim path keyframes
                    if self.easy_ease:
                        script.append(f"applyEasyEase(endProp{i}, {elem_ease_speed}, {elem_ease_influence});\n")
                # Add drop shadow if specified
                if elem.get("drop_shadow", False):
                    script.append(self._generate_drop_shadow_jsx(f"lineLayer{i}", f"{i}"))
            elif elem["type"] == "scatter":
                px, py = elem["x"], elem["y"]
                color_js = color_to_js(elem["color"])
                n_points = len(px)
                bar_anim_times = elem.get("bar_anim_times")
                total_anim = elem["animate"] if elem["animate"] else 1.0
                elem_ease_speed = self.ease_speed if elem.get("ease_speed") is None else elem.get("ease_speed")
                elem_ease_influence = self.ease_influence if elem.get("ease_influence") is None else elem.get("ease_influence")
                
                # Accept a single float for bar_anim_times and apply to all points
                if bar_anim_times is None:
                    bar_anim_times = [total_anim / n_points] * n_points
                elif isinstance(bar_anim_times, (int, float)):
                    bar_anim_times = [float(bar_anim_times)] * n_points
                # Defensive: if bar_anim_times is not a list of correct length, fallback to default
                if not hasattr(bar_anim_times, '__iter__') or len(bar_anim_times) != n_points:
                    bar_anim_times = [total_anim / n_points] * n_points
                
                # Overlapping animation: distribute start times evenly within total_anim
                start_times = np.linspace(0, total_anim - bar_anim_times[0], n_points)
                
                scatter_px, scatter_py = zip(*[self._data_to_shape(x, y, xmin_pad, xmax_pad, ymin_pad, ymax_pad) for x, y in zip(px, py)])
                for j, (sx, sy) in enumerate(zip(scatter_px, scatter_py)):
                    script.append(f"var scatterLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"scatterLayer{i}_{j}.name = \"Scatter_{i}_{j}\";\n")
                    script.append(f"scatterLayer{i}_{j}.property('Transform').property('Position').setValue([{center_x + sx}, {center_y + sy}]);\n")
                    script.append(f"scatterLayer{i}_{j}.parent = PlotAnchor;\n")
                    script.append(f"var contents{i}_{j} = scatterLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var ellipseGroup{i}_{j} = contents{i}_{j}.addProperty('ADBE Vector Shape - Ellipse');\n")
                    script.append(f"ellipseGroup{i}_{j}.property('ADBE Vector Ellipse Size').setValue([{elem['radius']*2}, {elem['radius']*2}]);\n")
                    script.append(f"var fill{i}_{j} = contents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    script.append(f"fill{i}_{j}.property('ADBE Vector Fill Color').setValue({color_js});\n")
                    
                    # Sequential animation for scatter points
                    if elem["animate"] and elem["animate"] > 0:
                        anim_time = bar_anim_times[j]
                        start_time = start_times[j]
                        script.append(f"var scale{i}_{j} = scatterLayer{i}_{j}.property('Transform').property('Scale');\n")
                        script.append(f"scale{i}_{j}.setValueAtTime({delay + start_time}, [0, 0, 100]);\n")
                        script.append(f"scale{i}_{j}.setValueAtTime({delay + start_time + anim_time}, [100, 100, 100]);\n")
                        # Apply easy ease to scale keyframes
                        if self.easy_ease:
                            script.append(f"applyEasyEase(scale{i}_{j}, {elem_ease_speed}, {elem_ease_influence});\n")
                    
                    # Add drop shadow if specified
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"scatterLayer{i}_{j}", f"{i}_{j}"))
            elif elem["type"] == "histogram":
                alpha = elem.get("alpha", 0.8)
                bin_left = elem["bin_left"]
                bin_right = elem["bin_right"]
                heights = elem["heights"]
                n_bars = len(bin_left)
                bar_anim_times = elem.get("bar_anim_times")
                total_anim = elem["animate"] if elem["animate"] else 1.0
                elem_ease_speed = self.ease_speed if elem.get("ease_speed") is None else elem.get("ease_speed")
                elem_ease_influence = self.ease_influence if elem.get("ease_influence") is None else elem.get("ease_influence")
                # Accept a single float for bar_anim_times and apply to all bars
                if bar_anim_times is None:
                    bar_anim_times = [total_anim / n_bars] * n_bars
                elif isinstance(bar_anim_times, (int, float)):
                    bar_anim_times = [float(bar_anim_times)] * n_bars
                # Defensive: if bar_anim_times is not a list of correct length, fallback to default
                if not hasattr(bar_anim_times, '__iter__') or len(bar_anim_times) != n_bars:
                    bar_anim_times = [total_anim / n_bars] * n_bars
                # Overlapping animation: distribute start times evenly within total_anim
                start_times = np.linspace(0, total_anim - bar_anim_times[0], n_bars)
                
                # Check if gradient colors are defined
                gradient_colors = elem.get("gradient_colors")
                if not gradient_colors:
                    # Use single color for all bars
                    color_js = color_to_js(elem["color"])
                
                for j, (left, right, height) in enumerate(zip(bin_left, bin_right, heights)):
                    # Use gradient color for this bar if available
                    if gradient_colors and j < len(gradient_colors):
                        bar_color_js = f"[{gradient_colors[j][0]}, {gradient_colors[j][1]}, {gradient_colors[j][2]}]"
                    else:
                        bar_color_js = color_js
                    
                    # Rectangle corners in data coordinates
                    x0, y0 = left, 0  # bottom left (axis)
                    x1, y1 = right, height  # top right
                    # Convert to shape coordinates
                    sx0, sy0 = self._data_to_shape(x0, y0, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    sx1, sy1 = self._data_to_shape(x1, y1, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    bar_width = abs(sx1 - sx0)
                    bar_height = abs(sy1 - sy0)
                    anchor_x = (sx0 + sx1) / 2
                    anchor_y = sy0  # axis (bottom)
                    position_x = center_x + anchor_x
                    position_y = center_y + anchor_y
                    script.append(f"var histLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"histLayer{i}_{j}.name = 'Histogram_{i}_{j}';\n")
                    script.append(f"histLayer{i}_{j}.property('Transform').property('Position').setValue([{position_x}, {position_y}]);\n")
                    script.append(f"histLayer{i}_{j}.parent = PlotAnchor;\n")
                    script.append(f"var histContents{i}_{j} = histLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var histRect{i}_{j} = histContents{i}_{j}.addProperty('ADBE Vector Shape - Rect');\n")
                    script.append(f"histRect{i}_{j}.property('ADBE Vector Rect Size').setValue([{bar_width}, {bar_height}]);\n")
                    script.append(f"histRect{i}_{j}.property('ADBE Vector Rect Position').setValue([0, -{bar_height/2}]);\n")
                    script.append(f"var histFill{i}_{j} = histContents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    script.append(f"histFill{i}_{j}.property('ADBE Vector Fill Color').setValue({bar_color_js});\n")
                    script.append(f"histFill{i}_{j}.property('ADBE Vector Fill Opacity').setValue({int(alpha*100)});\n")
                    # Animate bar height (scale Y from axis)
                    anim_time = bar_anim_times[j]
                    start_time = start_times[j]
                    script.append(f"var histScale{i}_{j} = histLayer{i}_{j}.property('Transform').property('Scale');\n")
                    script.append(f"histScale{i}_{j}.setValueAtTime({delay + start_time}, [100, 0, 100]);\n")
                    script.append(f"histScale{i}_{j}.setValueAtTime({delay + start_time + anim_time}, [100, 100, 100]);\n")
                    if self.easy_ease:
                        script.append(f"applyEasyEase(histScale{i}_{j}, {elem_ease_speed}, {elem_ease_influence});\n")
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"histLayer{i}_{j}", f"{i}_{j}"))
            elif elem["type"] == "bar_graph":
                alpha = elem.get("alpha", 0.8)
                bin_left = elem["bin_left"]
                bin_right = elem["bin_right"]
                heights = elem["heights"]
                n_bars = len(bin_left)
                bar_anim_times = elem.get("bar_anim_times")
                total_anim = elem["animate"] if elem["animate"] else 1.0
                elem_ease_speed = self.ease_speed if elem.get("ease_speed") is None else elem.get("ease_speed")
                elem_ease_influence = self.ease_influence if elem.get("ease_influence") is None else elem.get("ease_influence")
                # Accept a single float for bar_anim_times and apply to all bars
                if bar_anim_times is None:
                    bar_anim_times = [total_anim / n_bars] * n_bars
                elif isinstance(bar_anim_times, (int, float)):
                    bar_anim_times = [float(bar_anim_times)] * n_bars
                # Defensive: if bar_anim_times is not a list of correct length, fallback to default
                if not hasattr(bar_anim_times, '__iter__') or len(bar_anim_times) != n_bars:
                    bar_anim_times = [total_anim / n_bars] * n_bars
                # Overlapping animation: distribute start times evenly within total_anim
                start_times = np.linspace(0, total_anim - bar_anim_times[0], n_bars)
                
                # Check if gradient colors are defined
                gradient_colors = elem.get("gradient_colors")
                if not gradient_colors:
                    # Use single color for all bars
                    color_js = color_to_js(elem["color"])
                
                for j, (left, right, height) in enumerate(zip(bin_left, bin_right, heights)):
                    # Use gradient color for this bar if available
                    if gradient_colors and j < len(gradient_colors):
                        bar_color_js = f"[{gradient_colors[j][0]}, {gradient_colors[j][1]}, {gradient_colors[j][2]}]"
                    else:
                        bar_color_js = color_js
                    
                    # Rectangle corners in data coordinates
                    x0, y0 = left, 0  # bottom left (axis)
                    x1, y1 = right, height  # top right
                    # Convert to shape coordinates
                    sx0, sy0 = self._data_to_shape(x0, y0, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    sx1, sy1 = self._data_to_shape(x1, y1, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    bar_width = abs(sx1 - sx0)
                    bar_height = abs(sy1 - sy0)
                    anchor_x = (sx0 + sx1) / 2
                    anchor_y = sy0  # axis (bottom)
                    position_x = center_x + anchor_x
                    position_y = center_y + anchor_y
                    script.append(f"var barLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"barLayer{i}_{j}.name = 'Bar_{i}_{j}';\n")
                    script.append(f"barLayer{i}_{j}.property('Transform').property('Position').setValue([{position_x}, {position_y}]);\n")
                    script.append(f"barLayer{i}_{j}.parent = PlotAnchor;\n")
                    script.append(f"var barContents{i}_{j} = barLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var barRect{i}_{j} = barContents{i}_{j}.addProperty('ADBE Vector Shape - Rect');\n")
                    script.append(f"barRect{i}_{j}.property('ADBE Vector Rect Size').setValue([{bar_width}, {bar_height}]);\n")
                    script.append(f"barRect{i}_{j}.property('ADBE Vector Rect Position').setValue([0, -{bar_height/2}]);\n")
                    script.append(f"var barFill{i}_{j} = barContents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    script.append(f"barFill{i}_{j}.property('ADBE Vector Fill Color').setValue({bar_color_js});\n")
                    script.append(f"barFill{i}_{j}.property('ADBE Vector Fill Opacity').setValue({int(alpha*100)});\n")
                    # Animate bar height (scale Y from axis)
                    anim_time = bar_anim_times[j]
                    start_time = start_times[j]
                    script.append(f"var barScale{i}_{j} = barLayer{i}_{j}.property('Transform').property('Scale');\n")
                    script.append(f"barScale{i}_{j}.setValueAtTime({delay + start_time}, [100, 0, 100]);\n")
                    script.append(f"barScale{i}_{j}.setValueAtTime({delay + start_time + anim_time}, [100, 100, 100]);\n")
                    if self.easy_ease:
                        script.append(f"applyEasyEase(barScale{i}_{j}, {elem_ease_speed}, {elem_ease_influence});\n")
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"barLayer{i}_{j}", f"{i}_{j}"))
            elif elem["type"] == "barh":
                # Horizontal bar graph rendering
                alpha = elem.get("alpha", 0.8)
                bin_bottom = elem["bin_bottom"]
                bin_top = elem["bin_top"]
                widths = elem["widths"]
                n_bars = len(bin_bottom)
                bar_anim_times = elem.get("bar_anim_times")
                total_anim = elem["animate"] if elem["animate"] else 1.0
                animate_downward = elem.get("animate_downward", False)
                anchor_at_y_axis = elem.get("anchor_at_y_axis", False)
                elem_ease_speed = self.ease_speed if elem.get("ease_speed") is None else elem.get("ease_speed")
                elem_ease_influence = self.ease_influence if elem.get("ease_influence") is None else elem.get("ease_influence")
                
                # Handle bar_anim_times (individual bar duration)
                if bar_anim_times is None:
                    # Default: sequential animation, no overlap
                    individual_duration = total_anim / n_bars
                    bar_anim_times = [individual_duration] * n_bars
                    # Bars animate one after another
                    start_times_base = np.linspace(0, total_anim - individual_duration, n_bars)
                elif isinstance(bar_anim_times, (int, float)):
                    # User specified individual bar duration - enable overlapping
                    individual_duration = float(bar_anim_times)
                    bar_anim_times = [individual_duration] * n_bars
                    if n_bars > 1:
                        # Distribute start times so all bars finish within total_anim
                        # Last bar starts at (total_anim - individual_duration) and finishes at total_anim
                        start_times_base = np.linspace(0, max(0, total_anim - individual_duration), n_bars)
                    else:
                        start_times_base = [0]
                else:
                    # List of durations provided
                    if not hasattr(bar_anim_times, '__iter__') or len(bar_anim_times) != n_bars:
                        bar_anim_times = [total_anim / n_bars] * n_bars
                    start_times_base = np.linspace(0, total_anim - bar_anim_times[0], n_bars)

                # Reorder start times to animate downward (top to bottom) if requested
                centers = np.asarray(elem.get("bin_centers", []))
                if animate_downward and len(centers) == n_bars:
                    order = np.argsort(centers)[::-1]  # descending by y value (top to bottom)
                    start_times = [0.0] * n_bars
                    for rank, j_idx in enumerate(order):
                        start_times[j_idx] = float(start_times_base[rank])
                else:
                    start_times = start_times_base
                
                # Check if gradient colors are defined
                gradient_colors = elem.get("gradient_colors")
                if not gradient_colors:
                    # Use single color for all bars
                    color_js = color_to_js(elem["color"])
                
                for j, (bottom, top, width) in enumerate(zip(bin_bottom, bin_top, widths)):
                    # Use gradient color for this bar if available
                    if gradient_colors and j < len(gradient_colors):
                        bar_color_js = f"[{gradient_colors[j][0]}, {gradient_colors[j][1]}, {gradient_colors[j][2]}]"
                    else:
                        bar_color_js = color_js
                    
                    # Rectangle corners in data coordinates (horizontal bars)
                    # Apply horizontal clipping to [xmin_pad, xmax_pad] when show_all_points is False
                    clip = not self.show_all_points
                    if width >= 0:
                        bar_start, bar_end = 0, width
                    else:
                        bar_start, bar_end = width, 0
                    if clip:
                        xi0 = max(bar_start, xmin_pad)
                        xi1 = min(bar_end, xmax_pad)
                    else:
                        xi0, xi1 = bar_start, bar_end
                    # If fully out of bounds, skip drawing this bar
                    if xi1 <= xi0 + 1e-12:
                        continue
                    x0, y0 = xi0, bottom  # left boundary after clipping
                    x1, y1 = xi1, top     # right boundary after clipping
                    # Convert to shape coordinates
                    sx0, sy0 = self._data_to_shape(x0, y0, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    sx1, sy1 = self._data_to_shape(x1, y1, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    bar_width = abs(sx1 - sx0)  # horizontal extent
                    bar_height = abs(sy1 - sy0)  # vertical thickness
                    # Anchor selection: pyramid bars anchor at y-axis; others anchor at bar start
                    if anchor_at_y_axis:
                        center_y_data = (bottom + top) / 2
                        axis_sx, _ = self._data_to_shape(0, center_y_data, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        anchor_x = axis_sx
                    else:
                        anchor_x = sx0  # left boundary (after clipping)
                    anchor_y = (sy0 + sy1) / 2  # center vertically
                    position_x = center_x + anchor_x
                    position_y = center_y + anchor_y
                    script.append(f"var barhLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"barhLayer{i}_{j}.name = 'BarhBar_{i}_{j}';\n")
                    script.append(f"barhLayer{i}_{j}.property('Transform').property('Position').setValue([{position_x}, {position_y}]);\n")
                    script.append(f"barhLayer{i}_{j}.parent = PlotAnchor;\n")
                    script.append(f"var barhContents{i}_{j} = barhLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var barhRect{i}_{j} = barhContents{i}_{j}.addProperty('ADBE Vector Shape - Rect');\n")
                    script.append(f"barhRect{i}_{j}.property('ADBE Vector Rect Size').setValue([{bar_width}, {bar_height}]);\n")
                    # Position rectangle so scaling originates at anchor
                    if anchor_at_y_axis:
                        if width >= 0:
                            script.append(f"barhRect{i}_{j}.property('ADBE Vector Rect Position').setValue([{bar_width/2}, 0]);\n")
                        else:
                            script.append(f"barhRect{i}_{j}.property('ADBE Vector Rect Position').setValue([{-bar_width/2}, 0]);\n")
                    else:
                        script.append(f"barhRect{i}_{j}.property('ADBE Vector Rect Position').setValue([{bar_width/2}, 0]);\n")
                    script.append(f"var barhFill{i}_{j} = barhContents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    script.append(f"barhFill{i}_{j}.property('ADBE Vector Fill Color').setValue({bar_color_js});\n")
                    script.append(f"barhFill{i}_{j}.property('ADBE Vector Fill Opacity').setValue({int(alpha*100)});\n")
                    # Animate bar width (scale X from axis)
                    anim_time = bar_anim_times[j]
                    start_time = start_times[j]
                    script.append(f"var barhScale{i}_{j} = barhLayer{i}_{j}.property('Transform').property('Scale');\n")
                    script.append(f"barhScale{i}_{j}.setValueAtTime({delay + start_time}, [0, 100, 100]);\n")
                    script.append(f"barhScale{i}_{j}.setValueAtTime({delay + start_time + anim_time}, [100, 100, 100]);\n")
                    if self.easy_ease:
                        script.append(f"applyEasyEase(barhScale{i}_{j}, {elem_ease_speed}, {elem_ease_influence});\n")
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"barhLayer{i}_{j}", f"{i}_{j}"))

        # Generate axes AFTER plot elements for histograms and bar graphs (so they appear on top)
        # Mixed plots get axes on top to accommodate bar/histogram rendering
        if has_bars_or_hist or not has_scatter:
            self._generate_axes_jsx(script, center_x, center_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad, ANIM_DURATION, has_barh)

        # --- DYNAMIC LEGEND PLACEMENT (relative to graph area) ---
        legend_entries = []
        legend_colors = []
        legend_styles = []  # Store line styles for line_style legend
        legend_widths = []  # Store line widths for line_style legend
        for elem in self.elements:
            if elem.get('label'):
                legend_entries.append(elem['label'])
                legend_colors.append(color_to_js(elem['color']))
                # Store line style info (for line_style legends)
                legend_styles.append(elem.get('linestyle', 'solid'))
                legend_widths.append(elem.get('linewidth', 4))
        has_heading = bool(self.title or self.subtitle)
        legend_loc = 'bottomright' if has_heading else 'topright'  # heading-aware default
        margin = 80
        legend_width = 300
        legend_height = 40 + 30 * len(legend_entries)
        # Reserve a center-top region for title/subtitle so legend placement avoids it.
        title_zone_top = -self.height / 2 + 30
        title_zone_bottom = -self.height / 2 + 80 + (52 * self.font_scale if (self.title and self.subtitle) else 0) + (45 * self.font_scale)
        title_zone_left = -self.width * 0.30
        title_zone_right = self.width * 0.30
        # Heuristic: choose legend position that maximizes minimum distance from data and text zones.
        data_px, data_py = [], []
        for elem in self.elements:
            if elem["type"] in ["line", "scatter"]:
                px, py = elem["x"], elem["y"]
            elif elem["type"] in ["histogram", "bar_graph"]:
                px = elem["bin_centers"]
                py = elem["heights"]
            elif elem["type"] == "barh":
                px = elem["widths"]
                py = elem["bin_centers"]
            else:
                continue

            if not px or not py:
                continue
            shape_px, shape_py = zip(*[self._data_to_shape(x, y, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                                    for x, y in zip(px, py)])
            data_px.extend(shape_px)
            data_py.extend(shape_py)

        # Approximate text exclusion zones in graph-local coordinates.
        text_zones = []
        if has_heading:
            text_zones.append((title_zone_left, title_zone_right, title_zone_top, title_zone_bottom))
        # Bottom tick/x-label text band.
        if self.xticks or self.xlabel:
            text_zones.append((-self.width / 2, self.width / 2, self.height / 2 - 45, self.height / 2 + 120))
        # Left y-label/y-tick text band.
        if self.yticks or self.ylabel:
            text_zones.append((-self.width / 2 - 170, -self.width / 2 + 70, -self.height / 2, self.height / 2))

        def point_to_rect_distance(px, py, left, right, top, bottom):
            dx = max(left - px, 0, px - right)
            dy = max(top - py, 0, py - bottom)
            return math.sqrt(dx * dx + dy * dy)

        def rect_to_rect_distance(l1, r1, t1, b1, l2, r2, t2, b2):
            dx = max(l2 - r1, l1 - r2, 0)
            dy = max(t2 - b1, t1 - b2, 0)
            return math.sqrt(dx * dx + dy * dy)

        x_min = -self.width / 2 + margin + legend_width / 2
        x_max = self.width / 2 - margin - legend_width / 2
        y_min = -self.height / 2 + margin + legend_height / 2
        y_max = self.height / 2 - margin - legend_height / 2

        candidate_positions = []
        if x_min <= x_max and y_min <= y_max:
            x_steps = 7
            y_steps = 6
            for xi in range(x_steps):
                lx = x_min + (x_max - x_min) * (xi / float(x_steps - 1))
                for yi in range(y_steps):
                    ly = y_min + (y_max - y_min) * (yi / float(y_steps - 1))
                    candidate_positions.append((lx, ly))
        else:
            # Fallback in extremely small graph areas.
            candidate_positions.append((0, 0))

        # Slight bias to top-right when heading exists and score ties.
        preferred_x = self.width / 2 - margin - legend_width / 2
        preferred_y = -self.height / 2 + margin + legend_height / 2
        best_score = -1.0
        legend_x, legend_y = preferred_x, preferred_y

        for lx, ly in candidate_positions:
            legend_left = lx - legend_width / 2
            legend_right = lx + legend_width / 2
            legend_top = ly - legend_height / 2
            legend_bottom = ly + legend_height / 2

            min_data_dist = float("inf")
            for x, y in zip(data_px, data_py):
                dist = point_to_rect_distance(x, y, legend_left, legend_right, legend_top, legend_bottom)
                if dist < min_data_dist:
                    min_data_dist = dist
            if not data_px:
                min_data_dist = 1e9

            min_text_dist = float("inf")
            for zl, zr, zt, zb in text_zones:
                dist = rect_to_rect_distance(legend_left, legend_right, legend_top, legend_bottom, zl, zr, zt, zb)
                if dist < min_text_dist:
                    min_text_dist = dist
            if not text_zones:
                min_text_dist = 1e9

            score = min(min_data_dist, min_text_dist)
            # Small tie-breaker toward top-right for titled charts.
            if has_heading:
                score += 0.001 * (-(abs(lx - preferred_x) + abs(ly - preferred_y)))

            if score > best_score:
                best_score = score
                legend_x, legend_y = lx, ly
        if legend_entries:
            script.append(f"var legendGroup = [];\n")
            for i, (label, color) in enumerate(zip(legend_entries, legend_colors)):
                y_offset = legend_y + 20 + i * 30
                
                # Choose rendering mode based on legend_style
                if self.legend_style == 'line_style':
                    # Line-style legend: render actual line samples with styles
                    script.append(f"var legendLine{i} = comp.layers.addShape();\n")
                    script.append(f"legendLine{i}.property('Transform').property('Position').setValue([{center_x + legend_x - legend_width/2 + 30}, {center_y + y_offset}]);\n")
                    script.append(f"legendLine{i}.parent = PlotAnchor;\n")
                    script.append(f"var legendLineContents{i} = legendLine{i}.property('ADBE Root Vectors Group');\n")
                    # Create a line path - length depends on line style
                    linestyle = legend_styles[i]
                    script.append(f"var legendLinePath{i} = legendLineContents{i}.addProperty('ADBE Vector Shape - Group');\n")
                    script.append(f"var legendLineShape{i} = legendLinePath{i}.property('ADBE Vector Shape');\n")
                    script.append(f"var shape{i} = new Shape();\n")
                    # Solid line is slightly shorter to match dashed visual span (3 dashes + 2 gaps = 64 units).
                    if linestyle in ["solid", "-"]:
                        script.append(f"shape{i}.vertices = [[-32, 0], [32, 0]];\n")  # 64 units for solid
                    else:
                        script.append(f"shape{i}.vertices = [[-32, 0], [32, 0]];\n")  # 64 units for dashed/dotted
                    script.append(f"shape{i}.closed = false;\n")
                    script.append(f"legendLineShape{i}.setValue(shape{i});\n")
                    # Add stroke with color, width, and line cap
                    script.append(f"var legendLineStroke{i} = legendLineContents{i}.addProperty('ADBE Vector Graphic - Stroke');\n")
                    script.append(f"legendLineStroke{i}.property('ADBE Vector Stroke Color').setValue({color});\n")
                    linewidth = legend_widths[i]
                    script.append(f"legendLineStroke{i}.property('ADBE Vector Stroke Width').setValue({linewidth});\n")
                    # Set line cap to round for smooth appearance
                    script.append(f"legendLineStroke{i}.property('ADBE Vector Stroke Line Cap').setValue(2);\n")  # 2 = Round cap
                    script.append(f"legendLineStroke{i}.property('ADBE Vector Stroke Line Join').setValue(1);\n")  # 1 = Round join
                    
                    # Apply line style (dashes) - optimized for 2 complete dashes
                    if linestyle in ["dashed", "--"]:
                        # Dashed: ~2 dashes of 16 units with 8 unit gaps (16+8+16+8 = 48, fits in 70)
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue(16);\n")
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue(8);\n")
                    elif linestyle in ["dotted", ":"]:
                        # Dotted: ~2 dots with small gaps (3+5+3+5 = 16, fits well in 70)
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue(3);\n")
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue(5);\n")
                    elif linestyle in ["dashdot", "-."]:
                        # Dash-dot: dash then dot pattern
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue(12);\n")
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue(4);\n")
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 2').setValue(3);\n")
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 2').setValue(4);\n")
                    # else: solid lines have no dashes, renders full line
                    
                    # Fade-in animation for line
                    script.append(f"legendLine{i}.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                    script.append(f"legendLine{i}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION}, 100);\n")
                    # Apply easy ease to legend line opacity keyframes
                    if self.easy_ease:
                        script.append(f"applyEasyEase(legendLine{i}.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
                else:
                    # Color-only legend: render solid color swatches
                    script.append(f"var legendSwatch{i} = comp.layers.addShape();\n")
                    script.append(f"legendSwatch{i}.property('Transform').property('Position').setValue([{center_x + legend_x - legend_width/2 + 30}, {center_y + y_offset}]);\n")
                    script.append(f"legendSwatch{i}.parent = PlotAnchor;\n")
                    script.append(f"var legendSwatchContents{i} = legendSwatch{i}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var legendSwatchRect{i} = legendSwatchContents{i}.addProperty('ADBE Vector Shape - Rect');\n")
                    script.append(f"legendSwatchRect{i}.property('ADBE Vector Rect Size').setValue([24, 24]);\n")
                    script.append(f"var legendSwatchFill{i} = legendSwatchContents{i}.addProperty('ADBE Vector Graphic - Fill');\n")
                    script.append(f"legendSwatchFill{i}.property('ADBE Vector Fill Color').setValue({color});\n")
                    # Fade-in animation for swatch
                    script.append(f"legendSwatch{i}.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                    script.append(f"legendSwatch{i}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION}, 100);\n")
                    # Apply easy ease to legend swatch opacity keyframes
                    if self.easy_ease:
                        script.append(f"applyEasyEase(legendSwatch{i}.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
                    
                    # Add drop shadow to legend swatch if specified
                    if self.drop_shadow:
                        script.append(self._generate_drop_shadow_jsx(f"legendSwatch{i}", f"LegendSwatch{i}"))
                
                # Label
                script.append(f"var legendLabel{i} = comp.layers.addText(\"{label}\");\n")
                script.append(f"var legendLabelProp{i} = legendLabel{i}.property('Source Text');\n")
                script.append(f"var legendLabelDoc{i} = legendLabelProp{i}.value;\n")
                script.append(f"legendLabelDoc{i}.fontSize = {int(24 * self.font_scale)};\n")
                script.append(f"legendLabelDoc{i}.fillColor = {color_to_js(self.ui_color)};\n")
                script.append(f"legendLabelDoc{i}.justification = ParagraphJustification.LEFT_JUSTIFY;\n")
                script.append(f"legendLabelProp{i}.setValue(legendLabelDoc{i});\n")
                script.append(f"var legendLabelSR{i} = legendLabel{i}.sourceRectAtTime(0, false);\n")
                script.append(f"var legendLabelAP{i} = legendLabel{i}.property('Transform').property('Anchor Point');\n")
                script.append(f"legendLabelAP{i}.setValue([0, legendLabelSR{i}.top + legendLabelSR{i}.height/2]);\n")
                script.append(f"legendLabel{i}.property('Transform').property('Position').setValue([{center_x + legend_x - legend_width/2 + 70}, {center_y + y_offset + 2}]);\n")
                script.append(f"legendLabel{i}.parent = PlotAnchor;\n")
                # Fade-in animation for label
                script.append(f"legendLabel{i}.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                script.append(f"legendLabel{i}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION}, 100);\n")
                # Apply easy ease to legend label opacity keyframes
                if self.drop_shadow:
                    script.append(self._generate_drop_shadow_jsx(f"legendLabel{i}", f"LegendLabel{i}"))



        # Title and subtitle (relative to graph area)
        title_top_y = center_y - self.height/2 + 80
        if self.title:
            escaped_title = self.title.replace('"', '\\"').replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
            script.append(f"var titleLayer = comp.layers.addText(\"{escaped_title}\");\n")
            script.append(f"titleLayer.property('Transform').property('Position').setValue([{center_x}, {title_top_y}]);\n")
            script.append(f"titleLayer.parent = PlotAnchor;\n")
            script.append("var titleProp = titleLayer.property('Source Text');\n")
            script.append("var titleDoc = titleProp.value;\n")
            script.append(f"titleDoc.fontSize = {int(48 * self.font_scale)};\n")
            script.append(f"titleDoc.fillColor = {color_to_js(self.ui_color)};\n")
            script.append("titleDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("titleProp.setValue(titleDoc);\n")
            # Anchor the title to top-center so its top edge aligns precisely
            script.append("var titleSR = titleLayer.sourceRectAtTime(0, false);\n")
            script.append("var titleAP = titleLayer.property('Transform').property('Anchor Point');\n")
            script.append("titleAP.setValue([titleSR.left + titleSR.width/2, titleSR.top]);\n")
            
            # Animate title
            if self.animate_opacity:
                script.append(f"titleLayer.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                script.append(f"titleLayer.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 0.9}, 100);\n")
                if self.easy_ease:
                    script.append(f"applyEasyEase(titleLayer.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
            else:
                script.append(f"titleLayer.property('Transform').property('Opacity').setValue(100);\n")
            
            # Add drop shadow to title if specified
            if self.drop_shadow:
                script.append(self._generate_drop_shadow_jsx("titleLayer", "Title"))

        if self.subtitle:
            escaped_subtitle = self.subtitle.replace('"', '\\"').replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
            subtitle_y = title_top_y + (52 * self.font_scale) if self.title else title_top_y
            subtitle_color = self.subtitle_color if self.subtitle_color is not None else self.ui_color
            script.append(f"var subtitleLayer = comp.layers.addText(\"{escaped_subtitle}\");\n")
            script.append(f"subtitleLayer.property('Transform').property('Position').setValue([{center_x}, {subtitle_y}]);\n")
            script.append(f"subtitleLayer.parent = PlotAnchor;\n")
            script.append("var subtitleProp = subtitleLayer.property('Source Text');\n")
            script.append("var subtitleDoc = subtitleProp.value;\n")
            script.append(f"subtitleDoc.fontSize = {int(28 * self.font_scale)};\n")
            script.append(f"subtitleDoc.fillColor = {color_to_js(subtitle_color)};\n")
            script.append("subtitleDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("subtitleProp.setValue(subtitleDoc);\n")
            script.append("var subtitleSR = subtitleLayer.sourceRectAtTime(0, false);\n")
            script.append("var subtitleAP = subtitleLayer.property('Transform').property('Anchor Point');\n")
            script.append("subtitleAP.setValue([subtitleSR.left + subtitleSR.width/2, subtitleSR.top]);\n")

            if self.animate_opacity:
                script.append(f"subtitleLayer.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                script.append(f"subtitleLayer.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 0.95}, 100);\n")
                if self.easy_ease:
                    script.append(f"applyEasyEase(subtitleLayer.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
            else:
                script.append(f"subtitleLayer.property('Transform').property('Opacity').setValue(100);\n")

            if self.drop_shadow:
                script.append(self._generate_drop_shadow_jsx("subtitleLayer", "Subtitle"))

        # Axis labels (relative to graph area)
        if self.xlabel:
            escaped_xlabel = self.xlabel.replace('"', '\\"').replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
            script.append(f"var xlabelLayer = comp.layers.addText(\"{escaped_xlabel}\");\n")
            # Position x-label below the x-axis ticks (if any) or below the axis
            if self.xticks:
                # If there are ticks, position below the tick labels
                script.append(f"xlabelLayer.property('Transform').property('Position').setValue([{center_x}, {center_y + self.height/2 + 60}]);\n")
            else:
                # If no ticks, position below the axis
                script.append(f"xlabelLayer.property('Transform').property('Position').setValue([{center_x}, {center_y + self.height/2 + 40}]);\n")
            script.append(f"xlabelLayer.parent = PlotAnchor;\n")
            script.append("var xlabelProp = xlabelLayer.property('Source Text');\n")
            script.append("var xlabelDoc = xlabelProp.value;\n")
            script.append(f"xlabelDoc.fontSize = {int(41 * self.font_scale)};\n")
            script.append(f"xlabelDoc.fillColor = {color_to_js(self.ui_color)};\n")
            script.append("xlabelDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("xlabelProp.setValue(xlabelDoc);\n")
            script.append("var xlabelSR = xlabelLayer.sourceRectAtTime(0, false);\n")
            script.append("var xlabelAP = xlabelLayer.property('Transform').property('Anchor Point');\n")
            script.append("xlabelAP.setValue([xlabelSR.left + xlabelSR.width/2, xlabelSR.top]);\n")
            
            # Animate xlabel
            if self.animate_opacity:
                script.append(f"xlabelLayer.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                script.append(f"xlabelLayer.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 0.9}, 100);\n")
                if self.easy_ease:
                    script.append(f"applyEasyEase(xlabelLayer.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
            else:
                script.append(f"xlabelLayer.property('Transform').property('Opacity').setValue(100);\n")
            
            # Add drop shadow to x-label if specified
            if self.drop_shadow:
                script.append(self._generate_drop_shadow_jsx("xlabelLayer", "XLabel"))
        if self.ylabel:
            escaped_ylabel = self.ylabel.replace('"', '\\"').replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
            script.append(f"var ylabelLayer = comp.layers.addText(\"{escaped_ylabel}\");\n")
            # Position y-label to the left of the y-axis ticks (if any) or to the left of the axis
            if self.yticks:
                # If there are ticks, position to the left of the tick labels
                script.append(f"ylabelLayer.property('Transform').property('Position').setValue([{center_x - self.width/2 - 60}, {center_y}]);\n")
            else:
                # If no ticks, position to the left of the axis
                script.append(f"ylabelLayer.property('Transform').property('Position').setValue([{center_x - self.width/2 - 20}, {center_y}]);\n")
            script.append(f"ylabelLayer.parent = PlotAnchor;\n")
            script.append("ylabelLayer.property('Transform').property('Rotation').setValue(-90);\n")
            script.append("var ylabelProp = ylabelLayer.property('Source Text');\n")
            script.append("var ylabelDoc = ylabelProp.value;\n")
            script.append(f"ylabelDoc.fontSize = {int(41 * self.font_scale)};\n")
            script.append(f"ylabelDoc.fillColor = {color_to_js(self.ui_color)};\n")
            script.append("ylabelDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("ylabelProp.setValue(ylabelDoc);\n")
            
            # Animate ylabel
            if self.animate_opacity:
                script.append(f"ylabelLayer.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                script.append(f"ylabelLayer.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 0.9}, 100);\n")
                if self.easy_ease:
                    script.append(f"applyEasyEase(ylabelLayer.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
            else:
                script.append(f"ylabelLayer.property('Transform').property('Opacity').setValue(100);\n")
            
            # Add drop shadow to y-label if specified
            if self.drop_shadow:
                script.append(self._generate_drop_shadow_jsx("ylabelLayer", "YLabel"))

        # --- ANNOTATIONS (generated on top of all other layers) ---
        annotation_count = 0
        for i, elem in enumerate(self.elements):
            if elem["type"] == "annotation":
                # Convert annotation coordinates from data space to shape coordinates
                ann_x, ann_y = self._data_to_shape(elem["x"], elem["y"], xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                ann_pos_x = center_x + ann_x
                ann_pos_y = center_y + ann_y
                
                # Escape special characters in text for JSX safety
                escaped_text = elem['text'].replace('"', '\\"').replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
                
                # Map alignment to After Effects justification
                alignment_map = {
                    "left": "ParagraphJustification.LEFT_JUSTIFY",
                    "center": "ParagraphJustification.CENTER_JUSTIFY", 
                    "right": "ParagraphJustification.RIGHT_JUSTIFY"
                }
                justification = alignment_map.get(elem.get("alignment", "left"), "ParagraphJustification.LEFT_JUSTIFY")
                
                # Create text layer for annotation
                script.append(f"var annotationLayer{annotation_count} = comp.layers.addText(\"{escaped_text}\");\n")
                script.append(f"annotationLayer{annotation_count}.name = \"Annotation_{annotation_count}\";\n")
                script.append(f"annotationLayer{annotation_count}.property('Transform').property('Position').setValue([{ann_pos_x}, {ann_pos_y}]);\n")
                script.append(f"annotationLayer{annotation_count}.parent = PlotAnchor;\n")
                
                # Set text properties
                script.append(f"var annotationProp{annotation_count} = annotationLayer{annotation_count}.property('Source Text');\n")
                script.append(f"var annotationDoc{annotation_count} = annotationProp{annotation_count}.value;\n")
                script.append(f"annotationDoc{annotation_count}.fontSize = {int(elem['fontsize'] * self.font_scale)};\n")
                script.append(f"annotationDoc{annotation_count}.font = \"{elem['font']}\";\n")
                script.append(f"annotationDoc{annotation_count}.fillColor = {color_to_js(self.ui_color)};\n")
                script.append(f"annotationDoc{annotation_count}.justification = {justification};\n")
                script.append(f"annotationProp{annotation_count}.setValue(annotationDoc{annotation_count});\n")
                
                # Fade-in animation for annotation
                script.append(f"annotationLayer{annotation_count}.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                script.append(f"annotationLayer{annotation_count}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 0.9}, 100);\n")
                
                # Apply easy ease to annotation opacity keyframes
                if self.easy_ease:
                    script.append(f"applyEasyEase(annotationLayer{annotation_count}.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
                
                # Add drop shadow if globally enabled
                if self.drop_shadow:
                    script.append(self._generate_drop_shadow_jsx(f"annotationLayer{annotation_count}", f"Annotation{annotation_count}"))
                
                annotation_count += 1

        # --- CINEMATIC ADJUSTMENT LAYER (generated last to appear on top) ---
        if self.cinematic_effects:
            script.append(f"var adj = comp.layers.addSolid([1,1,1], \"CinematicAdjustment\", {self.comp_width}, {self.comp_height}, 1.0);\n")
            script.append(f"adj.adjustmentLayer = true;\n")
            script.append(f"adj.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
            script.append(f"adj.parent = PlotAnchor;\n")
            script.append(f"adj.moveToBeginning();\n")
            
            # Add CC Vignette effect
            vignette = True
            if vignette:
                script.append(f"var vignette = adj.property(\"Effects\").addProperty(\"CC Vignette\");\n")
                script.append(f"if (vignette != null) {{\n")
                script.append(f"    vignette.property(\"Amount\").setValue(50);     // Darken edges\n")
                script.append(f"    vignette.property(\"Angle of View\").setValue(50);   // Vignette spread\n")
                script.append(f"    vignette.property(\"Center\").setValue([{self.comp_width/2}, {self.comp_height/2}]);\n")
                script.append(f"    vignette.property(\"Pin Highlights\").setValue(0);   // Preserve highlights\n")
                script.append(f"}} else {{\n")
                script.append(f"    // Fallback to CS Vignette if CC Vignette not available\n")
                script.append(f"    var csVignette = adj.property(\"Effects\").addProperty(\"CS Vignette\");\n")
                script.append(f"    if (csVignette != null) {{\n")
                script.append(f"        csVignette.property(\"Amount\").setValue(50);     // Darken edges\n")
                script.append(f"        csVignette.property(\"Angle of View\").setValue(50);   // Vignette spread\n")
                script.append(f"        csVignette.property(\"Center\").setValue([{self.comp_width/2}, {self.comp_height/2}]);\n")
                script.append(f"        csVignette.property(\"Pin Highlights\").setValue(0);   // Preserve highlights\n")
                script.append(f"    }}\n")
                script.append(f"}}\n")
            
            # Add Sharpen effect
            sharpen = False
            if sharpen:
                script.append(f"var sharpen = adj.property(\"Effects\").addProperty(\"Sharpen\");\n")
                script.append(f"if (sharpen != null) {{\n")
                script.append(f"    sharpen.property(\"Sharpen Amount\").setValue(87);     // Sharpening intensity\n")
                script.append(f"}}\n")
            
            # Add Noise effect
            
            noise = False
            if noise:
                script.append(f"var noise = adj.property(\"Effects\").addProperty(\"Noise\");\n")
                script.append(f"if (noise != null) {{\n")
                script.append(f"    noise.property(\"Amount of Noise\").setValue(11);     // Noise intensity\n")
                script.append(f"    noise.property(\"Noise Type\").setValue(1);     // 0=Uniform, 1=Squared\n")
                script.append(f"    noise.property(\"Clipping\").setValue(1);     // 0=Clip, 1=Wrap\n")
                script.append(f"}}\n")

        # --- WIGGLE ADJUSTMENT LAYER (generated last for organic movement) ---
        if self.wiggle:
            script.append(f"var wiggleAdj = comp.layers.addSolid([1,1,1], \"WiggleAdjustment\", {self.comp_width}, {self.comp_height}, 1.0);\n")
            script.append(f"wiggleAdj.adjustmentLayer = true;\n")
            script.append(f"wiggleAdj.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
            script.append(f"wiggleAdj.parent = PlotAnchor;\n")
            script.append(f"wiggleAdj.moveToBeginning();\n")
            # Add Turbulent Displace effect
            script.append(f"var turbulent = wiggleAdj.property(\"Effects\").addProperty(\"Turbulent Displace\");\n")
            script.append(f"if (turbulent != null) {{\n")
            script.append(f"    turbulent.property(\"Displacement\").setValue(1);     // Displacement type\n")
            script.append(f"    turbulent.property(\"Amount\").setValue(3);     // Displacement amount\n")
            script.append(f"    turbulent.property(\"Size\").setValue(19);     // Turbulence size\n")
            script.append(f"    turbulent.property(\"Offset (Turbulence)\").setValue([{self.comp_width/2}, {self.comp_height/2}]);     // Center offset\n")
            script.append(f"    turbulent.property(\"Complexity\").setValue(10);     // Complexity\n")
            script.append(f"    turbulent.property(\"Evolution\").setValueAtTime(0, 0);     // Start evolution\n")
            script.append(f"    turbulent.property(\"Evolution\").setValueAtTime(0.2, -267);     // Keyframe 1\n")
            script.append(f"    turbulent.property(\"Evolution\").setValueAtTime(0.4, 76);     // Keyframe 2\n")
            script.append(f"    turbulent.property(\"Evolution\").setValueAtTime(0.6, -143);     // Keyframe 3\n")
            script.append(f"    turbulent.property(\"Evolution\").setValueAtTime(0.8, -313);     // Keyframe 4\n")
            script.append(f"    turbulent.property(\"Evolution\").setValueAtTime(1.0, 0);     // End evolution\n")
            script.append(f"    // Set all keyframes to hold interpolation\n")
            script.append(f"    turbulent.property(\"Evolution\").setInterpolationTypeAtKey(1, KeyframeInterpolationType.HOLD);\n")
            script.append(f"    turbulent.property(\"Evolution\").setInterpolationTypeAtKey(2, KeyframeInterpolationType.HOLD);\n")
            script.append(f"    turbulent.property(\"Evolution\").setInterpolationTypeAtKey(3, KeyframeInterpolationType.HOLD);\n")
            script.append(f"    turbulent.property(\"Evolution\").setInterpolationTypeAtKey(4, KeyframeInterpolationType.HOLD);\n")
            script.append(f"    turbulent.property(\"Evolution\").setInterpolationTypeAtKey(5, KeyframeInterpolationType.HOLD);\n")
            script.append(f"    turbulent.property(\"Evolution\").expression = \"loopOut('cycle')\";     // Loop animation\n")
            script.append(f"    turbulent.property(\"Pinning\").setValue(3);     // Pinning mode\n")
            script.append(f"    turbulent.property(\"Resize Layer\").setValue(0);     // Don't resize layer\n")
            script.append(f"    turbulent.property(\"Antialiasing for Best Quality\").setValue(1);     // Best quality\n")
            script.append(f"}}\n")

        script.append("app.endUndoGroup();\n")
        return "".join(script)

    def save(self, filename: str = "", folder_path: str = "./"):
        """
        Write the JSX script to a file for After Effects.
        """
        if filename == "":
            filename = f"AEGraph_{get_time()}.jsx"
        jsx = self._generate_jsx()
        with open(f"{folder_path.rstrip('/')}/{filename}", "w") as f:
            f.write(jsx)
        print(f"JSX script saved to {folder_path.rstrip('/')}/{filename}")
        return self

    def render(self, ae_version="Adobe After Effects 2025", folder_path: str = "./"):
        """
        Write and run the JSX script in After Effects (macOS only, requires AE scripting enabled).
        Now prints diagnostics and uses absolute path.
        """
        import os
        import subprocess
        filename = f"AEGraph_{get_time()}.jsx"
        self.save(filename, folder_path)
        abs_path = os.path.abspath(os.path.join(folder_path, filename))
        print(f"[AEGraph] Attempting to run JSX script: {abs_path}")
        apple_script = f'''
        tell application \"{ae_version}\"
            activate
            DoScriptFile \"{abs_path}\"
        end tell
        '''
        print("[AEGraph] AppleScript command:")
        print(apple_script)
        try:
            result = subprocess.run(["osascript", "-e", apple_script], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[AEGraph] Error running osascript: {result.stderr}")
            else:
                print(f"[AEGraph] Script sent to After Effects: {filename}")
        except Exception as e:
            print(f"[AEGraph] Exception during render: {e}")
        return self