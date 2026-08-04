import datetime
import json
import os
import numpy as np
from typing import List, Tuple, Optional, Union
import csv
import math
try:
    import pandas as pd
except ImportError:
    pd = None

from aegraph_config import config, UNSET, THEMES

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


def color_to_js(color: Union[str, List[float], Tuple[float, float, float], np.ndarray]) -> str:
    """Convert color name or RGB list/tuple to AE-friendly JS array [r,g,b].

    Accepts:
    - Named colors from COLOR_NAMES (0–1 floats)
    - RGB lists/tuples or numpy arrays in 0–1 floats or 0–255 ints
    """
    if isinstance(color, str):
        rgb = COLOR_NAMES.get(color.lower())
        if rgb is None:
            raise ValueError(f"Unknown color name: {color}")
    elif isinstance(color, (list, tuple, np.ndarray)) and len(color) == 3:
        rgb = list(color)
        # Auto-normalize 0–255 integers to 0–1 floats for AE
        if max(rgb) > 1:
            rgb = [float(v) / 255.0 for v in rgb]
        else:
            rgb = [float(v) for v in rgb]
    else:
        raise ValueError("Color must be a name or 3-value RGB list/tuple.")
    return f"[{rgb[0]}, {rgb[1]}, {rgb[2]}]"


def _is_rgb_triplet(color):
    if not isinstance(color, (list, tuple, np.ndarray)):
        return False
    if len(color) != 3:
        return False
    return all(isinstance(v, (int, float, np.integer, np.floating)) for v in color)


def _is_per_point_color(color, n):
    if isinstance(color, str):
        return False
    if isinstance(color, np.ndarray) and color.ndim == 2 and color.shape[0] == n and color.shape[1] == 3:
        return True
    if isinstance(color, (list, tuple)) and len(color) == n and not _is_rgb_triplet(color):
        return True
    return False


def _prepare_scatter_colors(color, n):
    if _is_per_point_color(color, n):
        if isinstance(color, np.ndarray):
            return [list(row) for row in color]
        return list(color)
    if _is_rgb_triplet(color) or isinstance(color, str):
        return [color] * n
    raise ValueError(f"Scatter color must be a single color or a list of {n} colors.")


def _prepare_scatter_radii(radius, n):
    if isinstance(radius, (int, float, np.integer, np.floating)):
        return [float(radius)] * n
    if isinstance(radius, np.ndarray):
        radius = radius.tolist()
    if isinstance(radius, (list, tuple)) and len(radius) == n and all(isinstance(v, (int, float, np.integer, np.floating)) for v in radius):
        return [float(v) for v in radius]
    raise ValueError(f"Scatter radius must be a single number or a list of {n} numeric values.")


# Marker shapes for scatter() / scatter_evolving(). "circle" is special-cased to
# an AE Ellipse so existing output is unchanged; every other marker is drawn with
# an AE Polystar (polygon or star), sized by its Outer Radius so it keeps the same
# "radius" meaning as a circle. ``rotation`` is in degrees (0 points a vertex
# straight up); ``inner_ratio`` is the star's inner/outer radius ratio (None for
# polygons). Aliases mirror matplotlib's single-character marker codes.
_MARKER_SHAPES = {
    "circle":   None,
    "square":   {"star_type": 2, "points": 4, "rotation": 45.0, "inner_ratio": None},
    "diamond":  {"star_type": 2, "points": 4, "rotation": 0.0,  "inner_ratio": None},
    "triangle": {"star_type": 2, "points": 3, "rotation": 0.0,  "inner_ratio": None},
    "pentagon": {"star_type": 2, "points": 5, "rotation": 0.0,  "inner_ratio": None},
    "hexagon":  {"star_type": 2, "points": 6, "rotation": 0.0,  "inner_ratio": None},
    "star":     {"star_type": 1, "points": 5, "rotation": 0.0,  "inner_ratio": 0.5},
}

_MARKER_ALIASES = {
    "o": "circle", "c": "circle",
    "s": "square",
    "d": "diamond",
    "^": "triangle", "tri": "triangle", "t": "triangle",
    "p": "pentagon",
    "h": "hexagon",
    "*": "star",
}


def _resolve_marker(marker):
    """Normalize a marker name/alias to a key in ``_MARKER_SHAPES``."""
    if marker is None:
        return "circle"
    key = str(marker).strip().lower()
    key = _MARKER_ALIASES.get(key, key)
    if key not in _MARKER_SHAPES:
        valid = ", ".join(sorted(_MARKER_SHAPES))
        raise ValueError(f"Unknown marker {marker!r}. Valid markers: {valid}.")
    return key


def _marker_static_jsx(contents_var, shape_var, marker, radius):
    """JSX that adds a fixed-size marker primitive (ellipse or polystar) to a
    shape layer's contents group and sizes it for ``radius`` (a pixel radius)."""
    spec = _MARKER_SHAPES[marker]
    s = ""
    if spec is None:
        s += f"var {shape_var} = {contents_var}.addProperty('ADBE Vector Shape - Ellipse');\n"
        s += f"{shape_var}.property('ADBE Vector Ellipse Size').setValue([{radius * 2}, {radius * 2}]);\n"
    else:
        s += f"var {shape_var} = {contents_var}.addProperty('ADBE Vector Shape - Star');\n"
        s += f"{shape_var}.property('ADBE Vector Star Type').setValue({spec['star_type']});\n"
        s += f"{shape_var}.property('ADBE Vector Star Points').setValue({spec['points']});\n"
        s += f"{shape_var}.property('ADBE Vector Star Rotation').setValue({spec['rotation']});\n"
        s += f"{shape_var}.property('ADBE Vector Star Outer Radius').setValue({radius});\n"
        if spec["inner_ratio"] is not None:
            s += f"{shape_var}.property('ADBE Vector Star Inner Radius').setValue({radius * spec['inner_ratio']});\n"
    return s


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# --- Shared JSX preamble pieces ---------------------------------------------
# These are emitted once per generated script. AEGraph emits them inline for a
# single composition; AEFigure emits them once and then appends each panel's
# body (wrapped in an IIFE) so panels share one comp without var collisions.

_JSX_EASY_EASE_FN = """
// --- Helper Function: Apply Easy Ease to All Keyframes of a Property ---
function applyEasyEase(prop, easeSpeed, easeInfluence) {
    if (!prop || !prop.numKeys || prop.numKeys < 2) return;

    // Determine the number of dimensions. Spatial properties (Position,
    // Anchor Point) always want exactly ONE KeyframeEase per key -- even
    // though their *value* is a 2- or 3-element array -- because temporal
    // ease for a spatial property controls speed along the whole motion
    // path, not each axis independently. Passing one-per-axis there throws
    // "Value array does not have 1 elements". Non-spatial multi-dimensional
    // properties (Scale, Color, etc.) still want one ease per component.
    var isSpatial = prop.propertyValueType === PropertyValueType.TwoD_SPATIAL ||
                     prop.propertyValueType === PropertyValueType.ThreeD_SPATIAL;
    var dim = isSpatial ? 1 : (prop.value.length ? prop.value.length : 1);

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
"""

_JSX_HELPER_FUNCTIONS = """
// --- Helper: set HOLD or LINEAR interpolation on every keyframe of a property ---
// For LINEAR, spatial tangents are zeroed too so motion paths stay straight
// (truly linear, never bezier). The spatial call is guarded for value-only
// properties (color, size, shape) that don't support spatial tangents.
function setKeyInterp(prop, hold) {
    if (!prop || !prop.numKeys) return;
    var t = hold ? KeyframeInterpolationType.HOLD : KeyframeInterpolationType.LINEAR;
    for (var k = 1; k <= prop.numKeys; k++) {
        prop.setInterpolationTypeAtKey(k, t, t);
        if (!hold) {
            try {
                var dim = (prop.value && prop.value.length) ? prop.value.length : 0;
                if (dim === 2) prop.setSpatialTangentsAtKey(k, [0, 0], [0, 0]);
                else if (dim === 3) prop.setSpatialTangentsAtKey(k, [0, 0, 0], [0, 0, 0]);
            } catch (e) {}
        }
    }
}

// Repeat a keyframed segment for the rest of the composition (studio loops, etc.).
// Requires the first and last keyframe values to match for a seamless cycle.
function loopOutCycle(prop) {
    if (!prop || !prop.numKeys || prop.numKeys < 2) return;
    prop.expression = 'loopOut("cycle")';
}

function maybeSetValue(propGroup, propName, value) {
    if (!propGroup) return;
    var prop = propGroup.property(propName);
    if (prop) prop.setValue(value);
}

// Import a still image once per absolute path (reuses project footage items).
function __aegraphImportFootage(absPath) {
    if (typeof __aegraphFootageByPath === 'undefined') {
        __aegraphFootageByPath = {};
    }
    var key = absPath.toLowerCase();
    if (__aegraphFootageByPath[key]) {
        return __aegraphFootageByPath[key];
    }
    var imgFile = new File(absPath);
    if (!imgFile.exists) {
        return null;
    }
    var fsKey = imgFile.fsName.toLowerCase();
    for (var __ai = 1; __ai <= app.project.numItems; __ai++) {
        var __aItem = app.project.item(__ai);
        if (__aItem instanceof FootageItem && __aItem.mainSource && __aItem.mainSource.file) {
            if (__aItem.mainSource.file.fsName.toLowerCase() === fsKey) {
                __aegraphFootageByPath[key] = __aItem;
                return __aItem;
            }
        }
    }
    var __aOpts = new ImportOptions(imgFile);
    var __aFoot = app.project.importFile(__aOpts);
    __aegraphFootageByPath[key] = __aFoot;
    return __aFoot;
}

function addTextAnimator(layer) {
    if (!layer) return null;
    var textProps = layer.property('ADBE Text Properties');
    if (!textProps) return null;
    var animators = textProps.property('ADBE Text Animators');
    if (!animators) return null;
    return animators.addProperty('ADBE Text Animator');
}

function addTextSelector(animator) {
    if (!animator) return null;
    var selectors = animator.property('ADBE Text Selectors');
    if (!selectors) return null;
    return selectors.addProperty('ADBE Text Selector');
}

function maybeSetValueDeep(group, matchName, value) {
    if (!group) return false;

    for (var i = 1; i <= group.numProperties; i++) {
        var p = group.property(i);

        if (p.matchName === matchName) {
            p.setValue(value);
            return true;
        }

        if (p.numProperties && p.numProperties > 0) {
            if (maybeSetValueDeep(p, matchName, value)) return true;
        }
    }

    return false;
}
"""

_JSX_FILM_STYLE_FN = """
// --- Experimental film style: per-element edge treatment -------------------
// Skips backgrounds, nulls, footage/solids (paper, light leak, distress,
// adjustment layers) and the film-style infrastructure itself, leaving only
// the actual chart/text content (shapes + text layers) to be treated.
function __filmStyleSkip(layer) {
    var skipPrefixes = ["PlotAnchor", "GraphBG", "FigureBG", "legendNull", "cmapNull", "AEGraph_Film"];
    for (var i = 0; i < skipPrefixes.length; i++) {
        if (layer.name.indexOf(skipPrefixes[i]) === 0) return true;
    }
    if (layer.nullLayer) return true;
    if (layer.adjustmentLayer) return true;
    var isVector = (layer.matchName === "ADBE Vector Layer");
    var isText = (layer instanceof TextLayer) || (layer.matchName === "ADBE Text Layer");
    if (!isVector && !isText) return true;
    return false;
}

// Maps an AEGraph-assigned layer name (see the various ``.name = "..."``
// calls throughout _generate_jsx) to a coarse "kind" string, so per-kind
// overrides (e.g. {"scatter": False}) can target just one element type.
var __FILM_KIND_TABLE = [
    ["ScatterEvolving_", "scatter"], ["ScatterImg_", "scatter"], ["ScatterMatte_", "scatter"],
    ["ScatterOutline_", "scatter"], ["Scatter_", "scatter"],
    ["LineEvolving_", "line"], ["Line_", "line"],
    ["BarEvolving_", "bar"], ["BarhEvolving_", "bar"], ["StackBar_", "bar"], ["BarhBar_", "bar"], ["Bar_", "bar"], ["Histogram_", "bar"],
    ["PieLeader_", "pie"], ["Pie_", "pie"],
    ["HeatmapEvolving_", "heatmap"], ["Heatmap_", "heatmap"],
    ["QuiverEvolving_", "quiver"], ["Quiver_", "quiver"],
    ["Annotation_", "annotation"],
    ["EvolvingText_", "evolving_text"],
    ["Grid_H_", "grid"], ["Grid_V_", "grid"],
    ["XTick_", "tick"], ["YTick_", "tick"],
    ["ErrorBar_", "errorbar"], ["RefLine_", "refline"], ["Band_", "band"],
    ["GradientArea_", "band"],
    ["cmapStrip", "colorbar"], ["cmapTick_", "colorbar"]
];

function __filmStyleKind(layer) {
    var name = layer.name || "";
    for (var i = 0; i < __FILM_KIND_TABLE.length; i++) {
        if (name.indexOf(__FILM_KIND_TABLE[i][0]) === 0) return __FILM_KIND_TABLE[i][1];
    }
    if (layer instanceof TextLayer || layer.matchName === "ADBE Text Layer") return "text";
    return "other";
}

// True unless ``kindMap`` explicitly disables ``kind`` (kindMap[kind] === false).
function __filmKindOn(kindMap, kind) {
    if (kindMap && kindMap.hasOwnProperty(kind)) return !!kindMap[kind];
    return true;
}

// Per-kind Roughen Edges border. Returns null to skip the effect for this kind.
// kindMap values: omitted -> defaultBorder; true -> defaultBorder;
// false/0 -> off; positive number -> that border width.
function __filmKindRoughenBorder(kindMap, kind, defaultBorder) {
    if (!kindMap || !kindMap.hasOwnProperty(kind)) return defaultBorder;
    var v = kindMap[kind];
    if (v === false || v === 0) return null;
    if (v === true) return defaultBorder;
    if (typeof v === "number") {
        if (v <= 0) return null;
        return v;
    }
    return defaultBorder;
}

function applyFilmElementFX(layer, opts) {
    var kind = __filmStyleKind(layer);
    if (opts.roughen) {
        var roughenBorder = __filmKindRoughenBorder(opts.roughenKinds, kind, opts.roughenBorder);
        if (roughenBorder !== null) {
            try {
                var roughen = layer.property("Effects").addProperty("ADBE Roughen Edges");
                if (roughen) {
                    maybeSetValue(roughen, "ADBE Roughen Edges-0001", opts.roughenEdgeType);
                    maybeSetValue(roughen, "ADBE Roughen Edges-0010", opts.roughenEdgeColor);
                    maybeSetValue(roughen, "ADBE Roughen Edges-0002", roughenBorder);
                    maybeSetValue(roughen, "ADBE Roughen Edges-0003", opts.roughenSharpness);
                    maybeSetValue(roughen, "ADBE Roughen Edges-0004", 1);
                    maybeSetValue(roughen, "ADBE Roughen Edges-0005", opts.roughenScale);
                }
            } catch (e) {}
        }
    }
    if (opts.blur && __filmKindOn(opts.blurKinds, kind)) {
        try {
            var fblur = layer.property("Effects").addProperty("ADBE Gaussian Blur 2");
            if (fblur) {
                maybeSetValue(fblur, "ADBE Gaussian Blur 2-0001", opts.blurAmount);
                maybeSetValue(fblur, "ADBE Gaussian Blur 2-0003", 1);
            }
        } catch (e) {}
    }
    if (opts.multiply && __filmKindOn(opts.multiplyKinds, kind)) {
        try { layer.blendingMode = BlendingMode.MULTIPLY; } catch (e) {}
    }
}

function __runFilmStyleElementPass(compRef, opts) {
    for (var __fi = 1; __fi <= compRef.numLayers; __fi++) {
        var __flayer = compRef.layer(__fi);
        if (!__filmStyleSkip(__flayer)) {
            applyFilmElementFX(__flayer, opts);
        }
    }
}
"""


def _jsx_comp_header(comp_name, comp_width, comp_height, fps):
    """JSX that finds-or-creates the target composition and binds it to `comp`."""
    return f"""
var comp = app.project.activeItem;
if (!comp || !(comp instanceof CompItem) || comp.name != '{comp_name}') {{
    comp = app.project.items.addComp('{comp_name}', {comp_width}, {comp_height}, 1, {config.comp_duration}, {fps});
    comp.openInViewer();
}}
"""


def _write_jsx_file(jsx: str, filename: str = "", folder_path=UNSET) -> str:
    """Write a JSX string to disk and return the absolute path."""
    if folder_path is UNSET:
        folder_path = config.jsx_output_dir
    if filename == "":
        filename = f"AEGraph_{get_time()}.jsx"
    os.makedirs(folder_path, exist_ok=True)
    out_path = os.path.join(folder_path, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(jsx)
    print(f"JSX script saved to {out_path}")
    return out_path


def _run_jsx_file(abs_path: str, ae_version=UNSET):
    """Run a saved .jsx file in After Effects (macOS via AppleScript, Windows via AfterFX.exe)."""
    import platform
    import subprocess

    if ae_version is UNSET:
        ae_version = config.ae_version

    abs_path = os.path.abspath(abs_path)
    print(f"[AEGraph] Attempting to run JSX script: {abs_path}")
    system = platform.system()
    try:
        if system == "Darwin":
            apple_script = f'''
            tell application "{ae_version}"
                activate
                DoScriptFile "{abs_path}"
            end tell
            '''
            result = subprocess.run(
                ["osascript", "-e", apple_script],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print("[AEGraph] Error running script:")
                print(result.stderr)
            else:
                print(f"[AEGraph] Script sent to After Effects: {abs_path}")
        elif system == "Windows":
            afterfx_path = rf"C:\Program Files\Adobe\{ae_version}\Support Files\AfterFX.exe"
            print(f"[AEGraph] Windows AfterFX path: {afterfx_path}")
            subprocess.Popen(
                [afterfx_path, "-r", abs_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
            print(f"[AEGraph] Script sent to After Effects: {abs_path}")
        else:
            raise RuntimeError(f"Unsupported operating system: {system}")
    except Exception as e:
        print(f"[AEGraph] Exception during render: {e}")


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
                 width=UNSET,
                 height=UNSET,
                 comp_name="AEGraph_Comp",
                 bg_color=UNSET, comp_width=None,
                 comp_height=None, compwidth=None,
                 compheight=None, position=None,
                 show_all_points=False, drop_shadow=False,
                 full_bg=UNSET,
                 distress_texture=UNSET,
                 cinematic_effects=False, wiggle=False,
                 film_style=UNSET,
                 easy_ease=UNSET,
                 ease_speed=UNSET,
                 ease_influence=UNSET,
                 meta_easy_ease=UNSET,
                 meta_ease_speed=UNSET,
                 meta_ease_influence=UNSET,
                 animate_opacity=True,
                 animate_axes=True,
                 text_animate=UNSET,
                 ui_color: Union[str, List[float], Tuple[float, float, float]] = UNSET,
                 font_scale: float = UNSET,
                 font=UNSET,
                 font_title=UNSET,
                 font_subtitle=UNSET,
                 font_label=UNSET,
                 font_tick=UNSET,
                 font_legend=UNSET,
                 font_body=UNSET,
                 bg_stroke_width = 0,
                 bg_stroke_color = [0.15, 0.15, 0.15],
                 fps=60,
                 show_xaxis=True,
                 show_yaxis=True,
                 xaxis_location="auto",
                 yaxis_location="auto",
                 plot_frame=False,
                 label_color=UNSET,
                 subtitle_color=UNSET,
                 grid_color=UNSET,
                 theme=None,
                 view_keyframes=None):
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
            full_bg (bool): Whether to make background cover full composition (True) or just graph area (False).
                Defaults to ``aegraph_config.config.full_bg`` (True by default).
            distress_texture (int, optional): Distress texture preset to place above GraphBG but below
                all other graph layers. Currently supports `1` for `distress_textures/grunge1.jpg`
                (default: None)
            cinematic_effects (bool): Whether to add cinematic adjustment layer with vignette effects (default: False)
            wiggle (bool): Whether to add wiggle adjustment layer with Turbulent Displace (default: False)
            film_style (bool): Experimental documentary-style treatment: paper-texture
                background, front-heavy push-in zoom, looping light-leak overlay, a global
                adjustment stack (temporal posterize + exposure flicker, vignette, soft edge
                blur), and Roughen Edges + Gaussian Blur + Multiply blending on every chart/text
                layer. Fine-tune via ``aegraph_config.config.film_style_*``.
                Defaults to ``aegraph_config.config.film_style`` (False).
            easy_ease (bool): Whether to apply easy ease to all animation keyframes.
                Defaults to ``aegraph_config.config.easy_ease`` (True).
            ease_speed (int): Easy ease speed percentage. Defaults to
                ``aegraph_config.config.ease_speed`` (0).
            ease_influence (int): Easy ease influence percentage. Defaults to
                ``aegraph_config.config.ease_influence`` (33).
            meta_easy_ease (bool): Whether to ease the *sequence* of element
                entrances in multi-element plots (scatter, histogram, bar_graph,
                barh, ...). With it on, the staggered in-points of the elements
                follow an easy-ease curve, so the overall reveal eases in and out
                instead of sweeping at a constant rate. Defaults to
                ``aegraph_config.config.meta_easy_ease`` (True).
            meta_ease_speed (int): Meta easy ease speed percentage. Defaults to
                ``aegraph_config.config.meta_ease_speed`` (0).
            meta_ease_influence (int): Meta easy ease influence percentage.
                Higher values cushion the start/end of the sweep more strongly.
                Defaults to ``aegraph_config.config.meta_ease_influence`` (33).
            text_animate (float): Duration (seconds) of the slide-in animation
                for title, subtitle, axis labels, legend entries, and
                annotations. This also paces the timing of the surrounding
                chrome that reveals alongside that text (axis spine trim
                paths, tick marks/labels, grid lines, and the colorbar), since
                they're all keyed off the same reveal duration. Does not
                affect data element animations (``scatter``/``plot``/
                ``bar_graph`` etc. have their own ``animate`` argument).
                Defaults to ``aegraph_config.config.text_animate`` (1.5).
            fps (int): Frame rate for the After Effects composition (default: 24)
            ui_color (str or list/tuple): Color for all non-data UI elements like axes, tick marks,
                all tick/label/title/legend/annotation text (default: "black"). Accepts named colors
                from COLOR_NAMES or RGB values in 0–1 floats or 0–255 ints.
            font_scale (float): Global scale multiplier for all text sizes (ticks, labels, title,
                legend, and annotations). Default 1.0.
            show_xaxis (bool): Draw the horizontal x-axis spine, tick marks, and x tick labels
                (default: True).
            show_yaxis (bool): Draw the vertical y-axis spine (e.g. at x=0), tick marks, and y tick
                labels (default: True). Set False when x=0 is not meaningful for your data.
            xaxis_location (str or number): Where to draw the horizontal spine — ``'auto'`` (zero
                if in range, else bottom), ``'bottom'``, ``'top'``, or a y data value (e.g. ``0``).
            yaxis_location (str or number): Where to draw the vertical spine — ``'auto'`` (zero
                if in range, else left), ``'left'``, ``'right'``, or an x data value.
            plot_frame (bool): If True, draw the top and right spines opposite the primary
                x/y axes so the plot area is a full rectangle (matplotlib-style box).
        """
        # Sizing resolution (backward-compatible):
        # - Comp: explicit compwidth/compheight > comp_width/comp_height > graph
        #   width/height if given (preserves old single-arg scripts) > config
        #   defaults (1920x1080).
        # - Graph: explicit width/height > round(comp * config.graph_scale).
        # Priority: compwidth/compheight > comp_width/comp_height > width/height > config
        graph_width_given = width is not UNSET
        graph_height_given = height is not UNSET

        if compwidth is not None:
            self.comp_width = compwidth
        elif comp_width is not None:
            self.comp_width = comp_width
        elif graph_width_given:
            self.comp_width = width
        else:
            self.comp_width = config.comp_width

        if compheight is not None:
            self.comp_height = compheight
        elif comp_height is not None:
            self.comp_height = comp_height
        elif graph_height_given:
            self.comp_height = height
        else:
            self.comp_height = config.comp_height

        self.width = width if graph_width_given else int(round(self.comp_width * config.graph_scale))
        self.height = height if graph_height_given else int(round(self.comp_height * config.graph_scale))
        self.comp_name = comp_name

        # Apply a theme into the module-level config first so any UNSET args
        # below pick up the theme defaults. Passing `theme=` on the constructor
        # is sugar for calling ``aegraph_config.apply_theme(name)`` beforehand.
        if theme is not None:
            from aegraph_config import apply_theme as _apply_theme  # local import to avoid cycles
            _apply_theme(theme)

        # Theme-driven defaults: caller-supplied wins, otherwise fall back to
        # the (possibly theme-mutated) module-level config.
        self.bg_color = config.bg_color if bg_color is UNSET else bg_color
        self.ui_color = config.ui_color if ui_color is UNSET else ui_color
        self.label_color = config.label_color if label_color is UNSET else label_color
        self.subtitle_color = config.subtitle_color if subtitle_color is UNSET else subtitle_color
        self.distress_texture = config.distress_texture if distress_texture is UNSET else distress_texture
        self.position = position  # (x, y) in comp coordinates, or None for center
        self.show_all_points = show_all_points
        self.drop_shadow = drop_shadow  # Global drop shadow setting
        self.full_bg = config.full_bg if full_bg is UNSET else full_bg
        self.elements = []  # List of plot elements (dicts)
        self.title = None
        self.subtitle = None
        self.xlabel = None
        self.ylabel = None
        self.legend = []
        self.legend_style = 'color_only'  # 'color_only' or 'line_style'
        self.legend_pos = None            # None = auto; string = named region; (x,y) = data coords
        self.xlim = None
        self.ylim = None
        self.xscale = "linear"  # 'linear' or 'log' (matplotlib-style)
        self.yscale = "linear"
        # Keyframed view window (animated x/y limits). Parsed into a sorted list
        # of (t, xmin, xmax, ymin, ymax) tuples by ``set_view_keyframes``.
        self._view_kf = None
        self._view_animated = False
        self._view_hold = False
        self._view_sample_fps = None
        self._view_ease = True
        self._view_ease_speed = None
        self._view_ease_influence = None
        self._view_visibility_fade = 0.35  # seconds (wall-clock, independent of fps); fade when crossing the view window edge
        self._adaptive_ticks = True
        self._adaptive_tick_target_n = 7
        self._adaptive_tick_fade = 0.4  # seconds (wall-clock, independent of fps); crossfade when a finer tick level's density threshold is crossed
        self._adaptive_tick_min_opacity = 0.72
        if view_keyframes is not None:
            self.set_view_keyframes(view_keyframes)
        self.xticks = None  # X-axis tick positions and labels
        self.yticks = None  # Y-axis tick positions and labels
        self.show_grid = False   # Whether to show grid (call .grid() to enable)
        self.grid_color = config.grid_color if grid_color is UNSET else grid_color
        self.grid_alpha = 0.3     # Grid opacity
        self.grid_linewidth = config.grid_linewidth
        self.grid_linestyle = "dashed"  # Grid line style (default: dashed)
        self.grid_dash_size = config.dash_size  # Grid dash size scale factor
        self.hide_horizontal = False  # Hide horizontal grid lines
        self.hide_vertical = False  # Hide vertical grid lines
        self.show_tick_labels = True
        self.show_xaxis = bool(show_xaxis)
        self.show_yaxis = bool(show_yaxis)
        self.xaxis_location = xaxis_location
        self.yaxis_location = yaxis_location
        self.plot_frame = bool(plot_frame)
        self.cinematic_effects = cinematic_effects  # Whether to add cinematic adjustment layer
        self.wiggle = wiggle  # Whether to add wiggle adjustment layer with Turbulent Displace
        self.film_style = bool(config.film_style if film_style is UNSET else film_style)
        self._film_style_overrides = {}  # set via film_style_parameters()
        # Easy ease (per-keyframe smoothing). Unset args fall back to config.
        self.easy_ease = config.easy_ease if easy_ease is UNSET else easy_ease
        self.ease_speed = config.ease_speed if ease_speed is UNSET else ease_speed
        self.ease_influence = config.ease_influence if ease_influence is UNSET else ease_influence
        # "Meta" easy ease eases the *sequence* of element entrances (the in-point
        # of each bar/point/arrow) so the overall sweep of a multi-element plot
        # accelerates and decelerates instead of marching in at a constant rate.
        # Unset args fall back to the config defaults.
        self.meta_easy_ease = config.meta_easy_ease if meta_easy_ease is UNSET else meta_easy_ease
        self.meta_ease_speed = config.meta_ease_speed if meta_ease_speed is UNSET else meta_ease_speed
        self.meta_ease_influence = config.meta_ease_influence if meta_ease_influence is UNSET else meta_ease_influence
        self.animate_opacity = animate_opacity  # Enable/disable opacity fade animations
        self.animate_axes = animate_axes  # Enable/disable Trim Paths animation for axes
        # Slide-in animation duration for title/subtitle/labels/legend/annotations.
        self.text_animate = float(config.text_animate) if text_animate is UNSET else float(text_animate)
        self.fps = fps  # Frame rate for the After Effects composition
        self.bg_stroke_color = bg_stroke_color
        self.bg_stroke_width = bg_stroke_width
        self.font_scale = float(font_scale) if font_scale not in (None, UNSET) else float(config.font_scale)

        # Per-role fonts (After Effects PostScript names). Resolution order for
        # each role: an explicit per-role arg wins; otherwise the ``font=``
        # override applies to every role; otherwise the (possibly theme-mutated)
        # module-level config default for that role is used. Pass ``font=`` to
        # set one font for the whole figure, or any of the per-role args to
        # override just that element.
        def _resolve_font(role_arg, cfg_attr):
            if role_arg is not UNSET and role_arg is not None:
                return role_arg
            if font is not UNSET and font is not None:
                return font
            return getattr(config, cfg_attr)

        self.font_title = _resolve_font(font_title, "font_title")
        self.font_subtitle = _resolve_font(font_subtitle, "font_subtitle")
        self.font_label = _resolve_font(font_label, "font_label")
        self.font_tick = _resolve_font(font_tick, "font_tick")
        self.font_legend = _resolve_font(font_legend, "font_legend")
        self.font_body = _resolve_font(font_body, "font_body")
        # Tick formatting flags
        self.percent_tick_labels = False  # When True, x-ticks render as absolute percentages
        self._xtick_labels_auto = False
        self._ytick_labels_auto = False
        # On log axes, decade labels whose exponent magnitude is >= this
        # threshold render compactly as "10^n" instead of spelling out all the
        # digits (e.g. 10 -> 10^1, 1000 -> 10^3, 100000 -> 10^5). With the
        # default of 1, every decade above/below 1 uses superscript power
        # notation (1 stays "1"). Raise it to keep small decades spelled out
        # (e.g. 4 -> 10/100/1000 stay literal), or set to None to always spell
        # out.
        self.log_power_label_threshold = 1

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

    def plot(self, x, y, color="blue", label=None, linewidth=4, linestyle="solid", dash_size=UNSET, alpha=1.0, animate=UNSET, delay = 0.0, drop_shadow=False, ease_speed=None, ease_influence=None, **kwargs):
        """
        Add a line plot to the graph.
        x, y: Data points (list, tuple, numpy array, or pandas Series).
        color: Color name or RGB list.
        label: Legend label.
        linewidth: Stroke width.
        linestyle: Line style - 'solid' (default), 'dashed'/'--', 'dotted'/':', or 'dashdot'/'-.'.
        dash_size: Scale factor for dash lengths (default: 1.0). Only applies to non-solid linestyles.
        alpha: Line opacity from 0 (transparent) to 1 (opaque) (default: 1.0).
        animate: Animation duration in seconds. Defaults to
            ``aegraph_config.config.line_animate``.
        drop_shadow: Whether to add drop shadow effect (default: False).
        ease_speed: Optional per-element easy ease speed override. Uses AEGraph default when None.
        ease_influence: Optional per-element easy ease influence override. Uses AEGraph default when None.
        """
        if animate is UNSET:
            animate = config.line_animate
        if dash_size is UNSET:
            dash_size = config.dash_size
        x_name = getattr(x, "name", None) if pd is not None and isinstance(x, pd.Series) else None
        y_name = getattr(y, "name", None) if pd is not None and isinstance(y, pd.Series) else None
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
            "alpha": alpha,
            "animate": animate,
            "drop_shadow": drop_shadow,
            "delay": delay,
            "ease_speed": ease_speed,
            "ease_influence": ease_influence,
            "x_name": x_name,
            "y_name": y_name,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def scatter(self, x, y, color="red", label=None, radius=8, radius_size="literal", max_radius=UNSET, alpha=1.0, delay = 0.0, animate=UNSET, drop_shadow=False, bar_anim_times=None, point_duration=None, point_anim_times=UNSET, point_start_times=None, point_reveal_duration=0.3, bar_duration=None, ease_speed=None, ease_influence=None, meta_easy_ease=None, meta_ease_speed=None, meta_ease_influence=None, c=None, gradient=None, discrete=None, outline=UNSET, outline_width=UNSET, outline_alpha=1.0, outline_color=None, marker="circle", images=None, colorbar=True, clip_to_view=UNSET, move_from_x=None, move_from_y=None, **kwargs):
        """
        Add a scatter plot to the graph.
        x, y: Data points (list, tuple, numpy array, or pandas Series).
        color: Color name, RGB list, or per-point list of colors.
        label: Legend label.
        radius: Point radius or per-point list of radii.
        radius_size: How to interpret ``radius`` when a per-point list is given.
            - ``"literal"`` (default): values are taken as pixel radii as-is.
            - ``"dynamic_range"``: values are scaled proportionally so the
              largest equals ``max_radius``. Scalar radii are not affected.
              Useful when ``radius`` comes from raw data (e.g. populations)
              that would otherwise be huge.
        max_radius: Target radius (px) for the largest point when
            ``radius_size="dynamic_range"``. Defaults to
            ``aegraph_config.config.scatter_max_radius``; pass this instead of
            mutating the config when you just want a one-off override for this
            call.
        alpha: Marker fill opacity from 0 (transparent) to 1 (opaque)
            (default: 1.0). Controls the fill only, not the outline.
        animate: Total animation duration in seconds (points animate sequentially).
            Defaults to ``aegraph_config.config.scatter_animate``.
        drop_shadow: Whether to add drop shadow effect (default: False).
        bar_anim_times: Optional list of per-point animation durations (overrides sequential timing).
        point_anim_times: Per-point animation duration. Scalar applies to every
            point; list assigns one duration per point. Defaults to
            ``aegraph_config.config.scatter_point_anim_times``.
        ease_speed: Optional per-element easy ease speed override. Uses AEGraph default when None.
        ease_influence: Optional per-element easy ease influence override. Uses AEGraph default when None.
        meta_easy_ease: Optional per-element override for easing the staggered
            point entrances. None uses the AEGraph default.
        meta_ease_speed: Optional per-element meta easy ease speed override.
            Like AE easy ease, ``0`` gives the fullest cushion; higher values
            flatten the sweep toward a constant rate. None uses the AEGraph default.
        meta_ease_influence: Optional per-element meta easy ease influence
            override (higher = more pronounced slow-fast-slow sweep). None uses
            the AEGraph default.
        c: Optional data array (list, np.ndarray, or pandas Series) of values
            mapped to a color gradient. Length must match ``x``.
        gradient: Controls the color gradient. Accepts:
            - 2-tuple/list ``(low, high)`` of colors (overrides config endpoints).
            - data array (acts like passing ``c=``).
            When ``c`` or ``gradient`` is set, the gradient endpoints default to
            ``config.gradient_low`` and ``config.gradient_high``.
        discrete: Controls discrete vs. smooth color mapping for ``c``/
            ``gradient`` data. ``None`` (default) auto-detects: integer-only
            data with a small number of levels renders as discrete color bands
            (one per integer), and the colorbar shows banded swatches instead
            of a smooth gradient. ``True`` forces discrete; ``False`` forces a
            smooth gradient.
        outline: Whether to draw a darker marker outline. Defaults to
            ``aegraph_config.config.scatter_outline``.
        outline_width: Stroke width for marker outlines in pixels. Defaults to
            ``aegraph_config.config.scatter_outline_width``.
        outline_alpha: Outline opacity from 0 (transparent) to 1 (opaque)
            (default: 1.0). Independent of the fill ``alpha``.
        marker: Marker shape for every point (default ``"circle"``). One of
            ``"circle"``, ``"square"``, ``"diamond"``, ``"triangle"``,
            ``"pentagon"``, ``"hexagon"``, or ``"star"`` (matplotlib single-char
            aliases like ``"o"``, ``"s"``, ``"^"``, ``"*"`` also work). Non-circle
            markers keep the same ``radius`` meaning (outer radius in pixels).
        colorbar: Whether to show the gradient colorbar (default ``True``).
            Set to ``False`` to keep the gradient dot colours while hiding the
            colorbar legend that appears on the right side of the chart.
        images: Optional per-point image paths (same length as ``x``/``y``).
            When set, each marker is filled with its image clipped to the
            marker shape (via an alpha track matte) instead of a solid
            ``color``. Missing paths fall back to solid ``color``.
        clip_to_view: When the graph uses an animated view window (see
            ``set_view_keyframes``), whether markers should shrink away
            whenever their data point falls outside the *current* view window
            (rather than remaining visible past the plot frame's edge).
            Defaults to ``aegraph_config.config.scatter_clip_to_view``. Has no
            effect when the view isn't animated.
        point_start_times: Optional per-point explicit entrance start times
            (seconds, same length as ``x``/``y``), overriding the default
            evenly-spread ``linspace(0, animate, n)`` schedule. Used e.g. by
            ``lollipop(dot_follows_stem=True)`` to line dots up frame-for-frame
            with their paired stems.
        move_from_x, move_from_y: Optional per-point (or scalar) starting
            coordinate(s) in data space. When given, the marker's *position*
            animates from ``(move_from_x, move_from_y)`` to ``(x, y)`` over its
            entrance window instead of popping in at a fixed spot (the usual
            scale grow-in is skipped in favor of this slide). Omit whichever
            of the two you don't want to move; it defaults to the point's own
            final coordinate on that axis.
        point_reveal_duration: Seconds for the scale-in reveal when using
            ``move_from_x`` / ``move_from_y`` (default ``0.3``). Each marker
            stays at scale 0 until its entrance, then eases up to full size
            over this duration while sliding along its stem.
        """
        marker = _resolve_marker(marker)
        if animate is UNSET:
            animate = config.scatter_animate
        if point_anim_times is UNSET:
            point_anim_times = config.scatter_point_anim_times
        if outline is UNSET:
            outline = config.scatter_outline
        if outline_width is UNSET:
            outline_width = config.scatter_outline_width
        if max_radius is UNSET:
            max_radius = config.scatter_max_radius
        if clip_to_view is UNSET:
            clip_to_view = config.scatter_clip_to_view

        # Capture column names before stripping to .values for auto title/labels.
        x_name = getattr(x, "name", None) if pd is not None and isinstance(x, pd.Series) else None
        y_name = getattr(y, "name", None) if pd is not None and isinstance(y, pd.Series) else None

        gradient_endpoints = (config.gradient_low, config.gradient_high)
        gradient_data = None
        if gradient is not None:
            if isinstance(gradient, (tuple, list)) and len(gradient) == 2 and (
                isinstance(gradient[0], str) or _is_rgb_triplet(gradient[0])
            ) and (isinstance(gradient[1], str) or _is_rgb_triplet(gradient[1])):
                gradient_endpoints = (gradient[0], gradient[1])
            else:
                gradient_data = gradient
        if c is not None:
            gradient_data = c

        # Column name of the gradient source (used as the colorbar axis label).
        c_name = None
        if pd is not None and isinstance(gradient_data, pd.Series):
            c_name = getattr(gradient_data, "name", None)

        # Support pandas Series
        if pd is not None:
            if isinstance(x, pd.Series):
                x = x.values
            if isinstance(y, pd.Series):
                y = y.values
            if isinstance(color, pd.Series):
                color = color.values
            if isinstance(radius, pd.Series):
                radius = radius.values
            if isinstance(gradient_data, pd.Series):
                gradient_data = gradient_data.values

        # Apply radius_size policy. "literal" leaves the values alone; "dynamic_range"
        # rescales a per-point radius list proportionally so the max equals
        # `max_radius`. Scalars and missing data are unchanged.
        if radius_size not in ("literal", "dynamic_range"):
            raise ValueError(
                f"radius_size must be 'literal' or 'dynamic_range' (got {radius_size!r})."
            )
        if radius_size == "dynamic_range" and isinstance(radius, (list, tuple, np.ndarray)):
            radius_arr = np.asarray(list(radius), dtype=float)
            if radius_arr.size:
                finite = radius_arr[np.isfinite(radius_arr)]
                if finite.size:
                    max_r = float(np.nanmax(finite))
                    if max_r > 0:
                        scale = float(max_radius) / max_r
                        radius = (radius_arr * scale).tolist()

        # Global radius multiplier (e.g. set to 2.0 when using a 4K comp so
        # dots stay the same visual size as at 1080p).
        rs = float(config.scatter_radius_scale)
        if rs != 1.0:
            if isinstance(radius, (list, tuple, np.ndarray)):
                radius = (np.asarray(list(radius), dtype=float) * rs).tolist()
            else:
                radius = float(radius) * rs

        gradient_low_color = None
        gradient_high_color = None
        gradient_vmin = None
        gradient_vmax = None
        gradient_discrete = False
        gradient_levels = None
        if gradient_data is not None:
            gradient_data = list(gradient_data)
            if len(gradient_data) != len(list(x)):
                raise ValueError("Gradient data length must match x/y length.")
            gradient_discrete, gradient_levels = self._resolve_discrete_levels(
                gradient_data, discrete
            )
            if gradient_discrete:
                # Snap each value to its nearest integer level so point colors
                # land exactly on the discrete band colors of the colorbar.
                snapped = [
                    (min(gradient_levels, key=lambda L: abs(L - v)) if v == v else v)
                    for v in gradient_data
                ]
                color = self._gradient_colors(
                    snapped, gradient_endpoints[0], gradient_endpoints[1],
                    vmin=gradient_levels[0], vmax=gradient_levels[-1],
                )
                gradient_vmin = float(gradient_levels[0])
                gradient_vmax = float(gradient_levels[-1])
            else:
                color = self._gradient_colors(
                    gradient_data, gradient_endpoints[0], gradient_endpoints[1]
                )
                arr = np.asarray(gradient_data, dtype=float)
                if arr.size:
                    gradient_vmin = float(np.nanmin(arr))
                    gradient_vmax = float(np.nanmax(arr))
            gradient_low_color = gradient_endpoints[0]
            gradient_high_color = gradient_endpoints[1]

        # Convert points to list form for filtering and validation
        x_vals = list(x)
        y_vals = list(y)
        if len(x_vals) != len(y_vals):
            raise ValueError("x and y must have the same length")

        images_list = None
        if images is not None:
            images_list = list(images)
            if len(images_list) != len(x_vals):
                raise ValueError("images length must match x/y length")
            images_list = [
                os.path.abspath(str(p)) if p else None for p in images_list
            ]

        move_from_x_list = None
        if move_from_x is not None:
            move_from_x_list = (
                list(move_from_x) if isinstance(move_from_x, (list, tuple, np.ndarray))
                else [move_from_x] * len(x_vals)
            )
            if len(move_from_x_list) != len(x_vals):
                raise ValueError("move_from_x length must match x/y length")
        move_from_y_list = None
        if move_from_y is not None:
            move_from_y_list = (
                list(move_from_y) if isinstance(move_from_y, (list, tuple, np.ndarray))
                else [move_from_y] * len(x_vals)
            )
            if len(move_from_y_list) != len(x_vals):
                raise ValueError("move_from_y length must match x/y length")
        point_start_times_list = None
        if point_start_times is not None:
            point_start_times_list = list(point_start_times)
            if len(point_start_times_list) != len(x_vals):
                raise ValueError("point_start_times length must match x/y length")

        if not self.show_all_points:
            if self.xlim:
                xmin, xmax = self.xlim
            else:
                xmin, xmax = float('-inf'), float('inf')
            if self.ylim:
                ymin, ymax = self.ylim
            else:
                ymin, ymax = float('-inf'), float('inf')

            filtered_indices = [i for i, (xi, yi) in enumerate(zip(x_vals, y_vals)) if xmin <= xi <= xmax and ymin <= yi <= ymax]
            x_vals = [x_vals[i] for i in filtered_indices]
            y_vals = [y_vals[i] for i in filtered_indices]

            if _is_per_point_color(color, len(x)):
                color = [color[i] for i in filtered_indices]
            if isinstance(radius, (list, tuple, np.ndarray)) and len(radius) == len(x):
                radius = [radius[i] for i in filtered_indices]
            if images_list is not None:
                images_list = [images_list[i] for i in filtered_indices]
            if move_from_x_list is not None:
                move_from_x_list = [move_from_x_list[i] for i in filtered_indices]
            if move_from_y_list is not None:
                move_from_y_list = [move_from_y_list[i] for i in filtered_indices]
            if point_start_times_list is not None:
                point_start_times_list = [point_start_times_list[i] for i in filtered_indices]

        x, y = x_vals, y_vals
        self.elements.append({
            "type": "scatter",
            "x": list(x),
            "y": list(y),
            "color": color,
            "label": label,
            "radius": radius,
            "alpha": alpha,
            "animate": animate,
            "drop_shadow": drop_shadow,
            # `bar_anim_times` kept for backward compatibility; prefer `point_anim_times`
            "bar_anim_times": bar_anim_times,
            "point_duration": point_duration,
            "point_anim_times": point_anim_times,
            "point_start_times": point_start_times_list,
            "move_from_x": move_from_x_list,
            "move_from_y": move_from_y_list,
            "point_reveal_duration": float(point_reveal_duration),
            "delay": delay,
            "ease_speed": ease_speed,
            "ease_influence": ease_influence,
            "meta_easy_ease": meta_easy_ease,
            "meta_ease_speed": meta_ease_speed,
            "meta_ease_influence": meta_ease_influence,
            "radius_size": radius_size,
            "x_name": x_name,
            "y_name": y_name,
            "gradient_data": gradient_data,
            "gradient_low": gradient_low_color if colorbar else None,
            "gradient_high": gradient_high_color if colorbar else None,
            "gradient_vmin": gradient_vmin if colorbar else None,
            "gradient_vmax": gradient_vmax if colorbar else None,
            "gradient_name": c_name,
            "gradient_discrete": gradient_discrete,
            "gradient_levels": gradient_levels,
            "outline": bool(outline),
            "outline_width": float(outline_width),
            "outline_alpha": float(outline_alpha),
            "outline_color": outline_color,
            "marker": marker,
            "images": images_list,
            "clip_to_view": bool(clip_to_view),
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def quiver(self, x, y, u, v, scale=1.0, width=3.0, headwidth=3.0,
               headlength=11.0, color="blue", c=None, gradient=None,
               normalize=False, alpha=1.0, animate=1.2, pivot="tail",
               scale_mode="data", label=None, drop_shadow=False,
               meta_easy_ease=None, meta_ease_speed=None,
               meta_ease_influence=None, **kwargs):
        """Vector-field (quiver) plot, matplotlib-style.

        Draws an arrow at each base point ``(x, y)`` pointing along ``(u, v)``.
        Arrow size is fully customizable.

        Parameters:
            x, y: Arrow base coordinates (1D length N, or any array that
                flattens to N -- e.g. ``np.meshgrid`` output).
            u, v: Vector components at each base point (same shape as x, y).
            scale: Arrow-length control. Its meaning depends on ``scale_mode``:
                ``"data"`` (default) -- data-space length drawn per unit vector
                magnitude (use when x/y are in the same units, like the fluid
                demos); ``"comp"`` -- arrow length as a fraction of the plot
                width, sized only *relative* to each other (use when the axes
                are stretched or in wildly different units, e.g. lon/lat maps,
                so arrows stay visible regardless of the data range).
            scale_mode: ``"data"`` or ``"comp"`` -- see ``scale``.
            width: Shaft thickness in pixels.
            headwidth: Arrowhead full width as a multiple of ``width``.
            headlength: Arrowhead length in pixels.
            color: Single arrow color (used when neither ``c`` nor ``gradient``
                is given).
            c: Per-arrow values mapped to a color gradient. Pass ``c=True`` to
                color arrows by their own vector magnitude, or an array of length
                N for a custom field (e.g. vorticity).
            gradient: ``(low, high)`` endpoint colors for the ``c`` mapping.
                Defaults to the theme gradient.
            normalize: If True, every arrow is drawn the same length (direction
                only) -- pair with ``c=True`` to encode magnitude as color.
            alpha: Arrow opacity (0-1).
            animate: Seconds for a staggered fade-in of the field (0 = instant).
            pivot: Where the base point sits on the arrow: ``'tail'`` (default),
                ``'mid'``, or ``'tip'``.
            label: Optional legend label.
            drop_shadow: Whether to add a drop shadow to each arrow.

        Returns self for chaining.
        """
        xa = np.asarray(x, dtype=float).ravel()
        ya = np.asarray(y, dtype=float).ravel()
        ua = np.asarray(u, dtype=float).ravel()
        va = np.asarray(v, dtype=float).ravel()
        if not (len(xa) == len(ya) == len(ua) == len(va)):
            raise ValueError("quiver: x, y, u, v must all flatten to the same length.")

        mag = np.hypot(ua, va)
        if normalize:
            safe = np.where(mag > 0, mag, 1.0)
            ua, va = ua / safe, va / safe
        vx = ua * float(scale)
        vy = va * float(scale)

        # Relative magnitude (0..1) used only by scale_mode="comp" to size each
        # arrow as a fraction of the plot width. Normalized fields are uniform.
        if normalize:
            qrel = np.ones_like(mag)
        else:
            mmax = float(np.max(mag)) or 1.0
            qrel = mag / mmax

        # Per-arrow colors when c / gradient requested; else a single color.
        arrow_colors = None
        if c is not None or gradient is not None:
            if c is None or c is True:
                cvals = mag
            else:
                cvals = np.asarray(c, dtype=float).ravel()
            low, high = self._gradient_endpoints(gradient)
            vmin, vmax = float(np.min(cvals)), float(np.max(cvals))
            span = (vmax - vmin) or 1.0
            arrow_colors = [[low[k] + (high[k] - low[k]) * ((val - vmin) / span)
                             for k in range(3)] for val in cvals]

        self.elements.append({
            "type": "quiver",
            "qx": xa.tolist(),
            "qy": ya.tolist(),
            "qvx": vx.tolist(),
            "qvy": vy.tolist(),
            "arrow_colors": arrow_colors,
            "color": color,
            "width": float(width),
            "headwidth": float(headwidth),
            "headlength": float(headlength),
            "alpha": alpha,
            "animate": animate,
            "pivot": pivot,
            "scale_mode": scale_mode,
            "qscale": float(scale),
            "qrel": qrel.tolist(),
            "label": label,
            "drop_shadow": drop_shadow,
            "meta_easy_ease": meta_easy_ease,
            "meta_ease_speed": meta_ease_speed,
            "meta_ease_influence": meta_ease_influence,
            # Points exposed for limit / tick computation. In "comp" mode the
            # arrows are a fixed pixel length, so only the base points should
            # influence the data range; in "data" mode include the tips too.
            "x": xa.tolist() if scale_mode == "comp" else xa.tolist() + (xa + vx).tolist(),
            "y": ya.tolist() if scale_mode == "comp" else ya.tolist() + (ya + vy).tolist(),
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    @staticmethod
    def _edges_from_centers(centers):
        """Return cell-edge coordinates (length N+1) from N cell centers.

        Edges sit halfway between consecutive centers; the outermost edges
        extend by the same half-step so every center is centered in its cell.
        """
        c = np.asarray(centers, dtype=float)
        if len(c) == 1:
            return np.array([c[0] - 0.5, c[0] + 0.5])
        mid = (c[:-1] + c[1:]) / 2.0
        first = c[0] - (mid[0] - c[0])
        last = c[-1] + (c[-1] - mid[-1])
        return np.concatenate([[first], mid, [last]])

    def _resolve_gradient_specs(self, gradient):
        """Resolve (low_spec, high_spec, low_rgb01, high_rgb01) for a gradient.

        ``low_spec``/``high_spec`` are the raw endpoint colors (names or RGB
        triplets) suitable for the colorbar; the ``*_rgb01`` are the same
        endpoints normalized to 0-1 floats for per-cell color math.
        """
        if (
            isinstance(gradient, (tuple, list))
            and len(gradient) == 2
            and not _is_rgb_triplet(gradient)
        ):
            low_spec, high_spec = gradient[0], gradient[1]
        else:
            low_spec, high_spec = config.gradient_low, config.gradient_high
        return (low_spec, high_spec,
                self._color_to_rgb01(low_spec), self._color_to_rgb01(high_spec))

    def heatmap(self, data, x=None, y=None, gradient=None, vmin=None, vmax=None,
                alpha=1.0, animate=1.0, gap=0.0, origin="lower", label=None,
                colorbar=True, colorbar_label=None, edge_color=None,
                edge_width=0.0, drop_shadow=False, reveal="fade", **kwargs):
        """Heatmap drawn as a grid of colored squares, matplotlib-imshow style.

        Each value in the 2D ``data`` array becomes a filled rectangular cell
        whose color is mapped along a gradient -- "squares of heat" laid out
        over the plot area. Pair with ``x``/``y`` to place the grid on real
        data coordinates (e.g. a longitude/latitude map). Non-finite values
        (``np.nan``/``inf``) render as empty cells, which is handy for masking
        out regions of a map.

        Parameters:
            data: 2D array-like, shape ``(ny, nx)`` -- one value per cell.
            x: Optional cell-center x coordinates (length ``nx``). Defaults to
                ``0..nx-1``. Cell edges are inferred halfway between centers.
            y: Optional cell-center y coordinates (length ``ny``). Defaults to
                ``0..ny-1``.
            gradient: ``(low, high)`` endpoint colors for the value->color map.
                Defaults to the theme gradient
                (``config.gradient_low``/``gradient_high``).
            vmin, vmax: Value range mapped to the gradient ends. Default to the
                data's finite min/max.
            alpha: Cell opacity (0-1).
            animate: Seconds for the cells' fade-in (0 = instant). See
                ``reveal`` for how the fade is distributed across the grid.
            reveal: How the grid animates in over ``animate`` seconds.
                ``"fade"`` (default) fades the whole grid in together;
                ``"diagonal"`` fades each cell in individually, sweeping
                from the bottom-left corner up to the top-right corner.
            gap: Fraction (0-1) of each cell inset on all sides, producing a
                tiled "mosaic" look with space between squares. ``0`` = cells
                touch (continuous field).
            origin: ``"lower"`` (default) puts row 0 at the bottom (natural for
                axes); ``"upper"`` puts row 0 at the top (like ``imshow``).
            label: Optional legend label.
            colorbar: Whether this heatmap drives the colorbar (default True).
            colorbar_label: Optional colorbar axis label.
            edge_color: Optional cell border color (name or RGB). ``None`` = no
                border.
            edge_width: Cell border width in pixels (used when ``edge_color``).
            drop_shadow: Whether to add a drop shadow to each cell.

        Returns self for chaining.
        """
        grid = np.asarray(data, dtype=float)
        if grid.ndim != 2:
            raise ValueError("heatmap: data must be a 2D array (ny x nx).")
        ny, nx = grid.shape

        if x is None:
            xc = np.arange(nx, dtype=float)
        else:
            xc = np.asarray(x, dtype=float).ravel()
            if len(xc) != nx:
                raise ValueError(f"heatmap: x must have {nx} values (got {len(xc)}).")
        if y is None:
            yc = np.arange(ny, dtype=float)
        else:
            yc = np.asarray(y, dtype=float).ravel()
            if len(yc) != ny:
                raise ValueError(f"heatmap: y must have {ny} values (got {len(yc)}).")

        xedges = self._edges_from_centers(xc)
        yedges = self._edges_from_centers(yc)

        # "upper" mimics imshow (row 0 at the top); flip rows so the first row
        # ends up against the top edge while edges stay in ascending order.
        if str(origin).lower() == "upper":
            grid = grid[::-1]

        low_spec, high_spec, low, high = self._resolve_gradient_specs(gradient)
        finite = grid[np.isfinite(grid)]
        if vmin is None:
            vmin = float(np.min(finite)) if finite.size else 0.0
        if vmax is None:
            vmax = float(np.max(finite)) if finite.size else 1.0
        span = (float(vmax) - float(vmin)) or 1.0

        colors = []
        for i in range(ny):
            row = []
            for j in range(nx):
                val = grid[i, j]
                if not np.isfinite(val):
                    row.append(None)            # empty/masked cell
                    continue
                t = (val - vmin) / span
                t = min(1.0, max(0.0, t))
                row.append([low[k] + (high[k] - low[k]) * t for k in range(3)])
            colors.append(row)

        self.elements.append({
            "type": "heatmap",
            "hx_edges": xedges.tolist(),
            "hy_edges": yedges.tolist(),
            "cell_colors": colors,
            "nx": int(nx),
            "ny": int(ny),
            "alpha": float(alpha),
            "animate": float(animate) if animate else 0.0,
            "reveal": str(reveal).lower(),
            "gap": float(gap),
            "origin": str(origin).lower(),
            "edge_color": edge_color,
            "edge_width": float(edge_width),
            "drop_shadow": drop_shadow,
            "label": label,
            # Edges feed limit / tick computation so the grid sits flush.
            "x": xedges.tolist(),
            "y": yedges.tolist(),
            # Colorbar hookup (mirrors scatter's value-gradient fields).
            "gradient_low": low_spec if colorbar else None,
            "gradient_high": high_spec if colorbar else None,
            "gradient_vmin": float(vmin) if colorbar else None,
            "gradient_vmax": float(vmax) if colorbar else None,
            "gradient_name": colorbar_label,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    # ------------------------------------------------------------------
    # Time-evolving plots (keyframed datasets)
    # ------------------------------------------------------------------
    @staticmethod
    def _frames_to_2d(data, n_frames, n_pts, name):
        """Normalize ``data`` to a list of ``n_frames`` rows, each ``n_pts`` long.

        Accepts a 1D array (shared across every frame) or a 2D
        ``(n_frames, n_pts)`` array (one row per frame).
        """
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            if len(arr) != n_pts:
                raise ValueError(
                    f"{name}: expected {n_pts} values (got {len(arr)})."
                )
            return [arr.tolist() for _ in range(n_frames)]
        if arr.ndim == 2:
            if arr.shape != (n_frames, n_pts):
                raise ValueError(
                    f"{name}: expected shape ({n_frames}, {n_pts}); got {arr.shape}."
                )
            return [row.tolist() for row in arr]
        raise ValueError(f"{name}: must be a 1D or 2D array-like.")

    def _resolve_frame_times(self, n_frames, frame_times, frame_duration, start, delay=0.0):
        """Return a list of absolute keyframe times (seconds), one per frame.

        ``delay`` is a universal additive offset (seconds) applied to every
        keyframe time, including when explicit ``frame_times`` are supplied, so
        the whole animation can be pushed back to start later.
        """
        if frame_times is not None:
            times = [float(t) for t in frame_times]
            if len(times) != n_frames:
                raise ValueError(
                    f"frame_times must have {n_frames} entries (got {len(times)})."
                )
            return [t + float(delay) for t in times]
        if frame_duration is UNSET:
            frame_duration = config.frame_duration
        return [float(start) + i * float(frame_duration) + float(delay) for i in range(n_frames)]

    def _gradient_endpoints(self, gradient):
        """Resolve the (low, high) gradient endpoints as RGB 0-1 lists."""
        if (
            isinstance(gradient, (tuple, list))
            and len(gradient) == 2
            and not _is_rgb_triplet(gradient)
        ):
            low, high = gradient[0], gradient[1]
        else:
            low, high = config.gradient_low, config.gradient_high
        return self._color_to_rgb01(low), self._color_to_rgb01(high)

    def plot_evolving(self, x, y_frames, frame_times=None, frame_duration=UNSET,
                      hold_keyframes=UNSET, start=0.0, color="blue", label=None,
                      linewidth=4, linestyle="solid", dash_size=UNSET,
                      drop_shadow=False, fade_in=0.5, delay=0.0, loop=False, **kwargs):
        """Add a line whose shape evolves over time across multiple datasets.

        The line is a single shape layer; each dataset becomes a keyframe on the
        shape's Path property, so After Effects morphs the curve from one dataset
        to the next.

        Parameters:
            x: Shared x values (1D, length N) used for every frame, or a 2D
                ``(n_frames, N)`` array if x also changes over time.
            y_frames: 2D ``(n_frames, N)`` array-like — one y dataset per time
                step. Every frame must have the same length N (the Path keyframes
                interpolate vertex-by-vertex, so vertex counts must match).
            frame_times: Optional explicit times (seconds) for each frame's
                keyframe. If omitted, times are ``start + i * frame_duration``.
            frame_duration: Seconds between consecutive frames when
                ``frame_times`` is not given. Defaults to
                ``aegraph_config.config.frame_duration``.
            hold_keyframes: If True, Path keyframes use HOLD interpolation (the
                curve snaps between datasets); if False, LINEAR (smooth morph, no
                bezier easing). Defaults to ``aegraph_config.config.hold_keyframes``.
            start: Time (seconds) of the first frame's keyframe.
            color, label, linewidth, linestyle, dash_size, drop_shadow: Same as
                ``plot``.
            fade_in: Seconds for an opacity fade-in at ``start`` (0 disables).
            delay: Universal offset (seconds) added to every keyframe time
                (and to ``frame_times`` when given), pushing the whole animation
                back so it starts later. Stacks with ``start``.
            loop: If True, apply AE ``loopOut("cycle")`` on the path so the
                morph repeats for the full composition duration. For a seamless
                cycle, make the first and last datasets match (and set
                ``aegraph_config.config.comp_duration`` longer than one cycle).
        """
        if hold_keyframes is UNSET:
            hold_keyframes = config.hold_keyframes

        y_frames = [np.asarray(f, dtype=float).ravel() for f in y_frames]
        n_frames = len(y_frames)
        if n_frames < 2:
            raise ValueError("plot_evolving needs at least 2 frames (datasets).")
        n_pts = len(y_frames[0])
        if any(len(f) != n_pts for f in y_frames):
            raise ValueError(
                "All y-frames must share the same length so Path keyframes have "
                "matching vertex counts."
            )

        y_rows = [f.tolist() for f in y_frames]
        x_rows = self._frames_to_2d(x, n_frames, n_pts, "x")
        times = self._resolve_frame_times(n_frames, frame_times, frame_duration, start, delay)
        if dash_size is UNSET:
            dash_size = config.dash_size

        flat_x, flat_y = [], []
        for xr, yr in zip(x_rows, y_rows):
            flat_x.extend(xr)
            flat_y.extend(yr)

        self.elements.append({
            "type": "line_evolving",
            "x_frames": x_rows,
            "y_frames": y_rows,
            "frame_times": times,
            "hold": bool(hold_keyframes),
            "color": color,
            "label": label,
            "linewidth": linewidth,
            "linestyle": linestyle,
            "dash_size": dash_size,
            "drop_shadow": drop_shadow,
            "fade_in": float(fade_in),
            "delay": float(delay),
            "loop": bool(loop),
            # Flattened across frames for limit / tick / legend computation.
            "x": flat_x,
            "y": flat_y,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def scatter_evolving(self, x_frames, y_frames, radius_frames=None,
                         color_frames=None, c_frames=None, gradient=None,
                         frame_times=None, frame_duration=UNSET,
                         hold_keyframes=UNSET, start=0.0, color="red", radius=8,
                         alpha=1.0, label=None, drop_shadow=False, fade_in=0.5,
                         outline=UNSET, outline_width=UNSET, outline_color=None,
                         delay=0.0, marker="circle", loop=False, **kwargs):
        """Add a scatter cloud whose points evolve over time.

        A fixed set of N points is tracked across frames; each point gets
        Position keyframes (movement), optional Ellipse-Size keyframes (changing
        radius), and optional Fill-Color keyframes (changing color).

        Parameters:
            x_frames, y_frames: 2D ``(n_frames, N)`` array-likes giving the
                position of each of the N points at each time step.
            radius_frames: Optional per-point radii over time — 2D
                ``(n_frames, N)`` or a shared 1D length-N array. Falls back to the
                scalar ``radius`` when omitted.
            color_frames: Optional explicit per-point colors over time — a 2D
                list ``(n_frames, N)`` of color names / RGB triplets.
            c_frames: Optional per-point scalar values over time — 2D
                ``(n_frames, N)`` — mapped onto a two-color gradient (theme object
                colors by default, or ``gradient=(low, high)``) to drive color.
            gradient: Optional ``(low, high)`` endpoint colors used with
                ``c_frames``.
            frame_times, frame_duration, hold_keyframes, start: As in
                ``plot_evolving``. ``hold_keyframes`` snaps vs. linearly morphs the
                position / size / color keyframes.
            color, radius, alpha, label, drop_shadow, fade_in: Static fallbacks /
                styling.
            outline: Whether to draw a darker marker outline. Defaults to
                ``aegraph_config.config.scatter_outline``. The outline color is
                derived from each marker's fill color and tracks animated colors.
            outline_width: Stroke width for marker outlines in pixels. Defaults
                to ``aegraph_config.config.scatter_outline_width``.
            delay: Universal offset (seconds) added to every keyframe time
                (and to ``frame_times`` when given), pushing the whole animation
                back so it starts later. Stacks with ``start``.
            marker: Marker shape for every point (default ``"circle"``). One of
                ``"circle"``, ``"square"``, ``"diamond"``, ``"triangle"``,
                ``"pentagon"``, ``"hexagon"``, or ``"star"`` (matplotlib
                single-char aliases like ``"o"``, ``"s"``, ``"^"``, ``"*"`` also
                work). Non-circle markers keep the same ``radius`` meaning.
            loop: If True, apply AE ``loopOut("cycle")`` on position / size /
                color keyframes so the motion repeats for the full composition
                duration. Make the first and last frames match for a seamless
                cycle.
        """
        marker = _resolve_marker(marker)
        if hold_keyframes is UNSET:
            hold_keyframes = config.hold_keyframes
        if outline is UNSET:
            outline = config.scatter_outline
        if outline_width is UNSET:
            outline_width = config.scatter_outline_width

        xf = np.asarray(x_frames, dtype=float)
        yf = np.asarray(y_frames, dtype=float)
        if xf.ndim != 2 or yf.ndim != 2:
            raise ValueError("scatter_evolving: x_frames and y_frames must be 2D (n_frames, N).")
        if xf.shape != yf.shape:
            raise ValueError("scatter_evolving: x_frames and y_frames must have the same shape.")
        n_frames, n_pts = xf.shape
        if n_frames < 2:
            raise ValueError("scatter_evolving needs at least 2 frames.")

        x_rows = [row.tolist() for row in xf]
        y_rows = [row.tolist() for row in yf]
        times = self._resolve_frame_times(n_frames, frame_times, frame_duration, start, delay)

        # Radii over time.
        if radius_frames is None:
            radius_rows = [[float(radius)] * n_pts for _ in range(n_frames)]
            radius_varies = False
        else:
            radius_rows = self._frames_to_2d(radius_frames, n_frames, n_pts, "radius_frames")
            radius_varies = True

        # Apply global scatter radius multiplier.
        rs = float(config.scatter_radius_scale)
        if rs != 1.0:
            radius_rows = [
                [v * rs for v in row] for row in radius_rows
            ]

        # Colors over time -> per-frame per-point RGB 0-1 lists.
        color_rows = None
        if color_frames is not None:
            if len(color_frames) != n_frames or any(len(r) != n_pts for r in color_frames):
                raise ValueError("color_frames must have shape (n_frames, N).")
            color_rows = [[self._color_to_rgb01(c) for c in row] for row in color_frames]
        elif c_frames is not None:
            cf = np.asarray(c_frames, dtype=float)
            if cf.shape != (n_frames, n_pts):
                raise ValueError("c_frames must have shape (n_frames, N).")
            low, high = self._gradient_endpoints(gradient)
            vmin, vmax = float(np.min(cf)), float(np.max(cf))
            span = (vmax - vmin) or 1.0
            color_rows = []
            for row in cf:
                frame_colors = []
                for v in row:
                    t = (float(v) - vmin) / span
                    frame_colors.append([low[k] + (high[k] - low[k]) * t for k in range(3)])
                color_rows.append(frame_colors)

        flat_x, flat_y = [], []
        for xr, yr in zip(x_rows, y_rows):
            flat_x.extend(xr)
            flat_y.extend(yr)

        self.elements.append({
            "type": "scatter_evolving",
            "x_frames": x_rows,
            "y_frames": y_rows,
            "radius_frames": radius_rows,
            "radius_varies": radius_varies,
            "color_frames": color_rows,   # None or (n_frames, N) of RGB01
            "frame_times": times,
            "hold": bool(hold_keyframes),
            "color": color,
            "alpha": alpha,
            "label": label,
            "drop_shadow": drop_shadow,
            "fade_in": float(fade_in),
            "delay": float(delay),
            "n_points": n_pts,
            "outline": bool(outline),
            "outline_width": float(outline_width),
            "outline_color": outline_color,
            "marker": marker,
            "loop": bool(loop),
            # Flattened across frames for limit / tick / legend computation.
            "x": flat_x,
            "y": flat_y,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def quiver_evolving(self, x, y, u_frames, v_frames, scale=1.0,
                        scale_mode="data", width=3.0, headwidth=3.0,
                        headlength=11.0, color="blue", c_frames=None,
                        gradient=None, normalize=False, alpha=1.0,
                        frame_times=None, frame_duration=UNSET,
                        hold_keyframes=UNSET, start=0.0, pivot="tail",
                        label=None, drop_shadow=False, fade_in=0.5, delay=0.0, **kwargs):
        """Animated vector field: each arrow rotates, recolors, and resizes over time.

        Unlike :meth:`quiver` (a static snapshot), this keyframes every arrow so
        it tracks a *time-varying* field -- the arrow swings to follow the local
        flow direction, its fill color tracks the field's value (e.g. speed), and
        (unless ``normalize=True``) its length tracks the magnitude. Perfect for
        an evolving forecast, a rotating storm, or any field that breathes.

        The base points ``(x, y)`` are fixed (the grid doesn't move); only the
        vectors change. Each frame becomes a keyframe on the arrow's Path (and
        Fill Color), so After Effects morphs smoothly between time steps.

        Parameters:
            x, y: Arrow base coordinates (1D length N, or anything that flattens
                to N -- e.g. ``np.meshgrid`` output). Shared across all frames.
            u_frames, v_frames: Vector components per frame -- 2D
                ``(n_frames, N)`` array-likes (or 1D length N for a constant
                field). Must share the same N as ``x``/``y``.
            scale, scale_mode, width, headwidth, headlength, normalize, pivot:
                Same meaning as :meth:`quiver`. For geographic / stretched axes
                use ``scale_mode="comp"`` (length as a fraction of plot width).
            color: Fallback fill color when no ``c_frames`` is given.
            c_frames: What drives the per-frame color. ``True`` colors each arrow
                by its own speed ``|(u, v)|`` that frame; a 2D ``(n_frames, N)``
                array maps a custom field (e.g. vorticity) through ``gradient``;
                ``None`` keeps a single static ``color``. The color scale is
                fixed across all frames for a stable legend.
            gradient: ``(low, high)`` endpoint colors for the ``c_frames`` map.
            alpha: Arrow opacity (0-1).
            frame_times / frame_duration / hold_keyframes / start: Keyframe
                timing, exactly as in :meth:`scatter_evolving`.
            label, drop_shadow, fade_in: As elsewhere.
            delay: Universal offset (seconds) added to every keyframe time
                (and to ``frame_times`` when given), pushing the whole animation
                back so it starts later. Stacks with ``start``.

        Returns self for chaining.
        """
        if hold_keyframes is UNSET:
            hold_keyframes = config.hold_keyframes

        xa = np.asarray(x, dtype=float).ravel()
        ya = np.asarray(y, dtype=float).ravel()
        n_pts = len(xa)
        if len(ya) != n_pts:
            raise ValueError("quiver_evolving: x and y must flatten to the same length.")

        uf = np.asarray(u_frames, dtype=float)
        vf = np.asarray(v_frames, dtype=float)
        if uf.ndim == 1:
            uf = np.tile(uf.ravel(), (1, 1))
        if vf.ndim == 1:
            vf = np.tile(vf.ravel(), (1, 1))
        # Each frame's vectors must flatten to N; allow (n_frames, ...) inputs.
        uf = uf.reshape(uf.shape[0], -1)
        vf = vf.reshape(vf.shape[0], -1)
        if uf.shape != vf.shape or uf.shape[1] != n_pts:
            raise ValueError(
                "quiver_evolving: u_frames and v_frames must both have shape "
                f"(n_frames, {n_pts})."
            )
        n_frames = uf.shape[0]
        if n_frames < 2:
            raise ValueError("quiver_evolving needs at least 2 frames.")

        times = self._resolve_frame_times(n_frames, frame_times, frame_duration, start, delay)

        mag = np.hypot(uf, vf)                      # (n_frames, N)
        gmax = float(np.max(mag)) or 1.0
        # Relative magnitude (0..1) used by scale_mode="comp"; uniform if normalized.
        qrel = np.ones_like(mag) if normalize else (mag / gmax)

        # Per-frame per-arrow colors -> RGB 0-1, with a color scale fixed across
        # the whole animation so the mapping doesn't flicker frame to frame.
        color_rows = None
        if c_frames is not None:
            if c_frames is True:
                cf = mag
            else:
                cf = np.asarray(c_frames, dtype=float).reshape(n_frames, -1)
                if cf.shape != (n_frames, n_pts):
                    raise ValueError(
                        f"c_frames must have shape ({n_frames}, {n_pts})."
                    )
            low, high = self._gradient_endpoints(gradient)
            vmin, vmax = float(np.min(cf)), float(np.max(cf))
            span = (vmax - vmin) or 1.0
            color_rows = [[[low[k] + (high[k] - low[k]) * ((float(v) - vmin) / span)
                            for k in range(3)] for v in row] for row in cf]

        self.elements.append({
            "type": "quiver_evolving",
            "qx": xa.tolist(),
            "qy": ya.tolist(),
            "u_frames": uf.tolist(),
            "v_frames": vf.tolist(),
            "qrel": qrel.tolist(),
            "color_frames": color_rows,
            "frame_times": times,
            "hold": bool(hold_keyframes),
            "scale_mode": scale_mode,
            "qscale": float(scale),
            "normalize": bool(normalize),
            "color": color,
            "width": float(width),
            "headwidth": float(headwidth),
            "headlength": float(headlength),
            "alpha": alpha,
            "pivot": pivot,
            "label": label,
            "drop_shadow": drop_shadow,
            "fade_in": float(fade_in),
            "delay": float(delay),
            "n_points": n_pts,
            # Base points only — arrows are a fixed pixel length and shouldn't
            # stretch the data limits.
            "x": xa.tolist(),
            "y": ya.tolist(),
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def heatmap_evolving(self, data_frames, x=None, y=None, gradient=None,
                         vmin=None, vmax=None, frame_times=None,
                         frame_duration=UNSET, hold_keyframes=UNSET, start=0.0,
                         alpha=1.0, gap=0.0, origin="lower", label=None,
                         colorbar=True, colorbar_label=None, fade_in=0.5,
                         drop_shadow=False, delay=0.0, **kwargs):
        """A heatmap whose cell colors evolve over time (keyframed).

        Same grid of colored squares as :meth:`heatmap`, but you pass one 2D
        field *per time step* and AEGraph keyframes every cell's fill color so
        the whole field animates -- ideal for a 2D heat equation, a reaction-
        diffusion pattern, or any scalar field that changes over time.

        The grid geometry is fixed; only the colors change. The color scale
        (``vmin``/``vmax``) is held constant across all frames so the mapping
        doesn't flicker.

        Parameters:
            data_frames: 3D array-like, shape ``(n_frames, ny, nx)`` -- one 2D
                field per frame.
            x, y, gradient, alpha, gap, origin, colorbar, colorbar_label:
                Same meaning as :meth:`heatmap`.
            vmin, vmax: Global value range mapped to the gradient ends. Default
                to the finite min/max across *all* frames.
            frame_times / frame_duration / hold_keyframes / start: Keyframe
                timing, exactly as in :meth:`scatter_evolving`.
            label, fade_in, drop_shadow: As elsewhere.
            delay: Universal offset (seconds) added to every keyframe time
                (and to ``frame_times`` when given), pushing the whole animation
                back so it starts later. Stacks with ``start``.

        Returns self for chaining.
        """
        if hold_keyframes is UNSET:
            hold_keyframes = config.hold_keyframes

        frames = np.asarray(data_frames, dtype=float)
        if frames.ndim != 3:
            raise ValueError("heatmap_evolving: data_frames must be 3D (n_frames, ny, nx).")
        n_frames, ny, nx = frames.shape
        if n_frames < 2:
            raise ValueError("heatmap_evolving needs at least 2 frames.")

        if x is None:
            xc = np.arange(nx, dtype=float)
        else:
            xc = np.asarray(x, dtype=float).ravel()
            if len(xc) != nx:
                raise ValueError(f"heatmap_evolving: x must have {nx} values (got {len(xc)}).")
        if y is None:
            yc = np.arange(ny, dtype=float)
        else:
            yc = np.asarray(y, dtype=float).ravel()
            if len(yc) != ny:
                raise ValueError(f"heatmap_evolving: y must have {ny} values (got {len(yc)}).")

        xedges = self._edges_from_centers(xc)
        yedges = self._edges_from_centers(yc)
        if str(origin).lower() == "upper":
            frames = frames[:, ::-1, :]

        times = self._resolve_frame_times(n_frames, frame_times, frame_duration, start, delay)
        low_spec, high_spec, low, high = self._resolve_gradient_specs(gradient)

        finite = frames[np.isfinite(frames)]
        if vmin is None:
            vmin = float(np.min(finite)) if finite.size else 0.0
        if vmax is None:
            vmax = float(np.max(finite)) if finite.size else 1.0
        span = (float(vmax) - float(vmin)) or 1.0

        # A cell is rendered if it is finite in any frame. For frames where a
        # rendered cell is non-finite, fall back to the low color (t=0).
        ever_finite = np.isfinite(frames).any(axis=0)   # (ny, nx)
        color_frames = []
        for f in range(n_frames):
            rows = []
            for i in range(ny):
                row = []
                for j in range(nx):
                    if not ever_finite[i, j]:
                        row.append(None)
                        continue
                    val = frames[f, i, j]
                    t = 0.0 if not np.isfinite(val) else (val - vmin) / span
                    t = min(1.0, max(0.0, t))
                    row.append([low[k] + (high[k] - low[k]) * t for k in range(3)])
                rows.append(row)
            color_frames.append(rows)

        self.elements.append({
            "type": "heatmap_evolving",
            "hx_edges": xedges.tolist(),
            "hy_edges": yedges.tolist(),
            "color_frames": color_frames,
            "cell_mask": ever_finite.tolist(),
            "frame_times": times,
            "hold": bool(hold_keyframes),
            "nx": int(nx),
            "ny": int(ny),
            "alpha": float(alpha),
            "gap": float(gap),
            "fade_in": float(fade_in),
            "delay": float(delay),
            "drop_shadow": drop_shadow,
            "label": label,
            "x": xedges.tolist(),
            "y": yedges.tolist(),
            "gradient_low": low_spec if colorbar else None,
            "gradient_high": high_spec if colorbar else None,
            "gradient_vmin": float(vmin) if colorbar else None,
            "gradient_vmax": float(vmax) if colorbar else None,
            "gradient_name": colorbar_label,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def histogram(self, data, bins=10, color=UNSET, label=None, delay = 0.0, alpha=0.8, animate=1.0, drop_shadow=False, bar_anim_times=UNSET, density=False, ease_speed=None, ease_influence=None, meta_easy_ease=None, meta_ease_speed=None, meta_ease_influence=None, c=None, gradient=None, discrete=None, **kwargs):
        """
        Add a histogram to the graph.
        data: 1D array-like data to bin.
        bins: Number of bins or bin edges.
        color: Bar color. If not given (and no ``c``/``gradient`` is supplied),
            bins are colored with a positional gradient between
            ``config.gradient_low`` and ``config.gradient_high`` (the theme's
            ``object_color_2`` / ``object_color_1`` pair).
        label: Legend label.
        alpha: Bar opacity (0-1).
        animate: Total animation duration in seconds (bars animate sequentially).
        drop_shadow: Whether to add drop shadow effect.
        bar_anim_times: Per-bar animation duration (seconds). Scalar or list.
            Defaults to ``aegraph_config.config.bar_anim_times`` (0.5s) so each
            bar overlaps within ``animate``. Pass ``None`` explicitly to fall
            back to legacy sequential timing where each bar gets
            ``animate / n_bars`` seconds.
        density: If True, normalize heights so the area under the histogram is 1 (PDF style); if False, heights are counts.
        ease_speed: Optional per-element easy ease speed override. Uses AEGraph default when None.
        ease_influence: Optional per-element easy ease influence override. Uses AEGraph default when None.
        meta_easy_ease: Optional per-element override for easing the staggered
            bar entrances. None uses the AEGraph default.
        meta_ease_speed: Optional per-element meta easy ease speed override.
            Like AE easy ease, ``0`` gives the fullest cushion; higher values
            flatten the sweep toward a constant rate. None uses the AEGraph default.
        meta_ease_influence: Optional per-element meta easy ease influence
            override (higher = more pronounced slow-fast-slow sweep). None uses
            the AEGraph default.
        c: Optional per-bin numeric data mapped to a color gradient (length
            must equal the number of bins). Works exactly like scatter's ``c=``
            and triggers the colorbar.
        gradient: Either a 2-tuple ``(low, high)`` of colors that override the
            gradient endpoints, or a data array (acts like ``c=``).
        """
        if bar_anim_times is UNSET:
            bar_anim_times = config.bar_anim_times
        x_name = getattr(data, "name", None) if pd is not None and isinstance(data, pd.Series) else None
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
        grad = self._resolve_bar_gradient(color, c, gradient, n_bars=len(bin_left), discrete=discrete)
        self.elements.append({
            "type": "histogram",
            "bin_left": list(bin_left),
            "bin_right": list(bin_right),
            "bin_centers": list(bin_centers),
            "heights": list(heights),
            "color": grad["color"],
            "label": label,
            "alpha": alpha,
            "animate": animate,
            "drop_shadow": drop_shadow,
            "bar_anim_times": bar_anim_times,
            "density": density,
            "delay": delay,
            "ease_speed": ease_speed,
            "ease_influence": ease_influence,
            "meta_easy_ease": meta_easy_ease,
            "meta_ease_speed": meta_ease_speed,
            "meta_ease_influence": meta_ease_influence,
            "x_name": x_name,
            "y_name": "Density" if density else "Count",
            "gradient_colors": grad["gradient_colors"],
            "gradient_data": grad["gradient_data"],
            "gradient_low": grad["gradient_low"],
            "gradient_high": grad["gradient_high"],
            "gradient_vmin": grad["gradient_vmin"],
            "gradient_vmax": grad["gradient_vmax"],
            "gradient_name": grad["gradient_name"],
            "gradient_discrete": grad["gradient_discrete"],
            "gradient_levels": grad["gradient_levels"],
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def bar_graph(self, x_values, heights, bar_width=None, color=UNSET, label=None, alpha=0.8, animate=1.0, drop_shadow=False, bar_anim_times=UNSET, ease_speed=None, ease_influence=None, meta_easy_ease=None, meta_ease_speed=None, meta_ease_influence=None, c=None, gradient=None, discrete=None, bottom=None, colorbar=True, x_pad=None, **kwargs):
        """
        Add a bar graph to the plot using x values and corresponding heights.
        x_values: 1D array-like x positions for bars.
        heights: 1D array-like heights for bars (must match length of x_values).
        bar_width: Width of each bar in data coordinates. If None, auto-calculated from spacing.
        color: Bar color. If not given (and no ``c``/``gradient`` is supplied),
            bars are colored with a positional gradient between
            ``config.gradient_low`` and ``config.gradient_high`` (the theme's
            ``object_color_2`` / ``object_color_1`` pair).
        label: Legend label.
        alpha: Bar opacity (0-1).
        animate: Total animation duration in seconds (bars animate sequentially).
        drop_shadow: Whether to add drop shadow effect.
        bar_anim_times: Per-bar animation duration (seconds). Scalar or list.
            Defaults to ``aegraph_config.config.bar_anim_times`` (0.5s). Pass
            ``None`` explicitly for legacy sequential timing.
        ease_speed: Optional per-element easy ease speed override. Uses AEGraph default when None.
        ease_influence: Optional per-element easy ease influence override. Uses AEGraph default when None.
        meta_easy_ease: Optional per-element override for easing the staggered
            bar entrances. None uses the AEGraph default.
        meta_ease_speed: Optional per-element meta easy ease speed override.
            Like AE easy ease, ``0`` gives the fullest cushion; higher values
            flatten the sweep toward a constant rate. None uses the AEGraph default.
        meta_ease_influence: Optional per-element meta easy ease influence
            override (higher = more pronounced slow-fast-slow sweep). None uses
            the AEGraph default.
        c: Optional per-bar numeric data mapped to a color gradient (length
            must equal the number of bars). Triggers the colorbar.
        gradient: Either a 2-tuple ``(low, high)`` of colors overriding the
            gradient endpoints, or a data array (acts like ``c=``).
        colorbar: Whether to show the gradient colorbar (default ``True``).
            Set to ``False`` to keep the gradient bar colours while hiding the
            colorbar legend on the right side of the chart.
        x_pad: Extra space in **data units** to add between the outermost bar
            edges and the auto-computed x-axis limits on each side (for
            vertical bars).  Defaults to ``None``, which uses the existing
            inter-bar gap so the edge spacing mirrors the gap between adjacent
            bars.  Pass ``0`` for bars flush against the axis edge, or a
            positive value for extra breathing room.  Has no effect when
            ``set_xlim()`` is also called (the explicit limit takes priority).
        """
        if bar_anim_times is UNSET:
            bar_anim_times = config.bar_anim_times
        x_name = getattr(x_values, "name", None) if pd is not None and isinstance(x_values, pd.Series) else None
        y_name = getattr(heights, "name", None) if pd is not None and isinstance(heights, pd.Series) else None
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

        # Optional per-bar baseline (``bottom``) so bars can stack. Defaults to
        # zero (bars grow from the x-axis).
        if bottom is None:
            baseline = np.zeros(len(heights))
        else:
            baseline = np.asarray(bottom, dtype=float)
            if baseline.ndim == 0:
                baseline = np.full(len(heights), float(baseline))

        grad = self._resolve_bar_gradient(color, c, gradient, n_bars=len(bin_left), discrete=discrete)
        self.elements.append({
            "type": "bar_graph",
            "bin_left": list(bin_left),
            "bin_right": list(bin_right),
            "bin_centers": list(bin_centers),
            "heights": list(heights),
            "baseline": list(baseline),
            "color": grad["color"],
            "label": label,
            "alpha": alpha,
            "animate": animate,
            "drop_shadow": drop_shadow,
            "bar_anim_times": bar_anim_times,
            "ease_speed": ease_speed,
            "ease_influence": ease_influence,
            "meta_easy_ease": meta_easy_ease,
            "meta_ease_speed": meta_ease_speed,
            "meta_ease_influence": meta_ease_influence,
            "x_name": x_name,
            "y_name": y_name,
            "gradient_colors": grad["gradient_colors"],
            "gradient_data": grad["gradient_data"],
            "gradient_low": grad["gradient_low"] if colorbar else None,
            "gradient_high": grad["gradient_high"] if colorbar else None,
            "gradient_vmin": grad["gradient_vmin"] if colorbar else None,
            "gradient_vmax": grad["gradient_vmax"] if colorbar else None,
            "gradient_name": grad["gradient_name"],
            "gradient_discrete": grad["gradient_discrete"],
            "gradient_levels": grad["gradient_levels"],
            "x_pad": x_pad,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def lollipop(self, categories, values, orientation="horizontal", color="red",
                 radius=12, marker="circle", alpha=1.0, label=None,
                 stem=True, baseline=0.0, stem_thickness=0.06, stem_color=None,
                 stem_alpha=0.55, value_labels=False, value_fmt="{:.2f}",
                 value_fontsize=26, value_gap=None, sort=None, set_ticks=True,
                 animate=UNSET, stem_animate=None, stem_duration=None, dot_delay=0.0,
                 dot_follows_stem=False, dot_reveal_duration=0.3,
                 outline=UNSET, outline_width=UNSET, outline_alpha=1.0,
                 outline_color=None, **kwargs):
        """
        Add a lollipop / Cleveland dot chart (one dot per category).

        Each category gets a single dot positioned at its value, optionally
        connected to a baseline by a thin stem. This is the cleanest way to
        compare a ranked list of one-number-per-category quantities (e.g. a
        country-level indicator) without the heavy ink of a bar chart.

        For the *statistical* dot plot, where individual observations stack on
        top of each other when they share a value/bin, see ``dot_plot``.

        Implemented by composing the existing primitives: stems are drawn with
        ``barh``/``bar_graph`` and dots with ``scatter``, so all the usual
        theming, animation, and marker options apply.

        Args:
            categories: Category labels (one per value). Used as tick labels.
            values: Numeric value for each category (same length as categories).
            orientation: ``"horizontal"`` (default) puts categories on the
                y-axis and values on the x-axis; ``"vertical"`` swaps them.
            color: Dot color (name or RGB list).
            radius: Dot radius in pixels.
            marker: Dot marker shape (see ``scatter``).
            alpha: Dot fill opacity.
            label: Legend label for the dots.
            stem: Whether to draw a stem from ``baseline`` to each dot.
            baseline: Value the stems grow from (default 0).
            stem_thickness: Stem thickness, as a fraction of the row spacing.
            stem_color: Stem color; defaults to the dot ``color``.
            stem_alpha: Stem opacity.
            value_labels: Whether to print each value next to its dot.
            value_fmt: Format string for value labels (default ``"{:.2f}"``).
            value_fontsize: Font size for value labels.
            value_gap: Data-space offset of value labels from the dot. If
                ``None``, a small fraction of the value span is used.
            sort: ``None`` (keep order), ``"ascending"``, or ``"descending"``
                (reorder categories by value).
            set_ticks: Whether to set the category axis ticks/labels.
            animate: Total dot animation duration (seconds).
            stem_animate: Total time window for the stem sweep (seconds).
                Defaults to ``animate``. When ``stem_duration`` is set, stems
                overlap within this window (like ``barh(..., animate=5,
                bar_duration=1)``); otherwise each stem gets
                ``stem_animate / n`` seconds sequentially.
            stem_duration: Per-stem grow duration in seconds. When set, stems
                (and ``dot_follows_stem`` dots) overlap: each row takes this
                long but start times are staggered across ``stem_animate``.
            dot_delay: Delay before the dots animate in (seconds). When
                ``dot_follows_stem=True``, this instead delays the whole
                stem+dot pair together (applied to both).
            dot_follows_stem: When ``True``, each dot doesn't pop in at its
                final spot -- it starts at ``baseline`` and slides out in
                lockstep with its own stem, arriving exactly as the stem
                finishes growing (so the marker visibly rides the growing
                tip). Ignores ``animate`` in favor of ``stem_animate`` so the
                dot and stem share identical per-row timing.
            dot_reveal_duration: Seconds for each dot's scale-in reveal when
                ``dot_follows_stem=True`` (default ``0.3``).
            outline, outline_width, outline_alpha, outline_color: Forwarded to
                ``scatter`` for the dot outlines.

        Returns:
            self: For method chaining.
        """
        if orientation not in ("horizontal", "vertical"):
            raise ValueError("orientation must be 'horizontal' or 'vertical'")
        if animate is UNSET:
            animate = config.scatter_animate
        if stem_animate is None:
            stem_animate = animate

        if pd is not None:
            if isinstance(categories, pd.Series):
                categories = categories.values
            if isinstance(values, pd.Series):
                values = values.values

        categories = list(categories)
        values = [float(v) for v in values]
        if len(categories) != len(values):
            raise ValueError("categories and values must have the same length")

        if sort in ("ascending", "descending"):
            order = sorted(range(len(values)), key=lambda i: values[i],
                           reverse=(sort == "descending"))
            categories = [categories[i] for i in order]
            values = [values[i] for i in order]
        elif sort is not None:
            raise ValueError("sort must be None, 'ascending', or 'descending'")

        n = len(values)
        positions = list(range(n))
        labels = [str(c) for c in categories]
        if stem_color is None:
            stem_color = color

        if value_gap is None:
            span = (max(values) - min(values)) if n > 1 else (max(values) or 1.0)
            value_gap = 0.02 * span if span else 0.02

        # When the dot rides its stem's growing tip, both need the exact same
        # per-row start/duration schedule. `barh`/`bar_graph` compute that
        # schedule lazily at render time as
        #   individual = <bar_anim_times>; start_i = linspace(0, max(0, stem_animate
        #   - individual), n) then eased via self._apply_meta_ease(...).
        # We recreate it here (forcing an explicit `bar_anim_times=individual`
        # on the stem call so it can't silently drift from this), then hand
        # the identical numbers to `scatter` via `point_start_times` /
        # `point_anim_times` (which skip their own scheduling when given).
        follow_kwargs = {}
        stem_kwargs = {}
        dot_animate = animate
        dot_delay_for_stem = dot_delay
        if dot_follows_stem and stem and n > 0:
            individual = (
                float(stem_duration) if stem_duration is not None
                else float(stem_animate) / n
            )
            stem_starts, individual = self._bar_entrance_schedule(
                n, stem_animate, individual,
            )
            follow_kwargs = {
                "point_start_times": stem_starts,
                "point_anim_times": individual,
            }
            stem_kwargs = {"bar_anim_times": individual}
            dot_animate = stem_animate
            dot_delay_for_stem = dot_delay

        if orientation == "horizontal":
            if stem:
                self.barh(positions, [v - baseline for v in values],
                          bar_height=stem_thickness, color=stem_color,
                          alpha=stem_alpha, animate=stem_animate, left=baseline,
                          delay=dot_delay_for_stem if dot_follows_stem else 0.0,
                          **stem_kwargs)
            if dot_follows_stem and stem:
                follow_kwargs["move_from_x"] = [
                    self._visible_axis_origin(baseline, "x")
                ] * n
            self.scatter(values, positions, color=color, radius=radius,
                         marker=marker, alpha=alpha, label=label,
                         animate=dot_animate, delay=dot_delay_for_stem,
                         point_reveal_duration=dot_reveal_duration,
                         outline=outline, outline_width=outline_width,
                         outline_alpha=outline_alpha, outline_color=outline_color,
                         **follow_kwargs, **kwargs)
            if value_labels:
                for v, p in zip(values, positions):
                    self.annotate(value_fmt.format(v), v + value_gap, p,
                                  fontsize=value_fontsize, alignment="left",
                                  vertical_alignment="center", delay=dot_delay)
            if set_ticks:
                self.set_yticks(positions, labels)
        else:
            if stem:
                self.bar_graph(positions, [v - baseline for v in values],
                               bar_width=stem_thickness, color=stem_color,
                               alpha=stem_alpha, animate=stem_animate,
                               bottom=baseline, **stem_kwargs)
            if dot_follows_stem and stem:
                follow_kwargs["move_from_y"] = [
                    self._visible_axis_origin(baseline, "y")
                ] * n
            self.scatter(positions, values, color=color, radius=radius,
                         marker=marker, alpha=alpha, label=label,
                         animate=dot_animate, delay=dot_delay_for_stem,
                         point_reveal_duration=dot_reveal_duration,
                         outline=outline, outline_width=outline_width,
                         outline_alpha=outline_alpha, outline_color=outline_color,
                         **follow_kwargs, **kwargs)
            if value_labels:
                for v, p in zip(values, positions):
                    self.annotate(value_fmt.format(v), p, v + value_gap,
                                  fontsize=value_fontsize, alignment="center",
                                  vertical_alignment="bottom", delay=dot_delay)
            if set_ticks:
                self.set_xticks(positions, labels)

        return self

    def dot_plot(self, values, bin_width=None, bins=None, orientation="vertical",
                 color="red", radius=10, marker="circle", alpha=1.0, label=None,
                 baseline=0.0, dot_gap=1.0, stack_origin=None, animate=UNSET,
                 set_count_ticks=True, auto_ylim=True,
                 outline=UNSET, outline_width=UNSET,
                 outline_alpha=1.0, outline_color=None, **kwargs):
        """
        Add a statistical (Wilkinson) dot plot: one dot per observation, with
        dots stacking on top of one another whenever they fall in the same bin.

        This is the "dots pile up when they share a number" chart -- a discrete,
        dot-based cousin of the histogram. Each value is snapped to the centre
        of its bin; dots in the same bin are stacked vertically (or horizontally
        if ``orientation="horizontal"``), so the height of each column is the
        count of observations at that value.

        For a ranked, one-dot-per-category comparison with stems, see
        ``lollipop``.

        Args:
            values: 1D array of individual observations.
            bin_width: Width of each bin in data units. If omitted it is derived
                from ``bins``, or auto-chosen (1.0 for integer-only data,
                otherwise ~1/30 of the data range).
            bins: Number of bins spanning the data range (ignored if
                ``bin_width`` is given).
            orientation: ``"vertical"`` (default) stacks dots upward with the
                values on the x-axis; ``"horizontal"`` stacks rightward with the
                values on the y-axis.
            color: Dot color (name, RGB list, or per-dot list).
            radius: Dot radius in pixels. Tune so stacked dots just touch.
            marker: Dot marker shape (see ``scatter``).
            alpha: Dot fill opacity.
            label: Legend label.
            baseline: Count-axis position of the first dot in each stack.
            dot_gap: Count-axis spacing between stacked dots (default 1.0,
                giving integer-count positions). To make the vertical pixel
                distance between dots equal the horizontal pixel distance,
                scale this by ``(x_range / y_range) * (plot_h_px / plot_w_px)``
                relative to ``bin_width`` — see script-level comments for the
                formula.
            stack_origin: Reference value the bin grid is centred on. Defaults
                to the minimum observation, so bins are centred on the data.
            animate: Total animation duration (seconds).
            set_count_ticks: Whether to label the count axis with integers.
            auto_ylim: When ``True`` (default), automatically sets the count
                axis limits so that one ``dot_gap`` unit spans the same number
                of pixels as one ``bin_width`` unit on the value axis — making
                the vertical gap between stacked dots equal to the horizontal
                gap between adjacent columns. Uses the composition dimensions
                and empirical plot-area fractions (≈72 % wide, ≈55 % tall)
                to compute the conversion. Set to ``False`` if you want to
                control ``set_ylim`` / ``set_xlim`` manually.
            outline, outline_width, outline_alpha, outline_color: Forwarded to
                ``scatter``.

        Returns:
            self: For method chaining.
        """
        if orientation not in ("vertical", "horizontal"):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")
        if animate is UNSET:
            animate = config.scatter_animate

        if pd is not None and isinstance(values, pd.Series):
            values = values.values
        vals = np.asarray([float(v) for v in values], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return self

        vmin, vmax = float(vals.min()), float(vals.max())
        if bin_width is None:
            if bins is not None and bins > 0:
                bin_width = (vmax - vmin) / bins if vmax > vmin else 1.0
            elif np.allclose(vals, np.round(vals)):
                bin_width = 1.0
            else:
                bin_width = (vmax - vmin) / 30.0 if vmax > vmin else 1.0
        if bin_width <= 0:
            bin_width = 1.0

        origin = vmin if stack_origin is None else float(stack_origin)

        # Snap each observation to its bin index, then stack within the bin.
        bin_idx = np.round((vals - origin) / bin_width).astype(int)
        order = np.argsort(bin_idx, kind="stable")
        bin_idx = bin_idx[order]

        levels = np.zeros(bin_idx.size, dtype=int)
        counts = {}
        for i, b in enumerate(bin_idx):
            levels[i] = counts.get(b, 0)
            counts[b] = levels[i] + 1

        centers = origin + bin_idx * bin_width
        stack = baseline + (levels + 0.5) * dot_gap
        max_count = int(max(counts.values()))

        if orientation == "vertical":
            xs, ys = centers, stack
        else:
            xs, ys = stack, centers

        self.scatter(xs, ys, color=color, radius=radius, marker=marker,
                     alpha=alpha, label=label, animate=animate, outline=outline,
                     outline_width=outline_width, outline_alpha=outline_alpha,
                     outline_color=outline_color, **kwargs)

        if set_count_ticks:
            tick_positions = [baseline + (k + 0.5) * dot_gap for k in range(max_count)]
            tick_labels = [str(k + 1) for k in range(max_count)]
            if orientation == "vertical":
                self.set_yticks(tick_positions, tick_labels)
            else:
                self.set_xticks(tick_positions, tick_labels)

        if auto_ylim:
            # Compute count-axis limits so that one dot_gap equals one bin_width
            # in pixel space — making vertical and horizontal spacings visually equal.
            #
            # Derivation: for equal pixel gaps,
            #   (plot_w_px / x_range) * bin_width  ==  (plot_h_px / ylim_range) * dot_gap
            #   => ylim_range = dot_gap * plot_h_px * x_range / (bin_width * plot_w_px)
            #
            # plot_w_px ≈ comp_width  * PW_FRAC  (plot area excl. margins/axes)
            # plot_h_px ≈ comp_height * PH_FRAC
            # x_range   ≈ data span * X_MARGIN_FACTOR  (accounts for AEGraph
            #             adding ~11 % padding on each side of the data range)
            PW_FRAC = 0.72
            PH_FRAC = 0.55
            X_MARGIN_FACTOR = 1.22          # xlim is typically ~22 % wider than data range
            x_data_span = float(vmax - vmin) if vmax > vmin else bin_width
            effective_x_range = x_data_span * X_MARGIN_FACTOR
            cw = float(self.comp_width  or 1920)
            ch = float(self.comp_height or 1080)
            ylim_range = dot_gap * ch * PH_FRAC * effective_x_range / (bin_width * cw * PW_FRAC)
            count_axis_lo = baseline
            count_axis_hi = baseline + ylim_range
            if orientation == "vertical":
                self.set_ylim(count_axis_lo, count_axis_hi)
            else:
                self.set_xlim(count_axis_lo, count_axis_hi)

        return self

    @staticmethod
    def _normalize_box_groups(data):
        """Coerce ``data`` into (list_of_1d_arrays, default_labels).

        Accepts a single 1D sequence (one box), a list of sequences (several
        boxes), a 2D numpy array (each column is a box, matplotlib-style), a
        pandas Series (one box) or DataFrame (each column is a box).
        """
        default_labels = None

        if pd is not None and isinstance(data, pd.DataFrame):
            default_labels = [str(c) for c in data.columns]
            groups = [data[c].to_numpy(dtype=float) for c in data.columns]
            return groups, default_labels
        if pd is not None and isinstance(data, pd.Series):
            return [data.to_numpy(dtype=float)], None

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return [data.astype(float)], None
            if data.ndim == 2:
                return [data[:, j].astype(float) for j in range(data.shape[1])], None
            raise ValueError("boxplot data array must be 1D or 2D")

        if isinstance(data, (list, tuple)):
            if len(data) == 0:
                raise ValueError("boxplot received empty data")
            first = data[0]
            is_scalar_first = np.isscalar(first) or (
                isinstance(first, (int, float, np.integer, np.floating))
            )
            if is_scalar_first:
                return [np.asarray(data, dtype=float)], None
            groups = []
            for g in data:
                if pd is not None and isinstance(g, pd.Series):
                    g = g.to_numpy()
                groups.append(np.asarray(g, dtype=float))
            return groups, None

        # Fallback: treat as a single iterable of numbers.
        return [np.asarray(list(data), dtype=float)], None

    def boxplot(self, data, positions=None, labels=None, orientation="vertical",
                color="blue", colors=None, box_width=0.6, alpha=0.45,
                whisker=1.5, show_outliers=True, outlier_radius=8,
                outlier_marker="circle", outlier_alpha=0.9,
                median_color=None, edge_color=None, whisker_color=None,
                cap_color=None, linewidth=4, edge_linewidth=None,
                median_linewidth=None, whisker_linewidth=None, cap_frac=0.55,
                show_edges=True, show_caps=True, legend_labels=None,
                set_ticks=True, animate=UNSET, box_animate=None,
                whisker_animate=None, group_delay=None, delay=0.0,
                outline=UNSET, outline_width=UNSET, outline_color=None, **kwargs):
        """
        Add an animated box-and-whisker plot (one box per data group).

        Each group is summarised by its quartiles: the box spans Q1-Q3 with a
        median line, whiskers reach to the most extreme observation within
        ``whisker`` * IQR of the box, and points beyond that are drawn as
        outliers. The plot is composed from the existing ``bar_graph``/``barh``
        (box), ``plot`` (whiskers, caps, median, edges) and ``scatter``
        (outliers) primitives, so all the usual theming carries over.

        Animation: each box first grows from Q1 toward Q3, then its whiskers,
        caps, median and edges draw in, and finally any outliers pop in. With
        multiple groups the boxes enter in sequence (see ``group_delay``).

        Args:
            data: One dataset (1D sequence / Series) for a single box, or
                several datasets for several boxes. A list of sequences, a 2D
                numpy array (columns = boxes) or a DataFrame (columns = boxes)
                all produce one box per column/sequence.
            positions: Numeric position of each box along the category axis.
                Defaults to ``0, 1, 2, ...``.
            labels: Tick labels for the boxes. Defaults to DataFrame column
                names when available, otherwise the positions.
            orientation: ``"vertical"`` (boxes stand up, values on the y-axis)
                or ``"horizontal"`` (boxes lie down, values on the x-axis).
            color: Default box color when ``colors`` is not given.
            colors: Per-box colors (name / hex / RGB). Cycled if shorter than
                the number of boxes. This is the main knob for color-coding
                boxes (e.g. one color per continent).
            box_width: Box thickness along the category axis, in data units.
            alpha: Box fill opacity.
            whisker: Whisker length as a multiple of the IQR (default 1.5, the
                Tukey convention).
            show_outliers: Whether to draw points beyond the whiskers.
            outlier_radius: Outlier marker radius (px).
            outlier_marker: Outlier marker shape (see ``scatter``).
            outlier_alpha: Outlier marker opacity.
            median_color: Median line color. Defaults to the edge color (same
                color as the box outline).
            edge_color: Box outline color. Defaults to the box color.
            whisker_color: Whisker line color. Defaults to the box color.
            cap_color: Whisker cap color. Defaults to the whisker color.
            linewidth: Base stroke width for whiskers (and the default for
                edges/median/caps when their specific widths are unset).
            edge_linewidth: Box outline width. Defaults to ``linewidth``.
            median_linewidth: Median line width. Defaults to ``linewidth*1.5``.
            whisker_linewidth: Whisker line width. Defaults to ``linewidth``.
            cap_frac: Cap length as a fraction of ``box_width``.
            show_edges: Whether to draw the box outline.
            show_caps: Whether to draw whisker end caps.
            legend_labels: Optional per-box legend labels (attached to the box
                fill). Use to build a color legend (e.g. continents).
            set_ticks: Whether to set the category-axis ticks/labels.
            animate: Total per-box animation budget (seconds). Defaults to
                ``aegraph_config.config.scatter_animate``.
            box_animate: Box grow-in duration. Defaults to ``animate``. The box
                fill grows concurrently with the outline/whisker draw.
            whisker_animate: Whisker/median/edge draw duration. Defaults to
                ``animate``.
            group_delay: Extra delay added per successive box, so boxes enter
                one after another. Defaults to ``box_animate*0.6``.
            delay: Global delay (seconds) before the first box animates.
            outline, outline_width, outline_color: Forwarded to ``scatter`` for
                the outlier markers.

        Returns:
            self: For method chaining.
        """
        if orientation not in ("vertical", "horizontal"):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")
        if animate is UNSET:
            animate = config.scatter_animate
        # Box fill and outline draw concurrently, so give them matching
        # durations by default (they finish together).
        if box_animate is None:
            box_animate = animate
        if whisker_animate is None:
            whisker_animate = animate
        if group_delay is None:
            group_delay = box_animate * 0.6
        if edge_linewidth is None:
            edge_linewidth = linewidth
        if median_linewidth is None:
            median_linewidth = linewidth * 1.5
        if whisker_linewidth is None:
            whisker_linewidth = linewidth

        groups, default_labels = self._normalize_box_groups(data)
        n = len(groups)

        if positions is None:
            positions = list(range(n))
        positions = [float(p) for p in positions]
        if len(positions) != n:
            raise ValueError("positions must have one entry per box")

        if labels is None:
            labels = default_labels
        if labels is not None:
            labels = [str(l) for l in labels]

        def _box_color(k):
            if colors is not None and len(colors) > 0:
                return colors[k % len(colors)]
            return color

        half_w = box_width / 2.0
        cap_w = box_width * cap_frac / 2.0
        # Each box draws on top of the next; later boxes overlap when packed.
        for k, raw in enumerate(groups):
            arr = np.asarray(raw, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue

            q1, med, q3 = (float(v) for v in np.percentile(arr, [25, 50, 75]))
            iqr = q3 - q1
            lo_fence = q1 - whisker * iqr
            hi_fence = q3 + whisker * iqr
            within = arr[(arr >= lo_fence) & (arr <= hi_fence)]
            w_low = float(within.min()) if within.size else q1
            w_high = float(within.max()) if within.size else q3
            outliers = arr[(arr < lo_fence) | (arr > hi_fence)]

            p = positions[k]
            col = _box_color(k)
            edge_col = edge_color if edge_color is not None else col
            med_col = median_color if median_color is not None else edge_col
            whisk_col = whisker_color if whisker_color is not None else col
            cap_col = cap_color if cap_color is not None else whisk_col
            box_label = legend_labels[k] if legend_labels is not None and k < len(legend_labels) else None

            # Box fill and the surrounding lines start together so the shaded
            # box grows in *while* its outline/whiskers draw, rather than before.
            d_box = delay + k * group_delay
            d_post = d_box
            d_out = d_box + max(box_animate, whisker_animate) * 0.85

            def _seg(xa, ya, xb, yb, ccol, lw):
                self.plot([xa, xb], [ya, yb], color=ccol, linewidth=lw,
                          animate=whisker_animate, delay=d_post)

            if orientation == "vertical":
                # Box grows from Q1 up to Q3.
                self.bar_graph([p], [q3 - q1], bar_width=box_width, color=col,
                               alpha=alpha, bottom=[q1], animate=box_animate,
                               bar_anim_times=box_animate, delay=d_box,
                               label=box_label)
                # Whiskers (vertical) and caps (horizontal).
                _seg(p, w_low, p, q1, whisk_col, whisker_linewidth)
                _seg(p, q3, p, w_high, whisk_col, whisker_linewidth)
                if show_caps:
                    _seg(p - cap_w, w_low, p + cap_w, w_low, cap_col, whisker_linewidth)
                    _seg(p - cap_w, w_high, p + cap_w, w_high, cap_col, whisker_linewidth)
                if show_edges:
                    _seg(p - half_w, q1, p - half_w, q3, edge_col, edge_linewidth)
                    _seg(p + half_w, q1, p + half_w, q3, edge_col, edge_linewidth)
                    _seg(p - half_w, q1, p + half_w, q1, edge_col, edge_linewidth)
                    _seg(p - half_w, q3, p + half_w, q3, edge_col, edge_linewidth)
                # Median on top of the fill.
                _seg(p - half_w, med, p + half_w, med, med_col, median_linewidth)
                if show_outliers and outliers.size:
                    self.scatter([p] * outliers.size, list(outliers), color=col,
                                 radius=outlier_radius, marker=outlier_marker,
                                 alpha=outlier_alpha, animate=whisker_animate,
                                 delay=d_out,
                                 outline=outline, outline_width=outline_width,
                                 outline_color=outline_color, **kwargs)
            else:
                # Horizontal: box grows from Q1 rightward to Q3.
                self.barh([p], [q3 - q1], bar_height=box_width, color=col,
                          alpha=alpha, left=q1, animate=box_animate,
                          bar_anim_times=box_animate, delay=d_box,
                          label=box_label)
                _seg(w_low, p, q1, p, whisk_col, whisker_linewidth)
                _seg(q3, p, w_high, p, whisk_col, whisker_linewidth)
                if show_caps:
                    _seg(w_low, p - cap_w, w_low, p + cap_w, cap_col, whisker_linewidth)
                    _seg(w_high, p - cap_w, w_high, p + cap_w, cap_col, whisker_linewidth)
                if show_edges:
                    _seg(q1, p - half_w, q3, p - half_w, edge_col, edge_linewidth)
                    _seg(q1, p + half_w, q3, p + half_w, edge_col, edge_linewidth)
                    _seg(q1, p - half_w, q1, p + half_w, edge_col, edge_linewidth)
                    _seg(q3, p - half_w, q3, p + half_w, edge_col, edge_linewidth)
                _seg(med, p - half_w, med, p + half_w, med_col, median_linewidth)
                if show_outliers and outliers.size:
                    self.scatter(list(outliers), [p] * outliers.size, color=col,
                                 radius=outlier_radius, marker=outlier_marker,
                                 alpha=outlier_alpha, animate=whisker_animate,
                                 delay=d_out,
                                 outline=outline, outline_width=outline_width,
                                 outline_color=outline_color, **kwargs)

        if set_ticks:
            tick_labels = labels if labels is not None else [str(p) for p in positions]
            if orientation == "vertical":
                self.set_xticks(positions, tick_labels)
            else:
                self.set_yticks(positions, tick_labels)

        return self

    def barh(self, y_values, widths, bar_height=None, color=UNSET, label=None, delay=0.0, alpha=0.8, animate=1.0, drop_shadow=False, bar_duration=None, bar_anim_times=UNSET, animate_downward: bool = False, anchor_at_y_axis: bool = False, ease_speed=None, ease_influence=None, meta_easy_ease=None, meta_ease_speed=None, meta_ease_influence=None, c=None, gradient=None, discrete=None, left=None, **kwargs):
        """
        Add a horizontal bar graph to the plot (similar to matplotlib's barh).

        y_values: 1D array-like y positions for bars (vertical positions).
        widths: 1D array-like widths for bars (horizontal lengths, must match length of y_values).
        bar_height: Height (thickness) of each bar in data coordinates. If None, auto-calculated from spacing.
        color: Bar color. If not given (and no ``c``/``gradient`` is supplied),
            bars are colored with a positional gradient between
            ``config.gradient_low`` and ``config.gradient_high`` (the theme's
            ``object_color_2`` / ``object_color_1`` pair).
        label: Legend label.
        alpha: Bar opacity (0-1).
        animate: Total animation duration in seconds (all bars complete within this time).
        drop_shadow: Whether to add drop shadow effect.
        bar_duration: Duration for each individual bar animation in seconds. If None, bars animate
                      sequentially without overlap (each gets animate/n_bars seconds). If specified,
                      bars will overlap - e.g., animate=5.0 with bar_duration=1.0 creates smooth
                      overlapping animations where each bar takes 1 second but spreads over 5 seconds total.
        bar_anim_times: (Deprecated, use bar_duration) Per-bar animation
            duration. Defaults to ``aegraph_config.config.bar_anim_times``
            (0.5s). Pass ``None`` explicitly for legacy sequential timing.
        animate_downward (bool): If True, bars animate from top y to bottom y
                     (downward). Default animates bottom-to-top.
        ease_speed: Optional per-element easy ease speed override. Uses AEGraph default when None.
        ease_influence: Optional per-element easy ease influence override. Uses AEGraph default when None.
        meta_easy_ease: Optional per-element override for easing the staggered
            bar entrances. None uses the AEGraph default.
        meta_ease_speed: Optional per-element meta easy ease speed override.
            Like AE easy ease, ``0`` gives the fullest cushion; higher values
            flatten the sweep toward a constant rate. None uses the AEGraph default.
        meta_ease_influence: Optional per-element meta easy ease influence
            override (higher = more pronounced slow-fast-slow sweep). None uses
            the AEGraph default.
        c: Optional per-bar numeric data mapped to a color gradient (length
            must equal the number of bars). Triggers the colorbar.
        gradient: Either a 2-tuple ``(low, high)`` of colors overriding the
            gradient endpoints, or a data array (acts like ``c=``).

        Example:
            # Smooth overlapping animations
            plot = AEGraph().barh(y_positions, values, animate=5.0, bar_duration=1.0)

            # Sequential animations (no overlap)
            plot = AEGraph().barh(y_positions, values, animate=5.0)
        """
        if bar_anim_times is UNSET:
            bar_anim_times = config.bar_anim_times
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

        # Optional per-bar baseline (``left``) so horizontal bars can stack.
        if left is None:
            baseline = np.zeros(len(widths))
        else:
            baseline = np.asarray(left, dtype=float)
            if baseline.ndim == 0:
                baseline = np.full(len(widths), float(baseline))

        grad = self._resolve_bar_gradient(color, c, gradient, n_bars=len(bin_bottom), discrete=discrete)
        self.elements.append({
            "type": "barh",
            "bin_bottom": list(bin_bottom),
            "bin_top": list(bin_top),
            "bin_centers": list(bin_centers),
            "widths": list(widths),
            "baseline": list(baseline),
            "color": grad["color"],
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
            "meta_easy_ease": meta_easy_ease,
            "meta_ease_speed": meta_ease_speed,
            "meta_ease_influence": meta_ease_influence,
            "gradient_colors": grad["gradient_colors"],
            "gradient_data": grad["gradient_data"],
            "gradient_low": grad["gradient_low"],
            "gradient_high": grad["gradient_high"],
            "gradient_vmin": grad["gradient_vmin"],
            "gradient_vmax": grad["gradient_vmax"],
            "gradient_name": grad["gradient_name"],
            "gradient_discrete": grad["gradient_discrete"],
            "gradient_levels": grad["gradient_levels"],
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def _resolve_series_colors(self, colors, count):
        """Return ``count`` series colors, one per stack/group layer.

        When ``colors`` is omitted, interpolate between the active theme's
        ``object_color_2`` / ``object_color_1`` (``gradient_low`` ->
        ``gradient_high``), matching the default used by ``pie()``.
        """
        if colors is not None:
            return [colors[k % len(colors)] for k in range(count)]
        low, high = self._gradient_endpoints(None)
        if count <= 1:
            return [list(low)]
        return [
            [low[c] + (high[c] - low[c]) * (k / (count - 1)) for c in range(3)]
            for k in range(count)
        ]

    def bar_stacked(self, x_values, series, colors=None, labels=None,
                    bar_width=None, animate=UNSET, alpha=0.9, delay=0.0,
                    drop_shadow=False, **kwargs):
        """Stacked vertical bars.

        Renders as a single element whose layers reveal bottom-to-top within each
        column using the same smoothstep cumulative schedule as ``pie()``.

        Args:
            x_values: Shared category x positions.
            series: List of height arrays, one per stack layer (drawn bottom-up).
            colors: Optional list of colors, one per layer. Defaults to a ramp
                between the theme's ``object_color_2`` and ``object_color_1``.
            labels: Optional list of legend labels, one per layer.
        """
        series = [np.asarray(s, dtype=float) for s in series]
        x_values = np.asarray(x_values, dtype=float)
        n = len(x_values)
        if not series:
            return self
        for s in series:
            if len(s) != n:
                raise ValueError("Each series must match x_values length")

        if bar_width is None:
            if n > 1:
                spacings = np.diff(np.sort(x_values))
                min_spacing = np.min(spacings[spacings > 0]) if len(spacings) > 0 and np.any(spacings > 0) else 1.0
                bar_width = 0.8 * min_spacing
            else:
                bar_width = 1.0

        half_width = bar_width / 2
        bin_left = x_values - half_width
        bin_right = x_values + half_width
        bin_centers = x_values.copy()
        totals = np.sum(series, axis=0)

        if animate is UNSET:
            animate = getattr(config, "line_animate", 2.0)

        layer_colors = self._resolve_series_colors(colors, len(series))

        self.elements.append({
            "type": "bar_stacked",
            "series": [list(s) for s in series],
            "colors": layer_colors,
            "labels": list(labels) if labels else None,
            "bin_left": list(bin_left),
            "bin_right": list(bin_right),
            "bin_centers": list(bin_centers),
            "heights": list(totals),
            "baseline": [0.0] * n,
            "alpha": alpha,
            "animate": animate,
            "delay": float(delay),
            "drop_shadow": drop_shadow,
            **kwargs
        })
        if labels:
            for lbl in labels:
                self.legend.append(lbl)
        return self

    def bar_grouped(self, x_values, series, colors=None, labels=None,
                    group_width=0.8, animate=1.0, alpha=0.9, **kwargs):
        """Grouped (side-by-side) vertical bars.

        Args:
            x_values: Shared category x positions.
            series: List of height arrays, one per group member.
            colors: Optional list of colors, one per series. Defaults to a ramp
                between the theme's ``object_color_2`` and ``object_color_1``.
            group_width: Fraction of the category spacing the whole group spans.
        """
        x = np.asarray(x_values, dtype=float)
        n_groups = len(series)
        if n_groups == 0:
            return self
        spacing = float(np.min(np.diff(np.sort(x)))) if len(x) > 1 else 1.0
        total = group_width * spacing
        bw = total / n_groups
        resolved = self._resolve_series_colors(colors, n_groups)
        for k, heights in enumerate(series):
            offset = -total / 2 + bw * (k + 0.5)
            self.bar_graph(x + offset, np.asarray(heights, dtype=float),
                           bar_width=bw, color=resolved[k],
                           label=labels[k] if labels else None,
                           alpha=alpha, animate=animate, **kwargs)
        return self

    def barh_stacked(self, y_values, series, colors=None, labels=None,
                     bar_height=None, animate=1.0, alpha=0.9, **kwargs):
        """Stacked horizontal bars (segments grow left-to-right per layer)."""
        series = [np.asarray(s, dtype=float) for s in series]
        n = len(np.asarray(y_values))
        resolved = self._resolve_series_colors(colors, len(series))
        baseline = np.zeros(n)
        for k, widths in enumerate(series):
            self.barh(y_values, widths, bar_height=bar_height,
                      color=resolved[k],
                      label=labels[k] if labels else None,
                      alpha=alpha, animate=animate, left=list(baseline), **kwargs)
            baseline = baseline + widths
        return self

    def barh_grouped(self, y_values, series, colors=None, labels=None,
                     group_width=0.8, animate=1.0, alpha=0.9, **kwargs):
        """Grouped (stacked vertically within each category) horizontal bars."""
        y = np.asarray(y_values, dtype=float)
        n_groups = len(series)
        if n_groups == 0:
            return self
        spacing = float(np.min(np.diff(np.sort(y)))) if len(y) > 1 else 1.0
        total = group_width * spacing
        bh = total / n_groups
        resolved = self._resolve_series_colors(colors, n_groups)
        for k, widths in enumerate(series):
            offset = -total / 2 + bh * (k + 0.5)
            self.barh(y + offset, np.asarray(widths, dtype=float),
                      bar_height=bh, color=resolved[k],
                      label=labels[k] if labels else None,
                      alpha=alpha, animate=animate, **kwargs)
        return self

    def barh_evolving(self, y_values, width_frames, frame_times=None,
                      frame_duration=UNSET, hold_keyframes=UNSET, bar_height=None,
                      color=UNSET, c=None, gradient=None, c_frames=None,
                      color_vmin=None, color_vmax=None, discrete=None,
                      label=None, alpha=0.9, animate=1.0,
                      bar_duration=0.5, animate_downward=True,
                      anchor_at_y_axis=True, drop_shadow=False, ease_speed=None,
                      ease_influence=None, delay=0.0, loop=False, **kwargs):
        """Horizontal bars whose individual widths morph across datasets.

        Like ``barh``, every bar is its own shape layer that animates in with a
        staggered grow-in. In addition, each bar's width is keyframed across the
        ``width_frames`` datasets, so after settling on the first dataset the
        bars morph from one frame to the next (e.g. a population pyramid
        evolving over several census years).

        Parameters:
            y_values: 1D bar y positions (length N).
            width_frames: list of F arrays, each length N — one width-per-bar
                dataset per time step. Bar signs (left/right) must stay constant
                across frames.
            frame_times: Optional absolute settle times (seconds), one per frame.
                ``frame_times[0]`` is when the initial grow-in has fully settled
                and should be >= ``animate``. Defaults to
                ``animate + k * frame_duration``.
            frame_duration: Seconds between consecutive frames when
                ``frame_times`` is not given. Defaults to
                ``aegraph_config.config.frame_duration``.
            hold_keyframes: If True, the year-to-year morph snaps (HOLD) instead
                of gliding. The initial grow-in always eases. Defaults to
                ``aegraph_config.config.hold_keyframes``.
            bar_height, label, alpha, animate, bar_duration,
            animate_downward, anchor_at_y_axis, drop_shadow, ease_speed,
            ease_influence: As in ``barh``.
            color: Solid bar color when no gradient is configured.
            c: Per-bar numeric data mapped to a static gradient (length N).
            gradient: Endpoint colors ``(low, high)`` or a data array (like ``c=``).
            c_frames: Per-frame per-bar scalars, shape ``(F, N)``, mapped through
                ``gradient`` each frame so fill color can track a changing value
                (e.g. emissions in a bar-chart race). Pass ``color_vmin`` /
                ``color_vmax`` to lock the scale across several calls.
            color_vmin / color_vmax: Fixed color scale for ``c_frames`` (defaults
                to the min/max of the supplied frames).
            delay: Universal offset (seconds) added to the grow-in and every
                frame settle time (including ``frame_times`` when given),
                pushing the whole animation back so it starts later.
            loop: If True, skip the entrance grow-in and apply AE
                ``loopOut("cycle")`` on each bar's Scale (and color, if
                animated) so the morph repeats for the full composition
                duration. Make the first and last width frames match for a
                seamless cycle.
        """
        if hold_keyframes is UNSET:
            hold_keyframes = config.hold_keyframes

        y_values = np.asarray(y_values, dtype=float)
        frames = [np.asarray(f, dtype=float).ravel() for f in width_frames]
        n_frames = len(frames)
        if n_frames < 1:
            raise ValueError("barh_evolving needs at least one width frame.")
        n_bars = len(y_values)
        if any(len(f) != n_bars for f in frames):
            raise ValueError("Every width frame must match len(y_values).")

        grad = self._resolve_bar_gradient(color, c, gradient, n_bars, discrete=discrete)
        color_frames = None
        if c_frames is not None:
            cf = np.asarray(c_frames, dtype=float).reshape(n_frames, n_bars)
            low, high = grad["gradient_low"], grad["gradient_high"]
            if low is None or high is None:
                endpoints = self._gradient_endpoints(gradient)
                low, high = endpoints[0], endpoints[1]
            vmin = float(color_vmin) if color_vmin is not None else float(np.nanmin(cf))
            vmax = float(color_vmax) if color_vmax is not None else float(np.nanmax(cf))
            color_frames = [
                self._gradient_colors(cf[f], low, high, vmin=vmin, vmax=vmax)
                for f in range(n_frames)
            ]
            grad["gradient_vmin"] = vmin
            grad["gradient_vmax"] = vmax
            grad["gradient_low"] = low
            grad["gradient_high"] = high

        if bar_height is None:
            if n_bars > 1:
                spacings = np.diff(np.sort(y_values))
                pos = spacings[spacings > 0]
                bar_height = 0.8 * (np.min(pos) if len(pos) else 1.0)
            else:
                bar_height = 1.0
        half_height = bar_height / 2
        bin_bottom = y_values - half_height
        bin_top = y_values + half_height

        # Per-bar largest-magnitude (signed) width across frames, used both as
        # the rectangle's full extent (scale 100%) and for limit computation.
        ref_widths = []
        for j in range(n_bars):
            col = [f[j] for f in frames]
            ref_widths.append(max(col, key=abs) if col else 0.0)

        if frame_duration is UNSET:
            frame_duration = config.frame_duration
        if frame_times is None:
            frame_times = [float(animate) + k * float(frame_duration)
                           for k in range(n_frames)]
        else:
            frame_times = [float(t) for t in frame_times]
            if len(frame_times) != n_frames:
                raise ValueError(
                    f"frame_times must have {n_frames} entries (got {len(frame_times)})."
                )
        # ``delay`` pushes the whole animation back. The grow-in start times are
        # relative (the renderer adds ``elem['delay']``), but frame settle times
        # are absolute, so shift them here to keep grow-in and morph aligned.
        frame_times = [t + float(delay) for t in frame_times]

        self.elements.append({
            "type": "barh_evolving",
            "bin_bottom": list(bin_bottom),
            "bin_top": list(bin_top),
            "bin_centers": list(y_values),
            "widths": list(ref_widths),
            "width_frames": [f.tolist() for f in frames],
            "frame_times": frame_times,
            "hold": bool(hold_keyframes),
            "color": grad["color"],
            "color_frames": color_frames,
            "gradient_colors": grad["gradient_colors"],
            "gradient_data": grad["gradient_data"],
            "gradient_low": grad["gradient_low"],
            "gradient_high": grad["gradient_high"],
            "gradient_vmin": grad["gradient_vmin"],
            "gradient_vmax": grad["gradient_vmax"],
            "gradient_name": grad["gradient_name"],
            "gradient_discrete": grad["gradient_discrete"],
            "gradient_levels": grad["gradient_levels"],
            "label": label,
            "alpha": alpha,
            "animate": animate,
            "bar_anim_times": bar_duration,
            "animate_downward": animate_downward,
            "anchor_at_y_axis": anchor_at_y_axis,
            "drop_shadow": drop_shadow,
            "delay": float(delay),
            "ease_speed": ease_speed,
            "ease_influence": ease_influence,
            "loop": bool(loop),
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def bar_evolving(self, x_values, height_frames, frame_times=None,
                     frame_duration=UNSET, hold_keyframes=UNSET, bar_width=None,
                     color=UNSET, label=None, alpha=0.9, animate=1.0,
                     bar_duration=0.5, anchor_at_x_axis=True, drop_shadow=False,
                     ease_speed=None, ease_influence=None, delay=0.0,
                     meta_easy_ease=None, meta_ease_speed=None,
                     meta_ease_influence=None, **kwargs):
        """Vertical bars whose individual heights morph across datasets.

        The vertical analogue of ``barh_evolving``: every bar is its own shape
        layer that grows up from the x-axis with a staggered grow-in, and each
        bar's height is keyframed across the ``height_frames`` datasets so the
        bars morph from one frame to the next (e.g. a bar-chart race over
        census years).

        Parameters (mirror ``barh_evolving`` with x/y swapped):
            x_values: 1D bar x positions (length N).
            height_frames: list of F arrays, each length N -- one
                height-per-bar dataset per time step. Bar signs (up/down) must
                stay constant across frames.
            frame_times: Optional absolute settle times (seconds), one per
                frame. ``frame_times[0]`` is when the grow-in has settled and
                should be >= ``animate``. Defaults to
                ``animate + k * frame_duration``.
            frame_duration: Seconds between consecutive frames when
                ``frame_times`` is not given. Defaults to
                ``aegraph_config.config.frame_duration``.
            hold_keyframes: If True the year-to-year morph snaps (HOLD) instead
                of gliding, so no inaccurate values appear between datasets. The
                initial grow-in always eases. Defaults to
                ``aegraph_config.config.hold_keyframes``.
            bar_width: Width of each bar in data units (defaults to 0.8 * the
                minimum spacing between ``x_values``).
            color, label, alpha, animate, bar_duration, anchor_at_x_axis,
                drop_shadow, ease_speed, ease_influence, delay: as in ``barh``.
        """
        if hold_keyframes is UNSET:
            hold_keyframes = config.hold_keyframes
        if color is UNSET:
            color = getattr(config, "default_bar_color", "blue")

        x_values = np.asarray(x_values, dtype=float)
        frames = [np.asarray(f, dtype=float).ravel() for f in height_frames]
        n_frames = len(frames)
        if n_frames < 1:
            raise ValueError("bar_evolving needs at least one height frame.")
        n_bars = len(x_values)
        if any(len(f) != n_bars for f in frames):
            raise ValueError("Every height frame must match len(x_values).")

        if bar_width is None:
            if n_bars > 1:
                spacings = np.diff(np.sort(x_values))
                pos = spacings[spacings > 0]
                bar_width = 0.8 * (np.min(pos) if len(pos) else 1.0)
            else:
                bar_width = 1.0
        half_width = bar_width / 2
        bin_left = x_values - half_width
        bin_right = x_values + half_width

        # Per-bar largest-magnitude (signed) height across frames, used as the
        # rectangle's full extent (scale 100%) and for limit computation.
        ref_heights = []
        for j in range(n_bars):
            col = [f[j] for f in frames]
            ref_heights.append(max(col, key=abs) if col else 0.0)

        if frame_duration is UNSET:
            frame_duration = config.frame_duration
        if frame_times is None:
            frame_times = [float(animate) + k * float(frame_duration)
                           for k in range(n_frames)]
        else:
            frame_times = [float(t) for t in frame_times]
            if len(frame_times) != n_frames:
                raise ValueError(
                    f"frame_times must have {n_frames} entries (got {len(frame_times)})."
                )
        frame_times = [t + float(delay) for t in frame_times]

        self.elements.append({
            "type": "bar_evolving",
            "bin_left": list(bin_left),
            "bin_right": list(bin_right),
            "bin_centers": list(x_values),
            "heights": list(ref_heights),
            "height_frames": [f.tolist() for f in frames],
            "frame_times": frame_times,
            "hold": bool(hold_keyframes),
            "color": color,
            "label": label,
            "alpha": alpha,
            "animate": animate,
            "bar_anim_times": bar_duration,
            "anchor_at_x_axis": anchor_at_x_axis,
            "drop_shadow": drop_shadow,
            "delay": float(delay),
            "ease_speed": ease_speed,
            "ease_influence": ease_influence,
            "meta_easy_ease": meta_easy_ease,
            "meta_ease_speed": meta_ease_speed,
            "meta_ease_influence": meta_ease_influence,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def bar_stacked_evolving(self, x_values, series_frames, colors=None,
                             labels=None, frame_times=None, frame_duration=UNSET,
                             hold_keyframes=UNSET, bar_width=None, animate=1.0,
                             bar_duration=0.5, alpha=1.0, drop_shadow=False,
                             delay=0.0, **kwargs):
        """Stacked vertical bars whose segments morph across datasets.

        A time-evolving stacked bar (a "stacked bar-chart race"). Each layer in
        ``series_frames`` is one stack segment, drawn bottom-to-top. Within each
        x position the segments are rendered as cumulative *overlapping* bars
        (tallest behind, shortest in front) via ``bar_evolving``, so every bar
        grows cleanly from the x-axis and each colored segment reads as a
        stacked band. Because the bars overlap, fills are forced opaque
        (``alpha`` defaults to 1.0).

        Parameters:
            x_values: Shared category x positions (length N).
            series_frames: List of L layers (bottom -> top). Each layer is a
                list of F frames, and each frame is a length-N array of heights,
                i.e. ``series_frames[layer][frame][bar]``. Every layer must have
                the same F and N.
            colors: Optional list of colors, one per layer. Defaults to a ramp
                between the theme's ``object_color_2`` and ``object_color_1``.
            labels: Optional list of legend labels, one per layer.
            frame_times, frame_duration, hold_keyframes, bar_width, animate,
                bar_duration, delay: as in ``bar_evolving``. Set
                ``hold_keyframes=True`` to snap between datasets (no interpolated
                in-between values).
        """
        n_layers = len(series_frames)
        if n_layers == 0:
            return self
        n_frames = len(series_frames[0])
        if any(len(layer) != n_frames for layer in series_frames):
            raise ValueError("Every layer must have the same number of frames.")

        # Cumulative top of the stack through layer k, per frame.
        cumulative = []
        running = None
        for k in range(n_layers):
            layer = [np.asarray(fr, dtype=float).ravel() for fr in series_frames[k]]
            if running is None:
                running = [fr.copy() for fr in layer]
            else:
                running = [running[f] + layer[f] for f in range(n_frames)]
            cumulative.append([fr.copy() for fr in running])

        resolved = self._resolve_series_colors(colors, n_layers)

        # Draw tallest cumulative bar (top of stack) first so the shorter
        # cumulative bars sit on top and each segment's color shows through.
        for k in range(n_layers - 1, -1, -1):
            self.bar_evolving(
                x_values, cumulative[k],
                frame_times=frame_times, frame_duration=frame_duration,
                hold_keyframes=hold_keyframes, bar_width=bar_width,
                color=resolved[k], label=(labels[k] if labels else None),
                alpha=alpha, animate=animate, bar_duration=bar_duration,
                drop_shadow=drop_shadow, delay=delay, **kwargs
            )
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
                               label_male: Optional[str] = None,
                               label_female: Optional[str] = None,
                               drop_shadow: Optional[bool] = None,
                               delay: float = 0.0,
                               ):
        """
        Add a mirrored horizontal bar chart (population pyramid) to the current graph.

        Provide either `csv_path` to a file with three columns (age, male, female)
        or pass `ages`, `male`, and `female` arrays directly.

        mode: 'percent' to compute per-age-group share of total population (%),
              'counts' to use raw counts (auto-scaled if very large).

        delay: Seconds to wait before the bar entrance animation starts
               (forwarded to the underlying ``barh`` calls for both sides).

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
            delay=delay,
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
            delay=delay,
        )

        # Axes and labels
        self.set_xlabel(xlabel)
        self.set_yticks(positions=y_positions, labels=ages)
        self.set_xlim(xmin, xmax)
        # Ensure percent tick labels by default for pyramids in percent mode
        if mode == "percent":
            self.percent_tick_labels = True
        # Format x-ticks as absolute percent labels for population pyramids
        positions = self._nice_ticks_for_axis(xmin, xmax, nticks=7, scale=self.xscale)
        step = positions[1] - positions[0] if len(positions) > 1 else 1.0
        if abs(step) < 1:
            labels = [f"{abs(p):.1f}%" for p in positions]
        else:
            labels = [f"{abs(p):.0f}%" for p in positions]
        self.set_xticks(positions=positions, labels=labels)
        if show_grid:
            self.grid(show=True, color="lightgray", alpha=0.25)

        return self

    def add_population_pyramid_evolving(self,
                                        ages: List[str],
                                        male_frames: List,
                                        female_frames: List,
                                        mode: str = "percent",
                                        frame_times=None,
                                        frame_duration=UNSET,
                                        hold_keyframes=UNSET,
                                        animate: float = 1.0,
                                        bar_duration: float = 0.5,
                                        animate_downward: bool = True,
                                        show_grid: bool = True,
                                        color_male: Union[str, List[float], Tuple[float, float, float]] = [73, 118, 222],
                                        color_female: Union[str, List[float], Tuple[float, float, float]] = [168, 61, 104],
                                        label_male: Optional[str] = None,
                                        label_female: Optional[str] = None,
                                        drop_shadow: Optional[bool] = None,
                                        ):
        """Population pyramid whose bars morph across multiple datasets (years).

        This is the evolving counterpart to ``add_population_pyramid``: each age
        band is a real bar (left = male, right = female) that animates in once
        and then keyframes its width across ``male_frames`` / ``female_frames``,
        so the whole pyramid morphs from one year to the next.

        Parameters:
            ages: Shared age-band labels (length N).
            male_frames, female_frames: lists of F arrays, each length N — one
                raw count dataset per time step.
            mode: 'percent' (per-frame share of that frame's total population) or
                'counts' (auto-scaled raw counts, shared scale across frames).
            frame_times, frame_duration, hold_keyframes, animate, bar_duration,
            animate_downward: As in ``barh_evolving``.
            color_male, color_female, label_male, label_female, drop_shadow,
            show_grid: Styling, matching ``add_population_pyramid``.
        """
        male_frames = [np.asarray(f, dtype=float).ravel() for f in male_frames]
        female_frames = [np.asarray(f, dtype=float).ravel() for f in female_frames]
        n_frames = len(male_frames)
        if n_frames < 1 or len(female_frames) != n_frames:
            raise ValueError("male_frames and female_frames must be non-empty and equal length.")
        n = len(ages)
        if any(len(f) != n for f in (*male_frames, *female_frames)):
            raise ValueError("Every frame must match len(ages).")

        male_w_frames, female_w_frames = [], []
        if mode == "counts":
            all_max = max(
                float(np.max(f)) for f in (*male_frames, *female_frames)
            )
            scale = 1_000.0 if all_max > 1_000_000 else 1.0
            for mf, ff in zip(male_frames, female_frames):
                male_w_frames.append(-mf / scale)
                female_w_frames.append(ff / scale)
            xlabel = "Population (scaled)"
        else:
            for mf, ff in zip(male_frames, female_frames):
                total = float(mf.sum() + ff.sum())
                if total <= 0:
                    raise ValueError("A frame has zero total population; cannot compute percentages.")
                male_w_frames.append(-(mf / total) * 100.0)
                female_w_frames.append((ff / total) * 100.0)
            xlabel = "Population (%)"

        y_positions = np.arange(n)
        bar_shadow = self.drop_shadow if drop_shadow is None else bool(drop_shadow)

        # Shared, symmetric x-range across every frame.
        max_abs = max(
            float(np.max(np.abs(f))) for f in (*male_w_frames, *female_w_frames)
        )
        xpad = max_abs * 0.1
        xmin, xmax = -(max_abs + xpad), (max_abs + xpad)

        self.barh_evolving(
            y_positions, male_w_frames,
            frame_times=frame_times, frame_duration=frame_duration,
            hold_keyframes=hold_keyframes, animate=animate,
            bar_duration=bar_duration, alpha=0.9, color=color_male,
            label=label_male, animate_downward=animate_downward,
            anchor_at_y_axis=True, drop_shadow=bar_shadow,
        )
        self.barh_evolving(
            y_positions, female_w_frames,
            frame_times=frame_times, frame_duration=frame_duration,
            hold_keyframes=hold_keyframes, animate=animate,
            bar_duration=bar_duration, alpha=0.9, color=color_female,
            label=label_female, animate_downward=animate_downward,
            anchor_at_y_axis=True, drop_shadow=bar_shadow,
        )

        self.set_xlabel(xlabel)
        self.set_yticks(positions=y_positions, labels=ages)
        self.set_xlim(xmin, xmax)
        if mode == "percent":
            self.percent_tick_labels = True
        positions = self._nice_ticks_for_axis(xmin, xmax, nticks=7, scale=self.xscale)
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

    def annotate(self, text: str, x: float, y: float, fontsize: int = 12, font: str = None, alignment: str = "left", vertical_alignment: str = "center", delay: float = 0.0):
        """
        Add text annotation at specific coordinates on the graph.

        Args:
            text (str): The annotation text to display
            x (float): X coordinate in data space
            y (float): Y coordinate in data space
            fontsize (int): Font size (default: 12)
            font (str): After Effects PostScript font name. Defaults to the
                active theme's body font when omitted.
            alignment (str): Horizontal alignment relative to (x, y) -
                "left", "center", or "right" (default: "left").
            vertical_alignment (str): Vertical alignment relative to (x, y) -
                "top", "center"/"middle", or "bottom" (default: "center").
                "top" places the text below the point (its top edge at y),
                "bottom" places it above the point (its bottom edge at y), and
                "center" centers it on the point.

        Returns:
            self: For method chaining
        """
        # Validate alignment parameter
        valid_alignments = ["left", "center", "right"]
        if alignment.lower() not in valid_alignments:
            raise ValueError(f"alignment must be one of {valid_alignments}, got '{alignment}'")

        # Validate and normalize vertical alignment ("middle" is an alias for "center").
        valid_valignments = ["top", "center", "middle", "bottom"]
        va = vertical_alignment.lower()
        if va not in valid_valignments:
            raise ValueError(f"vertical_alignment must be one of {valid_valignments}, got '{vertical_alignment}'")
        if va == "middle":
            va = "center"

        # Store annotation in elements list for processing during JSX generation
        self.elements.append({
            "type": "annotation",
            "text": text,
            "x": x,
            "y": y,
            "fontsize": fontsize,
            "font": font,
            "alignment": alignment.lower(),
            "vertical_alignment": va,
            "delay": delay
        })
        return self

    def evolving_text(self, data, location=(0, 0), animate=None, before="",
                      after="", delimiter="", fontsize=48, font=None,
                      color=None, horizontal_alignment="left",
                      vertical_alignment="center",
                      frame_times=None, frame_duration=UNSET, start=0.0,
                      hold_keyframes=False, fade_in=0.5, label=None, delay=0.0,
                      **kwargs):
        """
        Add a text layer whose value is driven by data over time.

        A "Slider Control" effect is added to a text layer and keyframed with
        ``data``. The layer's Source Text is set to an expression that reads the
        slider, rounds it, and formats it with an optional prefix, suffix, and
        thousands delimiter. This is ideal for a counting "year" timer or a
        running value (e.g. ``"$1.234 Trillion"``) that stays in sync with an
        evolving plot.

        The generated expression mirrors::

            s = "<before>" + Math.round(effect(1)(1)[0]) + "<after>";
            s.replace(/\\B(?=(\\d{3})+(?!\\d))/g, "<delimiter>");

        Args:
            data: Sequence of numbers, one per keyframe. The slider is keyframed
                to these values; ``Math.round`` of the (interpolated) slider is
                what gets displayed.
            location (tuple): ``(x, y)`` in data coordinates. Only ``y`` is
                used for vertical placement; horizontal position comes from
                ``horizontal_alignment`` on the graph (default: ``(0, 0)``).
            animate (float, optional): Total animation duration in seconds over
                which the slider sweeps from the first to the last value. When
                given, keyframes are spaced evenly across this span. If omitted,
                ``frame_times`` / ``frame_duration`` are used instead.
            before (str): Prefix string placed before the number (default: "").
            after (str): Suffix string placed after the number (default: "").
            delimiter (str): Thousands delimiter inserted between digit groups,
                e.g. "," or "." (default: "" - no delimiter).
            fontsize (int): Font size (default: 48).
            font (str): Font name (default: "Helvetica").
            color: Text fill color (name or RGB). Defaults to the plot's
                ``ui_color`` when ``None``.
            horizontal_alignment (str): "left", "center", or "right" on the
                graph area (default: "left").
            vertical_alignment (str): "top", "center"/"middle", or "bottom"
                relative to the data ``y`` in ``location``.
            frame_times (list, optional): Explicit keyframe times (seconds), one
                per value. Overrides ``animate`` / ``frame_duration``.
            frame_duration (float, optional): Seconds between consecutive
                keyframes when ``animate`` and ``frame_times`` are not given.
            start (float): Time (seconds) of the first keyframe (default: 0.0).
            hold_keyframes (bool): If True, the slider steps between values
                (snaps to each data point). If False (default), it interpolates
                linearly so the number counts smoothly.
            fade_in (float): Fade-in duration in seconds (default: 0.5).
            label (str, optional): Optional label for bookkeeping.
            delay (float): Universal offset (seconds) added to every keyframe
                time (including ``frame_times`` and the ``animate`` span),
                pushing the whole animation back so it starts later. Stacks
                with ``start`` (default: 0.0).

        Returns:
            self: For method chaining.
        """
        arr = np.asarray(data, dtype=float).ravel()
        n = int(arr.size)
        if n < 1:
            raise ValueError("evolving_text: data must contain at least one value.")

        # Resolve absolute keyframe times.
        if frame_times is not None:
            times = [float(t) for t in frame_times]
            if len(times) != n:
                raise ValueError(
                    f"evolving_text: frame_times must have {n} entries (got {len(times)})."
                )
        elif animate is not None:
            total = float(animate)
            if n == 1:
                times = [float(start)]
            else:
                times = [float(start) + i * (total / (n - 1)) for i in range(n)]
        else:
            times = self._resolve_frame_times(n, None, frame_duration, start)

        # ``delay`` pushes the whole ticker back uniformly across all paths.
        if delay:
            times = [t + float(delay) for t in times]

        # Validate location.
        try:
            loc_x, loc_y = location
        except (TypeError, ValueError):
            raise ValueError("evolving_text: location must be an (x, y) pair in data coordinates.")

        # Validate horizontal alignment (comp-based).
        valid_halignments = ["left", "center", "right"]
        ha = horizontal_alignment.lower()
        if ha not in valid_halignments:
            raise ValueError(
                f"horizontal_alignment must be one of {valid_halignments}, got '{horizontal_alignment}'"
            )
        va = vertical_alignment.lower()
        valid_valignments = ["top", "center", "middle", "bottom"]
        if va not in valid_valignments:
            raise ValueError(f"vertical_alignment must be one of {valid_valignments}, got '{vertical_alignment}'")
        if va == "middle":
            va = "center"

        self.elements.append({
            "type": "evolving_text",
            "data": arr.tolist(),
            "frame_times": times,
            "loc_x": float(loc_x),
            "loc_y": float(loc_y),
            "before": str(before),
            "after": str(after),
            "delimiter": str(delimiter),
            "fontsize": fontsize,
            "font": font,
            "color": color,
            "horizontal_alignment": ha,
            "vertical_alignment": va,
            "hold": bool(hold_keyframes),
            "fade_in": float(fade_in),
            "delay": float(delay),
            "label": label,
        })
        return self

    def fill_between(self, x, y_low, y_high, color="blue", alpha=0.3, label=None,
                     edge=False, edge_color=None, edge_width=2, animate=UNSET,
                     delay=0.0, **kwargs):
        """Shade the area between two curves (confidence band / uncertainty).

        Args:
            x: Shared x values (1D).
            y_low: Lower boundary (scalar or 1D, same length as ``x``).
            y_high: Upper boundary (scalar or 1D, same length as ``x``).
            color: Fill color (name, 0-1 floats, or 0-255 ints).
            alpha: Fill opacity (0-1). Bands are usually semi-transparent.
            label: Optional legend label.
            edge: Draw a stroke around the band outline.
            edge_color: Stroke color (defaults to ``color``).
            edge_width: Stroke width when ``edge`` is True.
            animate: Reveal duration in seconds (left-to-right). Defaults to
                ``config.line_animate``.
            delay: Universal time offset (seconds).
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        y_low = np.asarray(y_low, dtype=float)
        y_high = np.asarray(y_high, dtype=float)
        if y_low.ndim == 0:
            y_low = np.full(n, float(y_low))
        if y_high.ndim == 0:
            y_high = np.full(n, float(y_high))
        if animate is UNSET:
            animate = getattr(config, "line_animate", 2.0)
        self.elements.append({
            "type": "band",
            "x": list(x),
            "y_low": list(y_low),
            "y_high": list(y_high),
            "color": color,
            "alpha": float(alpha),
            "label": label,
            "edge": bool(edge),
            "edge_color": edge_color if edge_color is not None else color,
            "edge_width": edge_width,
            "animate": animate,
            "delay": float(delay),
        })
        return self

    def area(self, x, y, baseline=0.0, color="blue", alpha=0.4, label=None,
             edge=True, edge_color=None, linewidth=4, animate=UNSET, delay=0.0,
             **kwargs):
        """Filled area chart from ``baseline`` up to ``y``.

        Implemented on top of :meth:`fill_between` (lower boundary = baseline)
        with a top edge stroke drawn by default.
        """
        x = np.asarray(x, dtype=float).ravel()
        return self.fill_between(
            x, np.full(x.size, float(baseline)), y,
            color=color, alpha=alpha, label=label, edge=edge,
            edge_color=edge_color if edge_color is not None else color,
            edge_width=linewidth, animate=animate, delay=delay,
        )

    def gradient_area(self, x, y_top, y_bottom=0.0, top_color="blue",
                      bottom_color="white", alpha=1.0, ramp_scatter=0.0,
                      gradient_top=None, gradient_bottom=None,
                      label=None, animate=UNSET, delay=0.0, **kwargs):
        """A shape whose top edge hugs a curve, colored with a native AE
        vertical gradient instead of a flat fill.

        The shape geometry is exactly :meth:`fill_between`'s (upper boundary
        ``y_top`` left->right, then lower boundary ``y_bottom`` back
        right->left) -- so it follows the data just like an area chart. The
        *coloring*, though, is a single straight top->bottom AE "Gradient
        Ramp" effect (``top_color`` -> ``bottom_color``) rather than a flat
        fill or (like :meth:`fill_between` would need for a fade) dozens of
        stacked translucent bands -- one shape layer, one effect, regardless
        of how smooth the fade looks. The gradient itself is a plain
        vertical wash and does NOT bend to follow the curve -- only the
        shape's silhouette does.

        Args:
            x: Shared x values (1D) -- the curve the top edge hugs.
            y_top: Upper boundary the shape hugs (scalar or 1D, same length
                as ``x``). Typically the line/series you're highlighting.
            y_bottom: Lower boundary (scalar or 1D). Defaults to ``0`` (the
                axis baseline), matching :meth:`area`.
            top_color, bottom_color: Gradient endpoint colors (name, 0-1
                floats, or 0-255 ints).
            alpha: Overall layer opacity (0-1).
            ramp_scatter: AE's "Ramp Scatter" dither amount (0 = a perfectly
                smooth ramp, matching the effect's own default range).
            gradient_top, gradient_bottom: Data y-values the gradient itself
                is anchored to (``top_color`` at ``gradient_top``,
                ``bottom_color`` at ``gradient_bottom``) -- independent of
                the curve's local height at any given x. Default to
                ``max(y_top)`` / ``min(y_bottom)`` so the ramp spans the
                shape's full vertical extent as one straight line.
            label: Optional legend label.
            animate: Fade-in duration in seconds (0 = appears instantly).
            delay: Start delay in seconds.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        y_top_arr = np.asarray(y_top, dtype=float)
        if y_top_arr.ndim == 0:
            y_top_arr = np.full(n, float(y_top_arr))
        y_bottom_arr = np.asarray(y_bottom, dtype=float)
        if y_bottom_arr.ndim == 0:
            y_bottom_arr = np.full(n, float(y_bottom_arr))
        if gradient_top is None:
            gradient_top = float(np.nanmax(y_top_arr))
        if gradient_bottom is None:
            gradient_bottom = float(np.nanmin(y_bottom_arr))
        if animate is UNSET:
            animate = 0.0
        self.elements.append({
            "type": "gradient_area",
            "x": list(x),
            "y_top": list(y_top_arr), "y_bottom": list(y_bottom_arr),
            "gradient_top": float(gradient_top), "gradient_bottom": float(gradient_bottom),
            "top_color": top_color, "bottom_color": bottom_color,
            "alpha": float(alpha),
            "ramp_scatter": float(ramp_scatter),
            "label": label,
            "animate": float(animate),
            "delay": float(delay),
        })
        return self

    def errorbar(self, x, y, yerr=None, xerr=None, color="black", linewidth=2,
                 capsize=6, alpha=1.0, label=None, animate=UNSET, delay=0.0,
                 **kwargs):
        """Draw error bars at ``(x, y)`` with optional x/y error extents.

        ``yerr`` / ``xerr`` may be ``None``, a scalar (symmetric), or a 1D array
        (per-point symmetric). The caps are short ticks of length ``capsize``
        (pixels) drawn perpendicular to each bar.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = x.size

        def _norm(err):
            if err is None:
                return None
            err = np.asarray(err, dtype=float)
            if err.ndim == 0:
                err = np.full(n, float(err))
            return list(np.abs(err))

        if animate is UNSET:
            animate = getattr(config, "line_animate", 2.0)
        self.elements.append({
            "type": "errorbar",
            "x": list(x),
            "y": list(y),
            "yerr": _norm(yerr),
            "xerr": _norm(xerr),
            "color": color,
            "linewidth": linewidth,
            "capsize": float(capsize),
            "alpha": float(alpha),
            "label": label,
            "animate": animate,
            "delay": float(delay),
        })
        return self

    def axhline(self, y, color="gray", linestyle="dashed", linewidth=3,
                alpha=1.0, label=None, animate=UNSET, delay=0.0, **kwargs):
        """Draw a horizontal reference line spanning the plot at data ``y``."""
        return self._add_refline("h", y, color, linestyle, linewidth, alpha,
                                 label, animate, delay)

    def axvline(self, x, color="gray", linestyle="dashed", linewidth=3,
                alpha=1.0, label=None, animate=UNSET, delay=0.0, **kwargs):
        """Draw a vertical reference line spanning the plot at data ``x``."""
        return self._add_refline("v", x, color, linestyle, linewidth, alpha,
                                 label, animate, delay)

    def _add_refline(self, orient, value, color, linestyle, linewidth, alpha,
                     label, animate, delay):
        if animate is UNSET:
            animate = getattr(config, "line_animate", 2.0)
        self.elements.append({
            "type": "refline",
            "orient": orient,
            "value": float(value),
            "color": color,
            "linestyle": linestyle,
            "linewidth": linewidth,
            "alpha": float(alpha),
            "label": label,
            "animate": animate,
            "delay": float(delay),
        })
        return self

    def add_trendline(self, x, y, kind="linear", degree=2, color="red",
                      linestyle="dashed", linewidth=3, label=None, npoints=100,
                      animate=UNSET, delay=0.0, **kwargs):
        """Fit a trend/regression line to ``(x, y)`` and overlay it.

        Args:
            kind: ``"linear"`` (degree-1 fit) or ``"poly"`` (degree ``degree``).
            degree: Polynomial degree when ``kind="poly"``.
            npoints: Number of points used to draw the smooth fitted curve.

        The fit is computed in Python with :func:`numpy.polyfit` and emitted as
        an ordinary dashed line element.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        deg = 1 if kind == "linear" else int(degree)
        coeffs = np.polyfit(x, y, deg)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), int(npoints))
        ys = np.polyval(coeffs, xs)
        return self.plot(xs, ys, color=color, linestyle=linestyle,
                         linewidth=linewidth, label=label, animate=animate,
                         delay=delay, **kwargs)

    def pie(self, values, labels=None, colors=None, donut=0.0, radius=None,
            start_angle=0.0, alpha=1.0, animate=UNSET, delay=0.0,
            stroke_color=None, stroke_width=0, show_labels=True,
            show_percent=False, label_color=None, gap=0.0, roundness=0.0,
            leader_lines=False, **kwargs):
        """Pie or donut chart.

        Pie charts are polar: they ignore x/y limits, ticks, and the grid, and
        are centered in the panel. Each wedge is drawn as a filled sector and
        fades in sequentially.

        Args:
            values: Wedge magnitudes (auto-normalized to fractions).
            labels: Optional wedge labels drawn outside each slice.
            colors: Optional wedge colors (defaults to a cycling palette).
            donut: Inner-hole radius as a fraction of the outer radius (0 = full
                pie, e.g. 0.5 = donut).
            radius: Outer radius in pixels (defaults to 40% of the smaller panel
                dimension).
            start_angle: Rotation offset in degrees from 12 o'clock (clockwise).
            alpha: Wedge fill opacity (0-1).
            animate: Total reveal duration over which wedges appear in order.
            show_labels: Draw ``labels`` next to each wedge.
            show_percent: Append the percentage to each label.
            gap: Gap **in pixels** between adjacent wedges, as an absolute
                distance. The gap is reserved from the circle and the rest is
                split proportionally, so every slice shrinks in proportion (small
                slices are not eaten away). Edges are drawn as parallel offset
                lines that meet at an apex (pie) or a deeper inner inset (donut),
                so the gap keeps a constant width all the way in instead of
                pinching toward the center.
            roundness: Corner-rounding radius **in pixels** applied to each
                wedge (rounds the sharp corners where the radial edges meet the
                arcs). 0 = square corners.
            leader_lines: If True, draw a callout line from each slice out to its
                label, colored to match that slice, and place the label at the
                end of the line.
        """
        values = np.asarray(values, dtype=float).ravel()
        total = float(np.sum(values))
        if total <= 0:
            raise ValueError("pie() requires values that sum to a positive number")
        fracs = values / total
        if animate is UNSET:
            animate = getattr(config, "line_animate", 2.0)
        colors = self._resolve_series_colors(colors, len(values))
        self.elements.append({
            "type": "pie",
            "values": list(values),
            "fracs": list(fracs),
            "labels": list(labels) if labels is not None else None,
            "colors": list(colors),
            "donut": float(donut),
            "radius": radius,
            "start_angle": float(start_angle),
            "alpha": float(alpha),
            "animate": animate,
            "delay": float(delay),
            "stroke_color": stroke_color,
            "stroke_width": stroke_width,
            "show_labels": bool(show_labels),
            "show_percent": bool(show_percent),
            "label_color": label_color,
            "gap": float(gap),
            "roundness": float(roundness),
            "leader_lines": bool(leader_lines),
            "label": None,  # excluded from the standard legend; uses slice labels
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

    def set_view_keyframes(self, view_keyframes, hold=False, sample_fps=None,
                           ease=True, ease_speed=None, ease_influence=None,
                           ease_influence_start=None, ease_influence_end=None,
                           adaptive_ticks=True, adaptive_tick_target_n=7,
                           adaptive_tick_fade=0.4,
                           adaptive_tick_min_opacity=0.72,
                           visibility_fade=0.35):
        """Animate the graph's view window (x/y limits) over time.

        Args:
            view_keyframes (dict): Maps keyframe time (seconds, float) to a pair
                ``[(xmin, xmax), (ymin, ymax)]`` describing the view window at
                that time. With two or more keys the bounds shift over time,
                re-mapping every coordinate-driven layer (plots, grid, ticks,
                tick labels, axes) as a true axis rescale (stroke widths and
                font sizes stay constant). Grid lines and tick marks/labels are
                only visible while their data coordinate falls inside the
                current view window on the relevant axis. Times before the first / after the
                last key are clamped (hold).
            adaptive_ticks (bool): When True (default) and an axis uses a
                linear scale, ticks/gridlines/labels are generated at several
                nested densities instead of one fixed set sized for the
                *widest* keyframed view. Finer ticks fade in smoothly as the
                view zooms in past the span where their density becomes
                appropriate, and fade back out zooming out again -- similar
                to how map tiles reveal more detail as you zoom in, instead
                of always showing however many gridlines fit the most
                zoomed-out frame. Set False to keep the single fixed tick set
                (sized for the first keyframe's view) at every zoom level.
                Has no effect on ``log``-scale axes, which always use the
                single fixed set. Ignored unless ``set_xticks``/``set_yticks``
                hasn't been called (manual tick lists are never densified).
            adaptive_tick_target_n (int): Roughly how many ticks should be on
                screen at any given zoom level; smaller = sparser levels
                (fewer, bigger zoom-in jumps needed to reveal more detail).
            adaptive_tick_fade (float): How long, in wall-clock seconds, a
                finer tick level takes to fade in once the view zooms in past
                its density threshold (and fade out zooming back past it) --
                the same fixed real-time treatment as ``visibility_fade``,
                just applied to the density crossfade instead of the
                view-window edge. Independent of fps and of how fast the
                view happens to be zooming when it crosses the threshold.
            adaptive_tick_min_opacity (float): Shapes the reveal/hide ramp so
                it front-loads (reaches near-full opacity quickly, then eases
                the rest of the way) instead of fading in linearly --
                prevents finer ticks from lingering at a low, "ghostly"
                opacity for a noticeable chunk of ``adaptive_tick_fade``.
                ``1.0`` front-loads the hardest, approximating (but not
                literally performing) a hard on/off snap; ``0.0`` is a plain
                linear fade.
            visibility_fade (float): How long, in wall-clock seconds, ticks/
                gridlines/labels take to fade in or out whenever the animated
                view window crosses past them (e.g. a gridline panning off the
                left edge, or a new one panning in from the right). This is
                always a fixed real-time duration -- it does NOT shrink as
                ``fps`` (or the composition frame rate) goes up; a higher fps
                just samples the same fade more finely, which should look
                smoother, not shorter. If fades still look like an instant pop
                rather than a fade, raise this (e.g. ``0.5``-``0.8``) --
                the previous hardcoded default (``0.12``s) was likely too
                brief to read as a fade at all. Has no effect on the
                axis-draw entrance reveal (ticks lighting up as the axis
                first draws past them), which uses its own timing.
            hold (bool): If True, snap between keyframe views instead of
                interpolating.
            sample_fps (float, optional): How densely (per second) to sample the
                view animation when baking AE keyframes. Higher = smoother
                approximation of the nonlinear motion, but larger JSX. Defaults
                to the composition ``fps``.
            ease (bool): When True (default), each transition between two
                keyframe views follows an ease-in-out cubic bezier (the bounds
                accelerate out of one view and decelerate into the next) instead
                of moving at a constant rate. Ignored when ``hold=True``.
            ease_influence (float, optional): Symmetric ease strength as a percent
                (0-100); larger = more pronounced slow-in/slow-out. Defaults to
                the graph's ``ease_influence`` (the same value Easy Ease uses).
                Used for both ends unless overridden by ``ease_influence_start`` /
                ``ease_influence_end``.
            ease_influence_start (float, optional): Ease strength (0-100) of the
                *outgoing* handle at the start of the transition (how softly the
                bounds accelerate out of the first view). Overrides
                ``ease_influence`` for the start only. Defaults to
                ``ease_influence``.
            ease_influence_end (float, optional): Ease strength (0-100) of the
                *incoming* handle at the end of the transition (how softly the
                bounds decelerate into the final view). Overrides
                ``ease_influence`` for the end only. Defaults to
                ``ease_influence``. Combine with ``ease_influence_start`` for an
                asymmetric ease (e.g. soft launch, hard stop).
            ease_speed (float, optional): Velocity at the keyframes as a percent
                (0-100). 0 (default, from the graph's ``ease_speed``) gives the
                classic Easy Ease with a full stop at each keyframe; higher
                values keep more momentum through the keyframes.

        Example::

            graph.set_view_keyframes({
                0.0: [(0, 10), (0, 100)],
                3.0: [(4, 6),  (40, 60)],
            }, ease_influence=50)

            # asymmetric: ease gently out of the first view, snap into the last
            graph.set_view_keyframes({
                0.0: [(0, 10), (0, 100)],
                3.0: [(4, 6),  (40, 60)],
            }, ease_influence_start=80, ease_influence_end=10)
        """
        if view_keyframes is None:
            self._view_kf = None
            self._view_animated = False
            return self
        if not isinstance(view_keyframes, dict):
            raise TypeError("view_keyframes must be a dict mapping time -> [(xmin, xmax), (ymin, ymax)]")
        if len(view_keyframes) < 1:
            raise ValueError("view_keyframes must contain at least one keyframe")
        parsed = []
        for t, bounds in view_keyframes.items():
            try:
                t = float(t)
            except (TypeError, ValueError):
                raise ValueError(f"view_keyframes keys must be numeric times; got {t!r}")
            if (not isinstance(bounds, (list, tuple))) or len(bounds) != 2:
                raise ValueError(
                    f"view_keyframes[{t}] must be [(xmin, xmax), (ymin, ymax)]; got {bounds!r}"
                )
            xpair, ypair = bounds
            if len(xpair) != 2 or len(ypair) != 2:
                raise ValueError(
                    f"view_keyframes[{t}] must be [(xmin, xmax), (ymin, ymax)]; got {bounds!r}"
                )
            xmin, xmax = float(xpair[0]), float(xpair[1])
            ymin, ymax = float(ypair[0]), float(ypair[1])
            if xmin == xmax or ymin == ymax:
                raise ValueError(f"view_keyframes[{t}] has a degenerate (zero-width) range: {bounds!r}")
            parsed.append((t, xmin, xmax, ymin, ymax))
        parsed.sort(key=lambda r: r[0])
        self._view_kf = parsed
        self._view_animated = len(parsed) >= 2
        self._view_hold = bool(hold)
        self._view_sample_fps = sample_fps
        self._view_ease = bool(ease)
        self._view_ease_speed = ease_speed
        self._view_ease_influence = ease_influence
        self._view_ease_influence_start = ease_influence_start
        self._view_ease_influence_end = ease_influence_end
        self._adaptive_ticks = bool(adaptive_ticks)
        self._adaptive_tick_target_n = max(2, int(adaptive_tick_target_n))
        self._adaptive_tick_fade = max(1e-3, float(adaptive_tick_fade))
        self._adaptive_tick_min_opacity = max(0.0, min(1.0, float(adaptive_tick_min_opacity)))
        self._view_visibility_fade = max(1e-3, float(visibility_fade))
        return self

    def _view_ease_fraction(self, f):
        """Apply the configured ease curve to a linear blend fraction ``f``.

        Reuses the same Easy-Ease cubic-bezier (``_cubic_bezier_ease``) the rest
        of the library uses, so view transitions match the look of element
        entrance easing. ``ease_influence``/``ease_speed`` default to the graph's
        Easy-Ease settings.
        """
        if not getattr(self, "_view_ease", True) or self._view_hold:
            return f
        infl = self._view_ease_influence
        if infl is None:
            infl = getattr(self, "ease_influence", 33)
        spd = self._view_ease_speed
        if spd is None:
            spd = getattr(self, "ease_speed", 0)
        # Per-keyframe (start/end) influence overrides: the outgoing handle uses
        # the start influence, the incoming handle the end influence. Each falls
        # back to the symmetric ``ease_influence`` when not set.
        infl_start = getattr(self, "_view_ease_influence_start", None)
        infl_end = getattr(self, "_view_ease_influence_end", None)
        start = infl if infl_start is None else infl_start
        end = infl if infl_end is None else infl_end
        return self._cubic_bezier_ease(f, start or 0, spd or 0,
                                       influence_end=(end or 0))

    def _view_limits_at(self, t):
        """Return (xmin, xmax, ymin, ymax) for the view window at time ``t``.

        Linearly interpolates between surrounding keyframes (or snaps when
        ``hold`` was requested). Times outside the keyframe range clamp to the
        first/last key.
        """
        kf = self._view_kf
        if not kf:
            return None
        if t <= kf[0][0]:
            _, xmin, xmax, ymin, ymax = kf[0]
            return xmin, xmax, ymin, ymax
        if t >= kf[-1][0]:
            _, xmin, xmax, ymin, ymax = kf[-1]
            return xmin, xmax, ymin, ymax
        for i in range(len(kf) - 1):
            t0, x0min, x0max, y0min, y0max = kf[i]
            t1, x1min, x1max, y1min, y1max = kf[i + 1]
            if t0 <= t <= t1:
                if self._view_hold or t1 == t0:
                    return x0min, x0max, y0min, y0max
                f = self._view_ease_fraction((t - t0) / (t1 - t0))
                return (
                    x0min + (x1min - x0min) * f,
                    x0max + (x1max - x0max) * f,
                    y0min + (y1min - y0min) * f,
                    y0max + (y1max - y0max) * f,
                )
        _, xmin, xmax, ymin, ymax = kf[-1]
        return xmin, xmax, ymin, ymax

    def _view_sample_times(self, extra_times=None):
        """Dense list of sample times (seconds) spanning the view animation.

        Samples from the first to the last keyframe at ``sample_fps`` (default
        comp fps) so the nonlinear pixel motion of a fixed data point under
        linearly-changing limits is well approximated by LINEAR AE keyframes.
        With ``hold`` mode, only the keyframe times themselves are returned.
        ``extra_times`` (e.g. an evolving element's own frame_times) are merged
        in and de-duplicated.
        """
        kf = self._view_kf
        if not kf:
            return []
        t_start = kf[0][0]
        t_end = kf[-1][0]
        key_times = [r[0] for r in kf]
        times = set(key_times)
        if not self._view_hold:
            fps = self._view_sample_fps or self.fps or 60
            dt = 1.0 / float(fps)
            # Only densely sample segments where the view actually changes; a
            # held (constant) segment needs just its two endpoint keyframes.
            for a in range(len(kf) - 1):
                t0, x0min, x0max, y0min, y0max = kf[a]
                t1, x1min, x1max, y1min, y1max = kf[a + 1]
                if (x0min, x0max, y0min, y0max) == (x1min, x1max, y1min, y1max):
                    continue
                n = max(1, int(math.ceil((t1 - t0) / dt)))
                for s in range(n + 1):
                    times.add(t0 + s * dt)
        if extra_times:
            times.update(float(t) for t in extra_times if t_start <= float(t) <= t_end)
        return sorted(round(t, 6) for t in times)

    def _view_union_range(self):
        """(xmin, xmax, ymin, ymax) spanning every keyframe view window."""
        kf = self._view_kf
        xmins = [r[1] for r in kf]
        xmaxs = [r[2] for r in kf]
        ymins = [r[3] for r in kf]
        ymaxs = [r[4] for r in kf]
        return min(xmins), max(xmaxs), min(ymins), max(ymaxs)

    def _view_set_fixed_ticks(self):
        """Compute the tick set(s) used across an animated view.

        By default (``adaptive_ticks=True`` on ``set_view_keyframes``, linear
        scale) this builds a multi-level, map-tile-style tick set via
        ``_view_set_adaptive_ticks``: finer ticks exist in the data but stay
        faded out until the view zooms in past the span where they're an
        appropriate density, then fade in (and back out zooming out) --
        see ``_emit_zoom_level_visibility_kf``. Otherwise (or for log-scale
        axes, or when the user set explicit ticks) it falls back to a single
        fixed tick set over the union of all keyframe views, unchanged for
        the whole animation.
        """
        ux_min, ux_max, uy_min, uy_max = self._view_union_range()
        adaptive = getattr(self, "_adaptive_ticks", True)
        if self.xticks is None:
            if adaptive and self._normalize_scale(self.xscale) == "linear":
                self._view_set_adaptive_ticks("x", ux_min, ux_max)
            else:
                self._auto_set_ticks_from_padded("x", ux_min, ux_max)
        if self.yticks is None:
            if adaptive and self._normalize_scale(self.yscale) == "linear":
                self._view_set_adaptive_ticks("y", uy_min, uy_max)
            else:
                self._auto_set_ticks_from_padded("y", uy_min, uy_max)

    def _view_set_adaptive_ticks(self, axis, vmin_union, vmax_union):
        """Build a nested, multi-density ('map-tile style') tick set for an
        animated view on a linear-scale axis.

        Samples the current view span across the whole animation to find its
        widest and narrowest extremes, then builds a ladder of nested nice
        spacings (see ``_ladder_step_snap``/``_ladder_next_finer``) spanning
        from a density appropriate for the widest view (level 0 -- always
        on, exactly the old fixed-tick behaviour) down to one appropriate for
        the narrowest (the finest level). Every tick position across every
        level is generated once (over the union range) and tagged with the
        *coarsest* level it belongs to; ``_emit_zoom_level_visibility_kf``
        later fades each one in/out based on how that level's threshold span
        compares to the current view span at each moment.
        """
        target_n = getattr(self, "_adaptive_tick_target_n", 7)
        spans = []
        for t in self._view_sample_times():
            xmn, xmx, ymn, ymx = self._view_limits_at(t)
            vmin, vmax = (xmn, xmx) if axis == "x" else (ymn, ymx)
            spans.append(abs(vmax - vmin))
        if not spans:
            spans = [abs(vmax_union - vmin_union)]
        span_max, span_min = max(spans), min(spans)

        step_coarse = self._ladder_step_snap(span_max / max(1, target_n - 1))
        step_fine = self._ladder_step_snap(span_min / max(1, target_n - 1))

        ladder = [step_coarse]
        guard = 0
        while ladder[-1] > step_fine * 1.0001 and guard < 30:
            nxt = self._ladder_next_finer(ladder[-1])
            if nxt <= 0 or nxt >= ladder[-1]:
                break
            ladder.append(nxt)
            guard += 1

        value_to_level = {}
        for level_idx, step in enumerate(ladder):
            for v in self._ticks_for_step(vmin_union, vmax_union, step):
                key = round(v, 9)
                if key not in value_to_level:
                    value_to_level[key] = level_idx

        positions = sorted(value_to_level.keys())
        scale = self._normalize_scale(getattr(self, "xscale" if axis == "x" else "yscale"))
        if axis == "x" and self.percent_tick_labels:
            step0 = positions[1] - positions[0] if len(positions) > 1 else 1.0
            fmt = "{:.1f}%" if abs(step0) < 1 else "{:.0f}%"
            labels = [fmt.format(abs(p)) for p in positions]
        else:
            labels = [self._format_tick_label(p, scale) for p in positions]
        thresholds = [step * target_n for step in ladder]

        if axis == "x":
            self._xtick_labels_auto = True
            self.xticks = list(zip(positions, labels))
            self._xtick_native_level = value_to_level
            self._xtick_level_thresholds = thresholds
        else:
            self._ytick_labels_auto = True
            self.yticks = list(zip(positions, labels))
            self._ytick_native_level = value_to_level
            self._ytick_level_thresholds = thresholds

    def _adaptive_tick_level(self, axis, pos):
        """Native (coarsest) zoom-density level a given tick belongs to, or
        0 if this axis has no adaptive-tick levels (static/single-level)."""
        levels = getattr(self, f"_{axis}tick_native_level", None)
        if not levels:
            return 0
        return levels.get(round(pos, 9), 0)

    def _adaptive_tick_threshold(self, axis, level):
        """View span at which ``level`` becomes an appropriate tick density,
        or None if there's no adaptive-tick data for this axis/level."""
        thresholds = getattr(self, f"_{axis}tick_level_thresholds", None)
        if not thresholds or level <= 0 or level >= len(thresholds):
            return None
        return thresholds[level]

    @staticmethod
    def _interp_frame_rows(frames, times, t, hold):
        """Linearly (or hold-) interpolate a list of per-frame rows at time ``t``.

        ``frames`` is a list of equal-length rows; ``times`` is the matching
        ascending list of frame times. Used to compose data morphs with an
        animated view window.
        """
        if t <= times[0]:
            return list(frames[0])
        if t >= times[-1]:
            return list(frames[-1])
        for idx in range(len(times) - 1):
            if times[idx] <= t <= times[idx + 1]:
                if hold or times[idx + 1] == times[idx]:
                    return list(frames[idx])
                f = (t - times[idx]) / (times[idx + 1] - times[idx])
                return [a + (b - a) * f for a, b in zip(frames[idx], frames[idx + 1])]
        return list(frames[-1])

    def _view_composite_times(self, frame_times):
        """Dense sample times spanning both the view animation and a data
        animation (``frame_times``), with all key/frame times included."""
        kf = self._view_kf
        vt0, vt1 = kf[0][0], kf[-1][0]
        dt0, dt1 = frame_times[0], frame_times[-1]
        t_lo = min(vt0, dt0)
        t_hi = max(vt1, dt1)
        fps = self._view_sample_fps or self.fps or 60
        step = 1.0 / float(fps)
        nseg = max(1, int(math.ceil((t_hi - t_lo) / step)))
        samp = [t_lo + s * step for s in range(nseg + 1)]
        samp.append(t_hi)
        samp.extend(float(t) for t in frame_times)
        samp.extend(r[0] for r in kf)
        return sorted({round(s, 6) for s in samp})

    def _emit_view_path_kf(self, script, path_var, tag, geom_fn, closed=False):
        """Emit LINEAR Path keyframes for a shape whose vertices follow the view.

        ``geom_fn(xmin, xmax, ymin, ymax)`` returns a list of (sx, sy) shape
        vertices for the given view window.
        """
        closed_js = "true" if closed else "false"
        for k, t in enumerate(self._view_sample_times()):
            xmin, xmax, ymin, ymax = self._view_limits_at(t)
            verts = geom_fn(xmin, xmax, ymin, ymax)
            pts_js = ",".join(f"[{sx},{sy}]" for sx, sy in verts)
            svar = f"vkShp_{tag}_{k}"
            script.append(f"var {svar} = new Shape();\n")
            script.append(f"{svar}.vertices = [{pts_js}];\n")
            script.append(f"{svar}.closed = {closed_js};\n")
            script.append(f"{path_var}.setValueAtTime({t}, {svar});\n")
        script.append(f"setKeyInterp({path_var}, {'true' if self._view_hold else 'false'});\n")

    def _emit_view_pos_kf(self, script, pos_var, geom_fn, z=False):
        """Emit LINEAR Position keyframes following the view window.

        ``geom_fn(xmin, xmax, ymin, ymax)`` returns an (sx, sy) tuple.
        """
        for t in self._view_sample_times():
            xmin, xmax, ymin, ymax = self._view_limits_at(t)
            sx, sy = geom_fn(xmin, xmax, ymin, ymax)
            if z:
                script.append(f"{pos_var}.setValueAtTime({t}, [{sx}, {sy}, 0]);\n")
            else:
                script.append(f"{pos_var}.setValueAtTime({t}, [{sx}, {sy}]);\n")
        script.append(f"setKeyInterp({pos_var}, {'true' if self._view_hold else 'false'});\n")

    def _emit_screen_space_mask_jsx(self, script, layer_var):
        """Add a rectangular Mask to ``layer_var`` that clips its rendered
        content to the plot box (``self.width`` x ``self.height``) -- the
        chart's "screen space" -- no matter how the layer (or its parent
        chain) is itself transformed or animated.

        Only emitted when the view window is animated via
        ``set_view_keyframes``: a static view already keeps drawn geometry
        within the box (data is padded/limited to ``xlim``/``ylim`` up
        front), but panning/zooming the view remaps every data-driven vertex
        against the *current* window, so points whose data now falls outside
        that window land far outside the box in pixel space -- visible as
        stray line/bar/marker fragments poking past the frame. A baked,
        per-frame keyframed mask would have to duplicate every Position/Path
        keyframe already on the layer, so instead this uses a live
        expression: ``PlotAnchor.toComp(...)`` finds the box's four corners
        in comp space (automatically following any Scale keyframes on
        ``PlotAnchor``, e.g. the film-style push-in zoom), and
        ``thisLayer.fromComp(...)`` converts them into this layer's own
        local space every frame -- correct regardless of whether this
        specific layer, a parent null, or neither carries the view-window
        animation.
        """
        script.extend(self._screen_space_mask_lines(layer_var))

    def _screen_space_mask_lines(self, layer_var):
        """Line list version of ``_emit_screen_space_mask_jsx`` (for call
        sites that need to indent the result, e.g. inside an ``if {}``
        block emitted by a helper). Returns ``[]`` when the view window
        isn't animated."""
        if not self._view_animated:
            return []
        pa_name = getattr(self, "_current_plotanchor_name", "PlotAnchor")
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        expr = (
            "(function(){"
            f"var W2={half_w},H2={half_h};"
            f"var pa=thisComp.layer({json.dumps(pa_name)});"
            "var c0=pa.toComp([-W2,-H2,0]);"
            "var c1=pa.toComp([W2,-H2,0]);"
            "var c2=pa.toComp([W2,H2,0]);"
            "var c3=pa.toComp([-W2,H2,0]);"
            "var p0=thisLayer.fromComp(c0);"
            "var p1=thisLayer.fromComp(c1);"
            "var p2=thisLayer.fromComp(c2);"
            "var p3=thisLayer.fromComp(c3);"
            "return createPath([[p0[0],p0[1]],[p1[0],p1[1]],[p2[0],p2[1]],[p3[0],p3[1]]],[],[],true);"
            "})()"
        )
        var_i = getattr(self, "_screen_mask_counter", 0)
        self._screen_mask_counter = var_i + 1
        mask_var = f"screenMask{var_i}"
        return [
            f"var {mask_var} = {layer_var}.property('ADBE Mask Parade').addProperty('ADBE Mask Atom');\n",
            f"{mask_var}.maskMode = MaskMode.ADD;\n",
            f"{mask_var}.property('ADBE Mask Shape').expression = {json.dumps(expr)};\n",
        ]

    def _view_value_in_range(self, val, vmin, vmax, scale: str):
        """True when a single data value lies inside the view window on one axis."""
        return bool(self._filter_ticks_to_view([val], vmin, vmax, scale))

    def _view_in_range_at(self, val, axis, t):
        """True when ``val`` lies inside the view window on ``axis`` at time ``t``."""
        xmin, xmax, ymin, ymax = self._view_limits_at(t)
        scale = self._normalize_scale(self.xscale if axis == "x" else self.yscale)
        if axis == "x":
            return self._view_value_in_range(val, xmin, xmax, scale)
        return self._view_value_in_range(val, ymin, ymax, scale)

    def _emit_boolean_crossfade_kf(
        self,
        script,
        prop_expr,
        times,
        in_view_fn,
        visible_opacity=100,
        entrance_start=None,
        entrance_fade=0.5,
        fade_seconds=0.35,
        shape_gamma=1.0,
    ):
        """Bake an opacity curve from a boolean "is this on-screen right now"
        function sampled at ``times`` (real seconds): full ``visible_opacity``
        while ``in_view_fn`` holds True, fading to 0 over exactly
        ``fade_seconds`` of wall-clock time whenever it flips off, and back
        up over the same fixed duration whenever it flips back on --
        independent of the composition's fps and of how fast whatever drives
        ``in_view_fn`` (e.g. the view window) happens to be changing at that
        moment.

        ``shape_gamma`` reshapes the fade-in/out ramp: ``1.0`` is a plain
        linear fade; values below ``1.0`` front-load it (rises to near-full
        quickly then eases in the rest of the way) so it never lingers at a
        low, "ghostly" opacity -- ``shape_gamma`` near 0 approximates a hard
        on/off snap while remaining perfectly continuous (no popping).

        Returns the cleaned ``[(time, opacity), ...]`` keyframe list.
        """
        in_views = [in_view_fn(t) for t in times]
        use_entrance = entrance_start is not None and in_views[0]
        entrance_start = float(entrance_start or 0.0)

        kfs = []

        def add(t, op):
            kfs.append((round(float(t), 6), round(float(op), 3)))

        # --- initial state / entrance reveal -------------------------------
        if in_views[0]:
            if use_entrance:
                add(0.0, 0)
                if entrance_start > 1e-6:
                    add(entrance_start, 0)
                add(entrance_start + entrance_fade, visible_opacity)
                reveal_done = entrance_start + entrance_fade
            else:
                add(0.0, visible_opacity)
                reveal_done = 0.0
        else:
            add(0.0, 0)
            reveal_done = 0.0

        # --- crossings (fixed-duration fade out / in) -----------------------
        plain = abs(shape_gamma - 1.0) < 1e-6
        shape_steps = 6  # extra points sampled across the fade for a shaped (non-linear) ramp
        for i in range(1, len(times)):
            t_prev, t = times[i - 1], times[i]
            prev, cur = in_views[i - 1], in_views[i]
            if t <= reveal_done + 1e-6:
                # Still inside the entrance ramp; let the reveal own it.
                continue
            if prev and not cur:        # leaving -> fade out
                add(t_prev, visible_opacity)        # anchor full opacity at exit
                if plain:
                    add(t_prev + fade_seconds, 0)
                else:
                    for s in range(1, shape_steps + 1):
                        frac = s / shape_steps
                        add(t_prev + frac * fade_seconds, visible_opacity * (1.0 - frac ** shape_gamma))
            elif not prev and cur:      # entering -> fade in
                add(t, 0)                           # anchor hidden at entry
                if plain:
                    add(t + fade_seconds, visible_opacity)
                else:
                    for s in range(1, shape_steps + 1):
                        frac = s / shape_steps
                        add(t + frac * fade_seconds, visible_opacity * (frac ** shape_gamma))

        # Collapse keyframes that share a timestamp (keep the last value).
        cleaned = []
        for t, op in kfs:
            if cleaned and abs(cleaned[-1][0] - t) < 1e-6:
                cleaned[-1] = (t, op)
            else:
                cleaned.append((t, op))

        for t, op in cleaned:
            script.append(f"{prop_expr}.setValueAtTime({t}, {op});\n")
        # Linear keys so the ramps actually fade (view_hold only snaps geometry).
        script.append(f"setKeyInterp({prop_expr}, false);\n")
        return cleaned

    def _emit_view_axis_visibility_kf(
        self,
        script,
        prop_expr,
        val,
        axis,
        visible_opacity=100,
        entrance_start=None,
        entrance_fade=0.5,
    ):
        """Opacity that combines the axis-draw reveal with view-window clipping.

        Two independent behaviours, layered onto one baked opacity curve:

        1. Entrance reveal: if the element starts inside the frame, it fades in
           once at the beginning, synced to the axis trim sweep
           (``entrance_start`` = the time that sweep reaches this tick;
           ``entrance_fade`` = how long the fade lasts). This is exactly the old
           "ticks light up as the axis draws past them" behaviour.

        2. View-window clipping: whenever the animated view window later crosses
           past the element, it fades out (on exit) or in (on entry) quickly
           over ``self._view_visibility_fade`` seconds, rather than popping.

        Keyframes are emitted only at the entrance ramp and at each window
        crossing (with explicit anchors), so the held value persists between
        them and the fades last their full intended duration.
        """
        win_fade = getattr(self, "_view_visibility_fade", 0.35) or 0.35
        times = sorted(set([0.0] + list(self._view_sample_times())))
        self._emit_boolean_crossfade_kf(
            script, prop_expr, times,
            in_view_fn=lambda t: self._view_in_range_at(val, axis, t),
            visible_opacity=visible_opacity,
            entrance_start=entrance_start, entrance_fade=entrance_fade,
            fade_seconds=win_fade, shape_gamma=1.0,
        )
        if self.easy_ease:
            script.append(
                f"applyEasyEase({prop_expr}, {self.ease_speed}, {self.ease_influence});\n"
            )

    def _emit_zoom_level_visibility_kf(
        self,
        script,
        prop_expr,
        val,
        axis,
        threshold_span,
        visible_opacity=100,
        entrance_start=None,
        entrance_fade=0.5,
    ):
        """Opacity for a finer adaptive-tick level (see
        ``_view_set_adaptive_ticks``): the existing view-window range
        clipping from ``_emit_view_axis_visibility_kf``, *plus* a crossfade
        that reveals the element once the animated view has zoomed in past
        ``threshold_span`` (the span at which this level's tick density is
        the "right" amount of detail), and hides it again zooming back out
        -- the map-tile-style progressive reveal behind ``adaptive_ticks``.

        The crossfade always takes exactly ``self._adaptive_tick_fade``
        seconds of real time, the same way ``_emit_view_axis_visibility_kf``
        times its window-edge fade -- independent of fps and of how fast the
        view happens to be zooming when it crosses the threshold. (The
        previous implementation sized the fade in natural-log *span* units,
        so how long it actually took depended on zoom speed, and a hard
        opacity-floor cutoff could resolve within a single frame -- looking
        like an abrupt pop rather than a fade, especially at high fps.)
        """
        fade_s = getattr(self, "_adaptive_tick_fade", 0.4) or 0.4
        min_op = max(0.0, min(1.0, getattr(self, "_adaptive_tick_min_opacity", 0.72)))
        # Front-loads the fade-in/out ramp so it never lingers at a low,
        # "ghostly" opacity -- gamma -> 0 (min_opacity -> 1) approximates the
        # documented "hard on/off snap" while staying perfectly continuous.
        shape_gamma = max(0.08, 1.0 - min_op)
        times = sorted(set([0.0] + list(self._view_sample_times())))

        def active(t):
            if not self._view_in_range_at(val, axis, t):
                return False
            xmn, xmx, ymn, ymx = self._view_limits_at(t)
            span = (xmx - xmn) if axis == "x" else (ymx - ymn)
            return span <= threshold_span

        self._emit_boolean_crossfade_kf(
            script, prop_expr, times, in_view_fn=active,
            visible_opacity=visible_opacity,
            entrance_start=entrance_start, entrance_fade=entrance_fade,
            fade_seconds=fade_s, shape_gamma=shape_gamma,
        )

    def _emit_axis_visibility_kf(
        self,
        script,
        prop_expr,
        val,
        axis,
        visible_opacity=100,
        entrance_start=None,
        entrance_fade=0.5,
    ):
        """Tick/gridline opacity dispatcher: routes to the density-aware
        zoom-level fade (``_emit_zoom_level_visibility_kf``) when ``val`` is
        an adaptive-tick position at a level finer than the base level,
        otherwise falls back to the plain view-window range fade
        (``_emit_view_axis_visibility_kf``) used for level-0/static ticks.
        """
        level = self._adaptive_tick_level(axis, val)
        threshold = self._adaptive_tick_threshold(axis, level) if level > 0 else None
        if threshold is not None:
            self._emit_zoom_level_visibility_kf(
                script, prop_expr, val, axis, threshold,
                visible_opacity=visible_opacity,
                entrance_start=entrance_start, entrance_fade=entrance_fade or 0.5,
            )
        else:
            self._emit_view_axis_visibility_kf(
                script, prop_expr, val, axis, visible_opacity,
                entrance_start=entrance_start, entrance_fade=entrance_fade or 0.5,
            )

    def _point_in_view_at(self, x_val, y_val, t):
        """True when a 2D data point ``(x_val, y_val)`` lies inside the view
        window on *both* axes at time ``t``."""
        xmin, xmax, ymin, ymax = self._view_limits_at(t)
        xscale = self._normalize_scale(self.xscale)
        yscale = self._normalize_scale(self.yscale)
        return (
            self._view_value_in_range(x_val, xmin, xmax, xscale)
            and self._view_value_in_range(y_val, ymin, ymax, yscale)
        )

    def _emit_point_scale_kf(
        self, scale_var, x_val, y_val, entrance_start=0.0, entrance_dur=0.0,
        ease_speed=None, ease_influence=None, clip_to_view=True,
    ):
        """Scale curve (as ``[s, s, 100]`` triples) for a single point marker,
        combining the pop-in entrance animation with (optionally) hiding the
        marker whenever the animated view window no longer contains it.

        This is the scatter-marker analogue of ``_emit_view_axis_visibility_kf``,
        but drives Scale (0/100) instead of Opacity so it also hides children
        parented under the same null (image + matte + outline layers), since
        Opacity does not cascade through parenting in After Effects the way
        Scale does. Returns the list of script lines to append (may be empty
        when there's nothing to animate).
        """
        ease_speed = self.ease_speed if ease_speed is None else ease_speed
        ease_influence = self.ease_influence if ease_influence is None else ease_influence
        do_clip = bool(clip_to_view) and self._view_animated
        lines = []

        if not do_clip:
            if entrance_dur and entrance_dur > 0:
                lines.append(f"{scale_var}.setValueAtTime({entrance_start}, [0, 0, 100]);\n")
                lines.append(f"{scale_var}.setValueAtTime({entrance_start + entrance_dur}, [100, 100, 100]);\n")
                if self.easy_ease:
                    lines.append(f"applyEasyEase({scale_var}, {ease_speed}, {ease_influence});\n")
            return lines

        win_fade = getattr(self, "_view_visibility_fade", 0.35) or 0.35
        times = sorted(set([0.0] + list(self._view_sample_times())))
        in_views = [self._point_in_view_at(x_val, y_val, t) for t in times]
        use_entrance = bool(entrance_dur) and entrance_dur > 0 and in_views[0]

        kfs = []

        def add(t, s):
            kfs.append((round(float(t), 6), s))

        if in_views[0]:
            if use_entrance:
                add(0.0, 0)
                if entrance_start > 1e-6:
                    add(entrance_start, 0)
                add(entrance_start + entrance_dur, 100)
                reveal_done = entrance_start + entrance_dur
            else:
                add(0.0, 100)
                reveal_done = 0.0
        else:
            add(0.0, 0)
            reveal_done = 0.0

        for i in range(1, len(times)):
            t_prev, t = times[i - 1], times[i]
            prev, cur = in_views[i - 1], in_views[i]
            if t <= reveal_done + 1e-6:
                continue
            if prev and not cur:
                add(t_prev, 100)
                add(t_prev + win_fade, 0)
            elif not prev and cur:
                add(t, 0)
                add(t + win_fade, 100)

        cleaned = []
        for t, s in kfs:
            if cleaned and abs(cleaned[-1][0] - t) < 1e-6:
                cleaned[-1] = (t, s)
            else:
                cleaned.append((t, s))

        if not cleaned:
            return lines
        for t, s in cleaned:
            lines.append(f"{scale_var}.setValueAtTime({t}, [{s}, {s}, 100]);\n")
        lines.append(f"setKeyInterp({scale_var}, false);\n")
        if self.easy_ease:
            lines.append(f"applyEasyEase({scale_var}, {ease_speed}, {ease_influence});\n")
        return lines

    def _emit_point_move_kf(
        self, pos_var, start_pos, end_pos, entrance_start=0.0, entrance_dur=0.0,
        ease_speed=None, ease_influence=None,
    ):
        """Position keyframes sliding a point marker from ``start_pos`` to
        ``end_pos`` (both ``(x, y)`` pairs in absolute layer/comp coordinates,
        i.e. already offset by the plot center). Lets a scatter marker visibly
        "ride" the growing tip of a paired bar/stem (e.g. lollipop dots with
        ``dot_follows_stem=True``) instead of popping in at a fixed spot.
        """
        ease_speed = self.ease_speed if ease_speed is None else ease_speed
        ease_influence = self.ease_influence if ease_influence is None else ease_influence
        sx0, sy0 = start_pos
        sx1, sy1 = end_pos
        lines = []
        if entrance_dur and entrance_dur > 0:
            lines.append(f"{pos_var}.setValueAtTime({entrance_start}, [{sx0}, {sy0}]);\n")
            lines.append(f"{pos_var}.setValueAtTime({entrance_start + entrance_dur}, [{sx1}, {sy1}]);\n")
            if self.easy_ease:
                lines.append(f"applyEasyEase({pos_var}, {ease_speed}, {ease_influence});\n")
        else:
            lines.append(f"{pos_var}.setValueAtTime({entrance_start}, [{sx1}, {sy1}]);\n")
        return lines

    def _bar_entrance_schedule(self, n, total_anim, individual_duration, elem=None):
        """Shared stagger schedule for overlapping bar/stem entrances.

        Matches the ``barh`` / ``bar_graph`` JSX renderer: start times are
        spread across ``[0, total_anim - individual_duration]`` then eased.
        """
        if n <= 0:
            return [], float(individual_duration)
        individual_duration = float(individual_duration)
        if n > 1:
            pre_ease = np.linspace(
                0, max(0.0, float(total_anim) - individual_duration), n
            )
        else:
            pre_ease = [0.0]
        starts = list(self._apply_meta_ease(pre_ease, elem or {}))
        return starts, individual_duration

    def _visible_axis_origin(self, baseline, axis="x"):
        """Where a stem/dot should visually start when ``baseline`` is off-screen.

        Stems are clipped to the current axis limits, so a bar with
        ``baseline=0`` and ``xlim=(10, 38)`` grows from the y-axis at x=10,
        not from x=0. Slide-in dots must use the same visible origin.
        """
        baseline = float(baseline)
        lim = self.xlim if axis == "x" else self.ylim
        scale = self._normalize_scale(self.xscale if axis == "x" else self.yscale)
        if not lim:
            return baseline
        edge = float(lim[0])
        if scale == "log":
            if edge <= 0:
                return baseline
            return max(baseline, edge) if baseline > 0 else edge
        return max(baseline, edge)

    def _emit_point_hide_until_kf(
        self, scale_var, entrance_start=0.0, reveal_duration=0.3,
        ease_speed=None, ease_influence=None,
    ):
        """Scale keyframes keeping a marker hidden until ``entrance_start``.

        Parent-null scale cascades to image/matte/outline children, which is
        more reliable in After Effects than Opacity alone when track mattes
        are involved. After ``entrance_start``, scale eases from 0 to 100 over
        ``reveal_duration`` seconds.
        """
        ease_speed = self.ease_speed if ease_speed is None else ease_speed
        ease_influence = self.ease_influence if ease_influence is None else ease_influence
        lines = []
        entrance_start = float(entrance_start)
        reveal_duration = max(0.0, float(reveal_duration))
        end_t = entrance_start + reveal_duration
        if entrance_start > 1e-6:
            lines.append(f"{scale_var}.setValueAtTime(0, [0, 0, 100]);\n")
            lines.append(f"{scale_var}.setValueAtTime({entrance_start}, [0, 0, 100]);\n")
        else:
            lines.append(f"{scale_var}.setValueAtTime(0, [0, 0, 100]);\n")
        if reveal_duration > 1e-6:
            lines.append(f"{scale_var}.setValueAtTime({end_t}, [100, 100, 100]);\n")
            if self.easy_ease:
                lines.append(
                    f"applyEasyEase({scale_var}, {ease_speed}, {ease_influence});\n"
                )
        else:
            lines.append(f"{scale_var}.setValueAtTime({entrance_start}, [100, 100, 100]);\n")
        return lines

    @staticmethod
    def _normalize_scale(scale: str) -> str:
        """Normalize scale name to 'linear' or 'log' (matplotlib-compatible aliases)."""
        s = (scale or "linear").lower().strip()
        aliases = {"lin": "linear", "log10": "log", "lg": "log"}
        s = aliases.get(s, s)
        if s not in ("linear", "log"):
            raise ValueError(f"Unsupported scale {scale!r}; use 'linear' or 'log'")
        return s

    def set_xscale(self, scale: str):
        """
        Set the x-axis scale, like matplotlib's ``Axes.set_xscale``.

        Parameters
        ----------
        scale : str
            ``'linear'`` (default) or ``'log'`` (base 10). Log scale requires
            positive data and limits.
        """
        self.xscale = self._normalize_scale(scale)
        if getattr(self, "_xtick_labels_auto", False):
            self.set_xticks(positions=None)
        return self

    def set_yscale(self, scale: str):
        """
        Set the y-axis scale, like matplotlib's ``Axes.set_yscale``.

        Parameters
        ----------
        scale : str
            ``'linear'`` (default) or ``'log'`` (base 10).
        """
        self.yscale = self._normalize_scale(scale)
        if getattr(self, "_ytick_labels_auto", False):
            self.set_yticks(positions=None)
        return self

    def set_xaxis(self, visible: bool = True):
        """
        Show or hide the horizontal x-axis (bottom spine, x tick marks, and x tick labels).

        Parameters
        ----------
        visible : bool
            If False, the x-axis line and x ticks are omitted. Use when only the y-axis spine
            is needed, or when x tick marks would clutter the chart.
        """
        self.show_xaxis = bool(visible)
        return self

    def set_yaxis(self, visible: bool = True):
        """
        Show or hide the vertical y-axis (spine at x=0 or plot edge, y ticks, and y tick labels).

        Parameters
        ----------
        visible : bool
            If False, the y-axis line at x=0 (when in range) is omitted. Use when zero is not
            a meaningful baseline—for example scatter plots over positive x only.
        """
        self.show_yaxis = bool(visible)
        return self

    def set_xaxis_location(self, location: Union[str, float, int]):
        """
        Where to draw the horizontal axis spine (matplotlib-style).

        Parameters
        ----------
        location : str or number
            ``'bottom'`` or ``'top'`` — spine on the plot edge (border-like).
            ``'auto'`` — at ``y=0`` when zero is in range, otherwise bottom (default).
            A number — spine at that y data coordinate (e.g. ``0``).
        """
        self.xaxis_location = location
        return self

    def set_yaxis_location(self, location: Union[str, float, int]):
        """
        Where to draw the vertical axis spine (matplotlib-style).

        Parameters
        ----------
        location : str or number
            ``'left'`` or ``'right'`` — spine on the plot edge (border-like).
            ``'auto'`` — at ``x=0`` when zero is in range, otherwise left (default).
            A number — spine at that x data coordinate (e.g. ``0``).
        """
        self.yaxis_location = location
        return self

    def set_plot_frame(self, enabled: bool = True):
        """
        Draw top and right spines opposite the primary axes (full plot rectangle).

        Parameters
        ----------
        enabled : bool
            When True, closes the plot area with lines at the edges opposite the main
            x- and y-axis spines (matplotlib-style frame).
        """
        self.plot_frame = bool(enabled)
        return self

    def _resolve_xaxis_y(self, ymin_pad: float, ymax_pad: float, has_barh: bool = False) -> float:
        """Y data coordinate for the horizontal axis spine."""
        loc = self.xaxis_location
        if has_barh and (loc is None or loc == "auto"):
            y = ymin_pad
        elif loc is None or loc == "auto":
            y = 0.0 if ymin_pad <= 0 <= ymax_pad else ymin_pad
        elif isinstance(loc, str):
            key = loc.lower().strip()
            if key == "bottom":
                y = ymin_pad
            elif key == "top":
                y = ymax_pad
            else:
                raise ValueError(f"Unknown xaxis_location {loc!r}; use 'bottom', 'top', 'auto', or a number.")
        else:
            y = float(loc)
        return min(max(y, ymin_pad), ymax_pad)

    def _resolve_yaxis_x(self, xmin_pad: float, xmax_pad: float) -> float:
        """X data coordinate for the vertical axis spine."""
        loc = self.yaxis_location
        if loc is None or loc == "auto":
            x = 0.0 if xmin_pad <= 0 <= xmax_pad else xmin_pad
        elif isinstance(loc, str):
            key = loc.lower().strip()
            if key == "left":
                x = xmin_pad
            elif key == "right":
                x = xmax_pad
            else:
                raise ValueError(f"Unknown yaxis_location {loc!r}; use 'left', 'right', 'auto', or a number.")
        else:
            x = float(loc)
        return min(max(x, xmin_pad), xmax_pad)

    def _x_tick_labels_below_axis(self, x_axis_y: float, ymin_pad: float, ymax_pad: float) -> bool:
        """True if x tick labels should sit below the spine (else above)."""
        loc = self.xaxis_location
        if isinstance(loc, str) and loc.lower().strip() == "top":
            return False
        if isinstance(loc, str) and loc.lower().strip() == "bottom":
            return True
        if loc is None or loc == "auto" or isinstance(loc, str):
            if ymin_pad <= 0 <= ymax_pad:
                return True
            return x_axis_y == ymin_pad
        return float(loc) <= (ymin_pad + ymax_pad) / 2

    def _y_tick_labels_left_of_axis(self, y_axis_x: float, xmin_pad: float, xmax_pad: float) -> bool:
        """True if y tick labels should sit left of the spine (else right)."""
        loc = self.yaxis_location
        if isinstance(loc, str) and loc.lower().strip() == "right":
            return False
        if isinstance(loc, str) and loc.lower().strip() == "left":
            return True
        if loc is None or loc == "auto" or isinstance(loc, str):
            if xmin_pad <= 0 <= xmax_pad:
                return True
            return y_axis_x == xmin_pad
        return float(loc) <= (xmin_pad + xmax_pad) / 2

    def _collect_all_x(self):
        """All x-like values from plotted elements (for limit / tick computation)."""
        all_x = []
        for elem in self.elements:
            if elem["type"] in ["line", "scatter", "line_evolving", "scatter_evolving", "quiver", "quiver_evolving", "heatmap", "heatmap_evolving"]:
                all_x.extend(elem.get("x", []))
            elif elem["type"] in ["histogram", "bar_graph", "bar_stacked", "bar_evolving"]:
                all_x.extend(elem.get("bin_left", []))
                all_x.extend(elem.get("bin_right", []))
                all_x.extend(elem.get("bin_centers", []))
            elif elem["type"] in ("barh", "barh_evolving"):
                _base = elem.get("baseline")
                if _base:
                    all_x.extend([b + w for b, w in zip(_base, elem.get("widths", []))])
                    all_x.extend(_base)
                else:
                    all_x.extend(elem.get("widths", []))
                if self._normalize_scale(self.xscale) != "log":
                    all_x.append(0)
            elif elem["type"] in ("band", "errorbar"):
                all_x.extend(elem.get("x", []))
            elif elem["type"] == "refline" and elem.get("orient") == "v":
                all_x.append(elem.get("value"))
        return all_x

    def _collect_all_y(self):
        """All y-like values from plotted elements (for limit / tick computation)."""
        all_y = []
        for elem in self.elements:
            if elem["type"] in ["line", "scatter", "line_evolving", "scatter_evolving", "quiver", "quiver_evolving", "heatmap", "heatmap_evolving"]:
                all_y.extend(elem.get("y", []))
            elif elem["type"] in ["histogram", "bar_graph", "bar_stacked", "bar_evolving"]:
                _base = elem.get("baseline")
                if _base:
                    all_y.extend([b + h for b, h in zip(_base, elem.get("heights", []))])
                    all_y.extend(_base)
                else:
                    all_y.extend(elem.get("heights", []))
            elif elem["type"] in ("barh", "barh_evolving"):
                all_y.extend(elem.get("bin_bottom", []))
                all_y.extend(elem.get("bin_top", []))
                all_y.extend(elem.get("bin_centers", []))
            elif elem["type"] == "band":
                all_y.extend(elem.get("y_low", []))
                all_y.extend(elem.get("y_high", []))
            elif elem["type"] == "errorbar":
                ys = np.asarray(elem.get("y", []), dtype=float)
                yerr = elem.get("yerr")
                if yerr is not None and len(ys):
                    yerr = np.asarray(yerr, dtype=float)
                    all_y.extend(list(ys - yerr))
                    all_y.extend(list(ys + yerr))
                else:
                    all_y.extend(list(ys))
            elif elem["type"] == "refline" and elem.get("orient") == "h":
                all_y.append(elem.get("value"))
        return all_y

    def _view_xlim(self):
        """Visible x range (xlim or data extrema, without tick padding)."""
        if self.xlim:
            return self.xlim[0], self.xlim[1]
        all_x = self._collect_all_x()
        if not all_x:
            return (1.0, 10.0) if self._normalize_scale(self.xscale) == "log" else (0.0, 1.0)
        return min(all_x), max(all_x)

    def _view_ylim(self):
        """Visible y range (ylim or data extrema, without tick padding)."""
        if self.ylim:
            return self.ylim[0], self.ylim[1]
        all_y = self._collect_all_y()
        if not all_y:
            return (1.0, 10.0) if self._normalize_scale(self.yscale) == "log" else (0.0, 1.0)
        return min(all_y), max(all_y)

    def _auto_set_ticks_from_padded(self, axis: str, vmin_pad: float, vmax_pad: float, nticks: int = 7):
        """Generate ticks aligned to the grid (which is drawn across the padded range).

        Mirrors what the grid renderer does (``_nice_ticks_for_axis(xmin_pad,
        xmax_pad)``) so every grid line has a matching tick + label. Used as
        the default tick generator at render time when the caller hasn't
        called ``set_xticks``/``set_yticks`` manually.
        """
        scale_attr = "xscale" if axis == "x" else "yscale"
        scale = self._normalize_scale(getattr(self, scale_attr))
        positions = self._nice_ticks_for_axis(vmin_pad, vmax_pad, nticks, scale)
        if scale == "log":
            positions = [p for p in positions if p > 0]
        positions = self._sanitize_tick_positions(positions, scale)
        if axis == "x":
            self._xtick_labels_auto = True
            if self.percent_tick_labels:
                step = positions[1] - positions[0] if len(positions) > 1 else 1.0
                if abs(step) < 1:
                    labels = [f"{abs(pos):.1f}%" for pos in positions]
                else:
                    labels = [f"{abs(pos):.0f}%" for pos in positions]
            else:
                labels = [self._format_tick_label(pos, self.xscale) for pos in positions]
            self.xticks = list(zip(positions, labels))
        else:
            self._ytick_labels_auto = True
            labels = [self._format_tick_label(pos, self.yscale) for pos in positions]
            self.yticks = list(zip(positions, labels))

    def _filter_ticks_to_view(self, positions, vmin, vmax, scale: str):
        """Drop auto-generated tick positions outside the visible axis limits."""
        if not positions:
            return []
        scale = self._normalize_scale(scale)
        if scale == "log":
            vmin, vmax = self._sanitize_log_limits(vmin, vmax)
            return [p for p in positions if vmin <= p <= vmax]
        margin = max(abs(vmax - vmin) * 1e-9, 1e-12)
        return [p for p in positions if vmin - margin <= p <= vmax + margin]

    def add_legend(self, style='color_only', legend_pos=None):
        """
        Add a legend to the plot (auto from labels).

        Parameters:
        style (str): Legend display style. Options:
            - 'color_only': Show solid color swatches (default)
            - 'line_style': Show line samples with actual line styles (dashes, dots, etc.)
                           Useful for distinguishing lines with same color but different styles.
        legend_pos: Controls where the legend is placed. Three forms are accepted:

            - ``None`` (default): fully automatic — the placement optimizer picks the
              spot that maximises distance from data and text elements.

            - A named region string: restricts the optimizer to a sub-region of the
              plot area and picks the best spot within it. Accepted values:
              ``"top"``, ``"bottom"``, ``"left"``, ``"right"``,
              ``"top_left"``, ``"top_right"``,
              ``"bottom_left"``, ``"bottom_right"``,
              ``"center"``.
              Spaces are treated the same as underscores (e.g. ``"top left"``).

            - A 2-tuple ``(x, y)`` of data coordinates: places the legend anchor
              at exactly that position (converted from your data space). The
              optimizer is bypassed.
        """
        self.legend_style = style
        self.legend_pos = legend_pos
        return self

    def set_xticks(self, positions=None, labels=None, nticks=7):
        """
        Set X-axis tick positions and labels.
        """
        self._xtick_labels_auto = labels is None
        scale = self._normalize_scale(self.xscale)
        positive_x = []

        if positions is None:
            all_x = self._collect_all_x()
            if not all_x:
                all_x = [1, 10] if scale == "log" else [0, 1]
            positive_x = [v for v in all_x if v > 0]
            if self._view_kf:
                ux_min, ux_max, _uy_min, _uy_max = self._view_union_range()
                view_min, view_max = ux_min, ux_max
            else:
                view_min, view_max = self._view_xlim()
            if scale == "log":
                view_min, view_max = self._sanitize_log_limits(view_min, view_max, positive_x or all_x)
            if (labels is None and self._view_animated and scale == "linear"
                    and getattr(self, "_adaptive_ticks", True)):
                # Animated view + linear scale: build the map-tile-style
                # multi-density tick set instead of one fixed set sized for
                # the widest keyframed view (see `_view_set_adaptive_ticks`).
                self._view_set_adaptive_ticks("x", view_min, view_max)
                return self
            tick_min, tick_max = self._pad_range(view_min, view_max, scale)
            positions = self._nice_ticks_for_axis(tick_min, tick_max, nticks, scale)
            positions = self._filter_ticks_to_view(positions, view_min, view_max, scale)
        else:
            positions = list(positions)
            if scale == "log":
                positions = [p for p in positions if p > 0]
                if not positions:
                    raise ValueError("Log x-axis ticks must be positive; none of the given positions are > 0.")

        positions = self._sanitize_tick_positions(positions, scale)

        if labels is None:
            if self.percent_tick_labels:
                step = positions[1] - positions[0] if len(positions) > 1 else 1.0
                if abs(step) < 1:
                    labels = [f"{abs(pos):.1f}%" for pos in positions]
                else:
                    labels = [f"{abs(pos):.0f}%" for pos in positions]
            else:
                labels = [self._format_tick_label(pos, self.xscale) for pos in positions]
        else:
            labels = list(labels)
            if len(labels) != len(positions):
                raise ValueError("xtick labels must match the number of tick positions.")

        self.xticks = list(zip(positions, labels))
        return self

    def set_yticks(self, positions=None, labels=None, nticks=7):
        """
        Set Y-axis tick positions and labels.
        """
        self._ytick_labels_auto = labels is None
        scale = self._normalize_scale(self.yscale)
        positive_y = []

        if positions is None:
            all_y = self._collect_all_y()
            if not all_y:
                all_y = [1, 10] if scale == "log" else [0, 1]
            positive_y = [v for v in all_y if v > 0]
            if self._view_kf:
                _ux_min, _ux_max, uy_min, uy_max = self._view_union_range()
                view_min, view_max = uy_min, uy_max
            else:
                view_min, view_max = self._view_ylim()
            if scale == "log":
                view_min, view_max = self._sanitize_log_limits(view_min, view_max, positive_y or all_y)
            if (labels is None and self._view_animated and scale == "linear"
                    and getattr(self, "_adaptive_ticks", True)):
                # Animated view + linear scale: build the map-tile-style
                # multi-density tick set instead of one fixed set sized for
                # the widest keyframed view (see `_view_set_adaptive_ticks`).
                self._view_set_adaptive_ticks("y", view_min, view_max)
                return self
            tick_min, tick_max = self._pad_range(view_min, view_max, scale)
            positions = self._nice_ticks_for_axis(tick_min, tick_max, nticks, scale)
            positions = self._filter_ticks_to_view(positions, view_min, view_max, scale)
        else:
            positions = list(positions)
            if scale == "log":
                positions = [p for p in positions if p > 0]
                if not positions:
                    raise ValueError("Log y-axis ticks must be positive; none of the given positions are > 0.")

        positions = self._sanitize_tick_positions(positions, scale)

        if labels is None:
            labels = [self._format_tick_label(pos, self.yscale) for pos in positions]
        else:
            labels = list(labels)
            if len(labels) != len(positions):
                raise ValueError("ytick labels must match the number of tick positions.")

        self.yticks = list(zip(positions, labels))
        return self

    def grid(self, show=True, color=UNSET, alpha=UNSET, linewidth=UNSET, linestyle=UNSET, dash_size=UNSET, hide_horizontal=False, hide_vertical=False):
        """
        Enable or disable grid lines with customizable appearance.

        Styling args left unset fall back to the values established by the
        active theme / constructor (e.g. ``config.grid_color`` from
        ``themes.json``), so calling a bare ``.grid()`` keeps the theme's grid
        color instead of forcing it back to gray.

        Args:
            show (bool): Whether to show grid (default: True)
            color (str or list): Grid color. Defaults to the theme/constructor
                grid color when omitted.
            alpha (float): Grid opacity 0-1 (default: 0.3 unless changed earlier)
            linewidth (float): Grid line stroke width in pixels (default: 1.0).
            linestyle (str): Line style - 'solid', 'dashed'/'--' (default),
                           'dotted'/':', or 'dashdot'/'-.'.
            dash_size (float): Scale factor for dash lengths (default: 1.0). Only applies to non-solid linestyles.
            hide_horizontal (bool): Hide horizontal grid lines (default: False)
            hide_vertical (bool): Hide vertical grid lines (default: False)
        """
        self.show_grid = show
        if color is not UNSET:
            self.grid_color = color
        if alpha is not UNSET:
            self.grid_alpha = alpha
        if linewidth is not UNSET:
            self.grid_linewidth = float(linewidth)
        if linestyle is not UNSET:
            self.grid_linestyle = linestyle
        if dash_size is not UNSET:
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
        script.append("var comp = app.project.activeItem;\n")
        script.append("if (comp && comp instanceof CompItem) {\n")
        script.append("    while (comp.numLayers > 0) {\n")
        script.append("        comp.layer(1).remove();\n")
        script.append("    }\n")
        script.append("}\n")
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
            tell application "{config.ae_version}"
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
        # Map data to graph logical coordinates (origin at top-left)
        px = [self.width * self._data_norm(xi, xmin, xmax, self.xscale) - self.width / 2 for xi in x]
        py = [
            self.height - self.height * self._data_norm(yi, ymin, ymax, self.yscale) - self.height / 2
            for yi in y
        ]
        # Shift to graph center (origin at 0,0)
        px = [p - self.width/2 for p in px]
        py = [p - self.height/2 for p in py]
        # Map to comp coordinates
        comp_px, comp_py = zip(*[self._graph_to_comp(xg, yg) for xg, yg in zip(px, py)])
        return list(comp_px), list(comp_py)

    @staticmethod
    def _elem_legend_color(elem):
        """Resolve a single swatch color for a legend entry from an element dict."""
        color = elem.get("color")
        if isinstance(color, str) or _is_rgb_triplet(color):
            return color_to_js(color)
        gc = elem.get("gradient_colors")
        if gc:
            return color_to_js(gc[0])
        return color_to_js(config.ui_color)

    @staticmethod
    def _cubic_bezier_ease(u, influence, speed, invert=False, influence_end=None):
        """Map ``u`` in [0, 1] to an eased value in [0, 1] using a
        cubic-Bezier timing curve, mirroring an After Effects "Easy Ease".

        ``influence`` (0-100) sets how far the cushion reaches in from the start
        (the outgoing handle length); ``speed`` (>= 0) tilts the handles so the
        sweep ramps faster through the middle. ``speed == 0`` with ``influence ==
        33`` reproduces the classic ease-in/ease-out cushion.

        ``influence_end`` (0-100, optional) sets the incoming handle length at the
        end independently of ``influence``. When ``None`` (default) the curve is
        symmetric and both handles use ``influence`` — i.e. start influence ==
        end influence. Pass it to get an asymmetric ease (e.g. a soft launch but
        an abrupt arrival, or vice versa).

        ``invert=True`` returns the functional inverse of that curve (the same
        Bezier with its x/y control coordinates swapped). This is what spaces a
        *sequence* of element entrances so the reveal velocity — not the in-point
        positions — follows the ease: big gaps at the ends (slow), small gaps in
        the middle (fast).
        """
        try:
            u = float(u)
        except (TypeError, ValueError):
            return 0.0
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        inf_start = min(max(float(influence) / 100.0, 0.0), 0.95)
        if influence_end is None:
            inf_end = inf_start
        else:
            inf_end = min(max(float(influence_end) / 100.0, 0.0), 0.95)
        sp = max(float(speed), 0.0) / 100.0
        # Control points: horizontal extent from the start/end influence, a small
        # vertical lift from speed (speed 0 -> flat handles -> maximal cushion).
        x1, y1 = inf_start, inf_start * sp
        x2, y2 = 1.0 - inf_end, 1.0 - inf_end * sp
        if invert:
            # Reflect the curve across the diagonal -> functional inverse.
            x1, y1, x2, y2 = y1, x1, y2, x2

        def bezier(t, p1, p2):
            mt = 1.0 - t
            return 3.0 * mt * mt * t * p1 + 3.0 * mt * t * t * p2 + t * t * t

        # Invert x(t) = u with a bounded binary search (curve is monotonic in t).
        lo, hi, t = 0.0, 1.0, u
        for _ in range(40):
            x = bezier(t, x1, x2)
            if abs(x - u) < 1e-6:
                break
            if x < u:
                lo = t
            else:
                hi = t
            t = 0.5 * (lo + hi)
        return bezier(t, y1, y2)

    def _apply_meta_ease(self, start_times, elem=None):
        """Remap a sequence of element start times so the *reveal velocity* eases
        in and out: the sweep starts slow (in-points spread far apart), speeds up
        through the middle (in-points bunch together), then slows again. This uses
        the inverse of the ease curve, because perceived sweep speed is inversely
        proportional to the gap between consecutive in-points. Preserves ordering
        and total span; a no-op when ``meta_easy_ease`` is disabled or there is
        nothing to stagger.

        Per-element overrides (``meta_easy_ease`` / ``meta_ease_speed`` /
        ``meta_ease_influence`` keys on ``elem``) take precedence over the
        instance-level defaults when present."""
        elem = elem or {}
        enabled = elem.get("meta_easy_ease")
        if enabled is None:
            enabled = getattr(self, "meta_easy_ease", False)
        if not enabled:
            return start_times
        st = [float(v) for v in start_times]
        if len(st) < 2:
            return st
        span = max(st)
        if span <= 1e-9:
            return st
        influence = elem.get("meta_ease_influence")
        if influence is None:
            influence = self.meta_ease_influence
        speed = elem.get("meta_ease_speed")
        if speed is None:
            speed = self.meta_ease_speed
        return [
            span * self._cubic_bezier_ease(v / span, influence, speed, invert=True)
            for v in st
        ]

    def _data_to_shape(self, x, y, xmin, xmax, ymin, ymax):
        """
        Convert data (x, y) to graph-local coordinates (centered at 0,0, width/height).
        """
        nx = self._data_norm(x, xmin, xmax, self.xscale)
        ny = self._data_norm(y, ymin, ymax, self.yscale)
        sx = self.width * nx - self.width / 2
        sy = self.height - self.height * ny - self.height / 2
        return sx, sy

    def _x_tick_shape_coords(self, pos, x_axis_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad, half_len: float = 8.0):
        """X-axis tick segment in shape space (constant pixel length; safe for log scales)."""
        tick_xs, tick_yc = self._data_to_shape(pos, x_axis_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
        return tick_xs, tick_yc - half_len, tick_xs, tick_yc + half_len

    def _y_tick_shape_coords(self, y_axis_x, pos, xmin_pad, xmax_pad, ymin_pad, ymax_pad, half_len: float = 8.0):
        """Y-axis tick segment in shape space (constant pixel length; safe for log scales)."""
        tick_xc, tick_ys = self._data_to_shape(y_axis_x, pos, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
        return tick_xc - half_len, tick_ys, tick_xc + half_len, tick_ys

    def _tick_label_gap(self) -> float:
        """Pixels between tick mark end and tick label (scales with font)."""
        return max(4.0, 6.0 * self.font_scale)

    def _compute_xlabel_shape_y(
        self,
        xmin_pad: float,
        xmax_pad: float,
        ymin_pad: float,
        ymax_pad: float,
        x_axis_y: float,
    ) -> tuple[float, bool]:
        """Shape-local Y for the x-axis title and whether its anchor is at the text top.

        Returns (y, anchor_at_top). Tick labels are accounted for so the title sits
        just outside them, not at a fixed offset from the plot frame edge.
        """
        tick_fs = int(27 * self.font_scale)
        tick_text_h = tick_fs * 1.12
        xlabel_gap = max(6.0, 10.0 * self.font_scale)
        tick_label_gap = self._tick_label_gap()
        x_labels_below = self._x_tick_labels_below_axis(x_axis_y, ymin_pad, ymax_pad)

        if self.show_xaxis and self.show_tick_labels and self.xticks:
            extreme = None
            for pos, label in self.xticks:
                if self._normalize_scale(self.xscale) == "log" and pos <= 0:
                    continue
                if not self._xtick_labels_auto and not str(label).strip():
                    continue
                _, tick_ys0, _, tick_ys1 = self._x_tick_shape_coords(
                    pos, x_axis_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad
                )
                if x_labels_below:
                    label_edge = max(tick_ys0, tick_ys1) + tick_label_gap + tick_text_h
                    extreme = label_edge if extreme is None else max(extreme, label_edge)
                else:
                    label_edge = min(tick_ys0, tick_ys1) - tick_label_gap - tick_text_h
                    extreme = label_edge if extreme is None else min(extreme, label_edge)
            if extreme is not None:
                if x_labels_below:
                    return extreme + xlabel_gap, True
                return extreme - xlabel_gap, False

        _, axis_sy = self._data_to_shape(
            (xmin_pad + xmax_pad) / 2.0,
            x_axis_y,
            xmin_pad,
            xmax_pad,
            ymin_pad,
            ymax_pad,
        )
        if x_labels_below:
            return axis_sy + tick_label_gap + tick_text_h + xlabel_gap, True
        return axis_sy - tick_label_gap - tick_text_h - xlabel_gap, False

    def _compute_ylabel_shape_x(
        self,
        xmin_pad: float,
        xmax_pad: float,
        ymin_pad: float,
        ymax_pad: float,
        y_axis_x: float,
    ) -> float:
        """Shape-local X for the y-axis title (left of tick labels when present).

        Tick labels are accounted for so the rotated title sits just outside
        them, not at a fixed offset from the plot frame edge.
        """
        ylabel_fs = int(41 * self.font_scale)
        ylabel_gap = max(10.0, 14.0 * self.font_scale)
        ylabel_half_w = ylabel_fs * 0.55
        tick_fs = int(27 * self.font_scale)
        char_w = tick_fs * 0.58
        tick_label_gap = self._tick_label_gap()
        y_labels_left = self._y_tick_labels_left_of_axis(y_axis_x, xmin_pad, xmax_pad)

        if self.show_yaxis and self.show_tick_labels and self.yticks:
            extreme = None
            for pos, label in self.yticks:
                if self._normalize_scale(self.yscale) == "log" and pos <= 0:
                    continue
                if self._ytick_labels_auto:
                    label = self._format_tick_label(pos, self.yscale)
                if not str(label).strip():
                    continue
                tick_xs0, _tick_ys, tick_xs1, _u = self._y_tick_shape_coords(
                    y_axis_x, pos, xmin_pad, xmax_pad, ymin_pad, ymax_pad
                )
                est_w = len(str(label)) * char_w
                if y_labels_left:
                    label_edge = min(tick_xs0, tick_xs1) - tick_label_gap - est_w
                    extreme = label_edge if extreme is None else min(extreme, label_edge)
                else:
                    label_edge = max(tick_xs0, tick_xs1) + tick_label_gap + est_w
                    extreme = label_edge if extreme is None else max(extreme, label_edge)
            if extreme is not None:
                if y_labels_left:
                    return extreme - ylabel_gap - ylabel_half_w
                return extreme + ylabel_gap + ylabel_half_w

        yax_sx, _ = self._data_to_shape(
            y_axis_x,
            (ymin_pad + ymax_pad) / 2.0,
            xmin_pad,
            xmax_pad,
            ymin_pad,
            ymax_pad,
        )
        if y_labels_left:
            return yax_sx - tick_label_gap - ylabel_gap - ylabel_half_w
        return yax_sx + tick_label_gap + ylabel_gap + ylabel_half_w

    def _data_norm(self, value, vmin, vmax, scale: str):
        """Map a data value to [0, 1] along an axis (linear or log10)."""
        scale = self._normalize_scale(scale)
        if vmax == vmin:
            return 0.5
        if scale == "log":
            if value <= 0 or vmin <= 0 or vmax <= 0:
                raise ValueError(
                    f"Log scale requires positive values (got value={value}, range=({vmin}, {vmax}))"
                )
            log_v = math.log10(value)
            log_min = math.log10(vmin)
            log_max = math.log10(vmax)
            return (log_v - log_min) / (log_max - log_min)
        return (value - vmin) / (vmax - vmin)

    def _effective_text(self):
        """Resolve auto axis labels + title from element column names.

        Returns ``(xlabel, ylabel, title)``. Honors any explicit value on
        ``self``. When ``config.auto_axis_titles`` is True and a label is
        unset, falls back to the first element's captured pandas column
        name (``x_name``/``y_name``). Title falls back to
        ``config.auto_title_template`` only when both effective labels
        resolve to non-empty strings and ``config.auto_title`` is True.
        """
        first_x_name = None
        first_y_name = None
        for elem in self.elements:
            if first_x_name is None and elem.get("x_name"):
                first_x_name = elem.get("x_name")
            if first_y_name is None and elem.get("y_name"):
                first_y_name = elem.get("y_name")
            if first_x_name is not None and first_y_name is not None:
                break

        xlabel = self.xlabel
        ylabel = self.ylabel
        if getattr(config, "auto_axis_titles", True):
            if not xlabel and first_x_name is not None:
                xlabel = str(first_x_name)
            if not ylabel and first_y_name is not None:
                ylabel = str(first_y_name)

        title = self.title
        if (
            not title
            and getattr(config, "auto_title", True)
            and xlabel
            and ylabel
        ):
            template = getattr(config, "auto_title_template", "{x} vs {y}")
            try:
                title = template.format(x=xlabel, y=ylabel)
            except Exception:
                title = f"{xlabel} vs {ylabel}"

        return xlabel, ylabel, title

    @staticmethod
    def _color_to_rgb01(color):
        """Resolve a color name or RGB triplet to a [r, g, b] list in 0-1 floats."""
        if isinstance(color, str):
            # Support CSS hex strings: #RGB, #RRGGBB, #RGBA, #RRGGBBAA
            stripped = color.strip()
            if stripped.startswith("#"):
                h = stripped.lstrip("#")
                if len(h) in (3, 4):   # shorthand — expand each nibble
                    h = "".join(c * 2 for c in h[:3])
                if len(h) >= 6:
                    r = int(h[0:2], 16) / 255.0
                    g = int(h[2:4], 16) / 255.0
                    b = int(h[4:6], 16) / 255.0
                    return [r, g, b]
            rgb = COLOR_NAMES.get(stripped.lower())
            if rgb is None:
                raise ValueError(f"Unknown color name: {color}")
            return [float(v) for v in rgb]
        if isinstance(color, (list, tuple, np.ndarray)) and len(color) == 3:
            rgb = [float(v) for v in color]
            if max(rgb) > 1:
                rgb = [v / 255.0 for v in rgb]
            return rgb
        raise ValueError("Gradient endpoint must be a color name or 3-value RGB list.")

    @classmethod
    def _darken_color(cls, color, factor=0.55):
        """Return a darker RGB triplet for marker outlines."""
        rgb = cls._color_to_rgb01(color)
        return [max(0.0, min(1.0, float(v) * factor)) for v in rgb]

    def _resolve_bar_gradient(self, color, c, gradient, n_bars, discrete=None):
        """Resolve color / c= / gradient= inputs for histogram, bar_graph, and barh.

        Mirrors the scatter gradient API so the three bar-like plots accept the
        same arguments:

        - ``c=values``: per-bar numeric data mapped to a color gradient.
        - ``gradient=(low, high)``: override the endpoint colors. Either both
          named colors / RGB triplets.
        - ``gradient=values``: shorthand for ``c=values``.
        - ``color=...``: solid fill color (no gradient).
        - Nothing passed: default to a positional gradient between
          ``config.gradient_low`` and ``config.gradient_high`` (the theme's
          ``object_color_2`` / ``object_color_1`` pair). With one bar, the
          high endpoint is used as a solid color.

        Returns a dict of element fields. ``gradient_colors`` (per-bar RGB
        floats) is set whenever a gradient is applied; ``gradient_data``,
        ``gradient_vmin/vmax``, and ``gradient_name`` are set only for
        value-based gradients (so the colorbar in ``_generate_cmap_jsx`` can
        pick them up).
        """
        default_endpoints = (config.gradient_low, config.gradient_high)
        endpoints = default_endpoints
        gradient_data = None

        if gradient is not None:
            if (
                isinstance(gradient, (tuple, list)) and len(gradient) == 2
                and (isinstance(gradient[0], str) or _is_rgb_triplet(gradient[0]))
                and (isinstance(gradient[1], str) or _is_rgb_triplet(gradient[1]))
            ):
                endpoints = (gradient[0], gradient[1])
            else:
                gradient_data = gradient
        if c is not None:
            gradient_data = c

        c_name = None
        if pd is not None and isinstance(gradient_data, pd.Series):
            c_name = getattr(gradient_data, "name", None)
            gradient_data = gradient_data.values

        user_passed_color = color is not UNSET

        out = {
            "color": color if user_passed_color else endpoints[1],
            "gradient_colors": None,
            "gradient_data": None,
            "gradient_low": None,
            "gradient_high": None,
            "gradient_vmin": None,
            "gradient_vmax": None,
            "gradient_name": c_name,
            "gradient_discrete": False,
            "gradient_levels": None,
        }

        if gradient_data is not None:
            gradient_data = list(gradient_data)
            if len(gradient_data) != n_bars:
                raise ValueError(
                    f"Gradient data length ({len(gradient_data)}) must match number of bars ({n_bars})."
                )
            is_discrete, levels = self._resolve_discrete_levels(gradient_data, discrete)
            out["gradient_data"] = gradient_data
            out["gradient_low"] = endpoints[0]
            out["gradient_high"] = endpoints[1]
            out["gradient_discrete"] = is_discrete
            out["gradient_levels"] = levels
            if is_discrete:
                snapped = [
                    (min(levels, key=lambda L: abs(L - v)) if v == v else v)
                    for v in gradient_data
                ]
                out["gradient_colors"] = self._gradient_colors(
                    snapped, endpoints[0], endpoints[1],
                    vmin=levels[0], vmax=levels[-1],
                )
                out["gradient_vmin"] = float(levels[0])
                out["gradient_vmax"] = float(levels[-1])
            else:
                out["gradient_colors"] = self._gradient_colors(
                    gradient_data, endpoints[0], endpoints[1]
                )
                arr = np.asarray(gradient_data, dtype=float)
                if arr.size:
                    out["gradient_vmin"] = float(np.nanmin(arr))
                    out["gradient_vmax"] = float(np.nanmax(arr))
            return out

        if not user_passed_color and n_bars > 1:
            # No color, no data: positional gradient between the two object colors.
            out["gradient_colors"] = self._gradient_colors(
                list(range(n_bars)), endpoints[0], endpoints[1]
            )

        return out

    def _resolve_discrete_levels(self, gradient_data, discrete):
        """Decide whether a value-based gradient should render as discrete bands.

        Returns ``(is_discrete, levels)`` where ``levels`` is the inclusive
        integer range ``[round(min), round(max)]`` when discrete, else ``None``.

        - ``discrete=True``: force discrete (as long as a >=2 level range exists).
        - ``discrete=False``: force a smooth gradient.
        - ``discrete=None`` (default): auto — discrete when every finite value
          is an integer and the number of integer levels is in
          ``[2, config.cmap_discrete_max_levels]``.
        """
        arr = np.asarray(list(gradient_data), dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return False, None
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        lo = int(round(vmin))
        hi = int(round(vmax))
        n_levels = hi - lo + 1
        if n_levels < 2:
            return False, None
        if discrete is False:
            return False, None
        if discrete is True:
            return True, list(range(lo, hi + 1))
        # Auto-detect: all values integral and a readable number of levels.
        all_integral = bool(np.all(np.equal(np.mod(finite, 1.0), 0.0)))
        cap = int(getattr(config, "cmap_discrete_max_levels", 20))
        if all_integral and n_levels <= cap:
            return True, list(range(lo, hi + 1))
        return False, None

    def _gradient_colors(self, data, low, high, vmin=None, vmax=None):
        """Map a 1-D numeric array to per-point RGB colors between two endpoints.

        Values are normalized to [0, 1] using ``vmin``/``vmax`` (defaults to
        the data range, ignoring NaNs) and then linearly interpolated in RGB
        space between ``low`` and ``high``. Returns a list of [r, g, b] lists
        in 0-1 floats, one per input value (suitable for the existing
        per-point color path).
        """
        arr = np.asarray(data, dtype=float)
        if vmin is None:
            vmin = float(np.nanmin(arr)) if arr.size else 0.0
        if vmax is None:
            vmax = float(np.nanmax(arr)) if arr.size else 1.0
        lo = np.array(self._color_to_rgb01(low), dtype=float)
        hi = np.array(self._color_to_rgb01(high), dtype=float)
        if vmax == vmin:
            t = np.full_like(arr, 0.5)
        else:
            t = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        nan_mask = np.isnan(t)
        if nan_mask.any():
            t = np.where(nan_mask, 0.5, t)
        return [(lo + ti * (hi - lo)).tolist() for ti in t]

    def _auto_pad_limits(self, vmin, vmax, user_lim, scale: str):
        """Return (lo, hi) for a data axis, padding to `config.lim_factor` when auto.

        Honors caller-set ``user_lim`` exactly. Otherwise, when
        ``config.auto_limits`` is True, expands the range so the total span
        becomes ``config.lim_factor`` (e.g. 1.2x). For log axes the expansion
        is applied multiplicatively in log space so each side grows by the
        same factor in dex.
        """
        if user_lim is not None:
            return vmin, vmax
        if not getattr(config, "auto_limits", True):
            return vmin, vmax
        factor = float(getattr(config, "lim_factor", 1.2))
        if factor <= 1.0:
            return vmin, vmax
        if self._normalize_scale(scale) == "log":
            if vmin <= 0 or vmax <= 0:
                return vmin, vmax
            log_min = math.log10(vmin)
            log_max = math.log10(vmax)
            log_span = log_max - log_min
            if log_span == 0:
                pad = math.log10(factor) / 2.0
            else:
                pad = log_span * (factor - 1.0) / 2.0
            return 10 ** (log_min - pad), 10 ** (log_max + pad)
        span = vmax - vmin
        if span == 0:
            pad = max(abs(vmin), 1.0) * (factor - 1.0) / 2.0
        else:
            pad = span * (factor - 1.0) / 2.0
        return vmin - pad, vmax + pad

    def _sanitize_log_limits(self, vmin, vmax, values=None):
        """Ensure log axis limits are positive and usable for tick generation."""
        vals = [float(v) for v in (values or []) if v is not None and float(v) > 0]
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if vmax <= 0:
            vmax = max(vals) if vals else 1.0
        if vmin <= 0:
            vmin = min(vals) if vals else vmax / 1000.0
        if vmin == vmax:
            vmin *= 0.1
            vmax *= 10.0
        return vmin, vmax

    @staticmethod
    def _sanitize_tick_positions(positions, scale: str):
        """Drop invalid tick positions (non-positive on log axes; dedupe sorted)."""
        scale = AEGraph._normalize_scale(scale)
        if not positions:
            return []
        if scale == "log":
            positions = [p for p in positions if p > 0]
        # stable unique sort
        seen = set()
        out = []
        for p in sorted(positions):
            key = round(float(p), 12)
            if key not in seen:
                seen.add(key)
                out.append(float(p))
        return out

    def _pad_range(self, vmin, vmax, scale: str, padding: float = 0.1):
        """Pad axis limits (linear additive or log multiplicative in log-space)."""
        scale = self._normalize_scale(scale)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if scale == "log":
            vmin, vmax = self._sanitize_log_limits(vmin, vmax)
            log_min, log_max = math.log10(vmin), math.log10(vmax)
            span = log_max - log_min if log_max > log_min else 1.0
            return (10 ** (log_min - span * padding), 10 ** (log_max + span * padding))
        span = vmax - vmin
        if span == 0:
            span = abs(vmin) * 0.1 or 1.0
        return (vmin - span * padding, vmax + span * padding)

    def _nice_log_ticks(self, vmin, vmax, nticks=7):
        """Major ticks at powers of ten (1, 10, 100, 1000, …) inside [vmin, vmax]."""
        vmin, vmax = self._sanitize_log_limits(vmin, vmax)
        log_min = math.floor(math.log10(vmin))
        log_max = math.ceil(math.log10(vmax))
        ticks = []
        for exp in range(int(log_min), int(log_max) + 1):
            tick = 10.0 ** exp
            if tick >= vmin * (1 - 1e-9) and tick <= vmax * (1 + 1e-9):
                ticks.append(tick)
        if not ticks:
            ticks = [10.0 ** log_min, 10.0 ** log_max]
        return ticks

    def _nice_ticks_for_axis(self, vmin, vmax, nticks=7, scale: str = "linear"):
        scale = self._normalize_scale(scale)
        if scale == "log":
            return self._nice_log_ticks(vmin, vmax, nticks)
        return self._nice_ticks(vmin, vmax, nticks)

    @staticmethod
    def _superscript(exp: int) -> str:
        """Render an integer exponent using Unicode superscript glyphs."""
        supers = {
            "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
            "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077",
            "8": "\u2078", "9": "\u2079", "-": "\u207b",
        }
        return "".join(supers.get(ch, ch) for ch in str(exp))

    def _format_tick_label(self, value, scale: str) -> str:
        """Format tick label text (decade labels on log axes: 1, 10, 100, …).

        On log axes, decade labels whose exponent magnitude is at least
        ``self.log_power_label_threshold`` are written compactly as ``10ⁿ``
        with a real Unicode superscript exponent (e.g. ``10`` -> ``10¹``,
        ``1000`` -> ``10³``, ``0.0001`` -> ``10⁻⁴``). With the default
        threshold of 1 every decade uses this power notation (1 stays ``1``);
        raise the threshold to keep small decades spelled out.
        """
        scale = AEGraph._normalize_scale(scale)
        if scale == "log":
            if value <= 0:
                return ""
            log_v = math.log10(value)
            exp = round(log_v)
            if abs(log_v - exp) < 1e-6:
                threshold = getattr(self, "log_power_label_threshold", None)
                if threshold is not None and abs(exp) >= threshold:
                    return f"10{self._superscript(exp)}"
                if exp == 0:
                    return "1"
                if exp == 1:
                    return "10"
                if exp > 1:
                    return "1" + "0" * exp
                # 0.1, 0.01, …
                return f"{10.0 ** exp:g}"
            return f"{value:g}"
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        return str(value)

    @staticmethod
    def _nice_step(raw_step):
        """Round a raw target spacing to a 'nice' 1-2-5-10 x 10^k step."""
        if raw_step <= 0:
            return 1.0
        mag = 10 ** math.floor(math.log10(abs(raw_step)))
        norm = raw_step / mag
        if norm < 1.5:
            return 1 * mag
        elif norm < 3:
            return 2 * mag
        elif norm < 7:
            return 5 * mag
        return 10 * mag

    def _nice_ticks(self, vmin, vmax, nticks=7):
        """Generate nice tick positions between vmin and vmax."""
        if vmin == vmax:
            return [vmin]
        step = self._nice_step((vmax - vmin) / max(1, nticks - 1))
        tick_start = math.ceil(vmin / step) * step
        tick_end = math.floor(vmax / step) * step
        ticks = []
        v = tick_start
        while v <= tick_end + 1e-8:
            ticks.append(round(v, 10))
            v += step
        return ticks

    @staticmethod
    def _ladder_step_snap(raw_step):
        """Snap a raw target spacing to the nearest member of the nested
        {1, 5} x 10^k 'ladder' used by adaptive (zoom-density) ticks.

        Unlike ``_nice_step`` (which also allows a "2" member), this ladder
        deliberately sticks to {1, 5} x 10^k so that dividing a step by 2
        then by 5 (alternating) always lands on the next member down --
        every finer level's tick set is then a strict superset of every
        coarser one, so ticks never need to jump between unrelated
        positions as they fade in/out across zoom levels.
        """
        if raw_step <= 0:
            return 1.0
        mag = 10 ** math.floor(math.log10(abs(raw_step)))
        norm = raw_step / mag
        if norm < math.sqrt(1 * 5):
            return 1.0 * mag
        elif norm < math.sqrt(5 * 10):
            return 5.0 * mag
        return 10.0 * mag

    @staticmethod
    def _ladder_next_finer(step):
        """The next-finer ladder member below ``step`` (see ``_ladder_step_snap``)."""
        mag = 10 ** math.floor(math.log10(step) + 1e-9)
        norm = step / mag
        if norm < 3:  # "1" (or "10" of the decade above) member -> halve
            return step / 2.0
        return step / 5.0  # "5" member -> divide by 5

    @staticmethod
    def _ticks_for_step(vmin, vmax, step, limit=100_000):
        """Nice multiples of ``step`` within ``[vmin, vmax]``."""
        if step <= 0 or vmax < vmin:
            return []
        start = math.ceil(vmin / step - 1e-9) * step
        ticks = []
        v = start
        n = 0
        while v <= vmax + abs(step) * 1e-6 and n < limit:
            ticks.append(round(v, 10))
            v += step
            n += 1
        return ticks

    def _find_gradient_scatter(self):
        """Return the first element rendered with a value-based gradient, else None.

        Now considers scatter, histogram, bar_graph, and barh elements — any
        element that recorded ``gradient_low``/``gradient_high`` endpoints and
        a ``gradient_vmin``/``gradient_vmax`` range qualifies for the colorbar.
        Purely positional gradients (no underlying data, no vmin/vmax) are
        skipped so the cmap doesn't render a meaningless scale.
        """
        for elem in self.elements:
            if elem.get("type") not in ("scatter", "histogram", "bar_graph", "barh",
                                        "barh_evolving", "heatmap", "heatmap_evolving"):
                continue
            if elem.get("gradient_low") is None or elem.get("gradient_high") is None:
                continue
            if elem.get("gradient_vmin") is None or elem.get("gradient_vmax") is None:
                continue
            return elem
        return None

    def _generate_cmap_jsx(self, script, center_x, center_y, ANIM_DURATION):
        """Emit a colorbar on the right when a scatter has a gradient.

        Builds a ``cmapNull`` (parented to ``PlotAnchor``) and stacks:
        gradient strip (N color slices), tick marks, tick labels, and a
        rotated axis label (the gradient column name). Every cmap asset is
        parented to ``cmapNull`` so it can be moved/animated as a group.
        """
        if not getattr(config, "show_cmap", True):
            return
        elem = self._find_gradient_scatter()
        if elem is None:
            return

        vmin = float(elem["gradient_vmin"])
        vmax = float(elem["gradient_vmax"])
        if vmin == vmax:
            vmax = vmin + 1.0  # avoid zero-span
        low = elem["gradient_low"]
        high = elem["gradient_high"]
        name = elem.get("gradient_name")
        low_rgb = self._color_to_rgb01(low)
        high_rgb = self._color_to_rgb01(high)

        # Discrete colorbars draw one solid band per integer level instead of a
        # smooth gradient, with a tick centered on each band.
        discrete = bool(elem.get("gradient_discrete"))
        levels = elem.get("gradient_levels") if discrete else None
        if discrete and (not levels or len(levels) < 2):
            discrete = False
            levels = None

        strip_w = float(getattr(config, "cmap_width", 32))
        gap = float(getattr(config, "cmap_gap", 80))
        n_ticks = max(2, int(getattr(config, "cmap_tick_count", 6)))
        if discrete:
            steps = len(levels)  # one band per integer level
        else:
            steps = max(2, int(getattr(config, "cmap_steps", 64)))

        # Geometry in comp coords. cmapNull sits at the strip center.
        strip_left = center_x + self.width / 2 + gap
        strip_top = center_y - self.height / 2
        strip_bottom = center_y + self.height / 2
        strip_height = strip_bottom - strip_top
        strip_cx = strip_left + strip_w / 2
        strip_cy = (strip_top + strip_bottom) / 2

        ui_color_js = color_to_js(self.ui_color)
        slice_h = strip_height / steps

        # cmapNull (parented to PlotAnchor; everything else parents to cmapNull).
        script.append("var cmapNull = comp.layers.addNull();\n")
        script.append("cmapNull.name = 'cmapNull';\n")
        script.append(
            f"cmapNull.property('Transform').property('Position').setValue([{strip_cx}, {strip_cy}]);\n"
        )
        script.append("cmapNull.parent = PlotAnchor;\n")

        # Gradient strip: one shape layer containing N color slices.
        script.append("var cmapStrip = comp.layers.addShape();\n")
        script.append("cmapStrip.name = 'cmapStrip';\n")
        script.append(
            f"cmapStrip.property('Transform').property('Position').setValue([{strip_cx}, {strip_cy}]);\n"
        )
        script.append("cmapStrip.parent = cmapNull;\n")
        script.append("var cmapStripContents = cmapStrip.property('ADBE Root Vectors Group');\n")
        for i in range(steps):
            # Slice i covers y in shape space from (top + i*slice_h) to (top + (i+1)*slice_h).
            # i=0 is the top slice (vmax / high color). The shape layer's anchor is its
            # own center (strip_cy), so each rect position is offset relative to that.
            slice_offset_y = -strip_height / 2 + slice_h * (i + 0.5)
            if discrete:
                # Band i from the top maps to level index (steps-1-i); color it at
                # the level's normalized position so points and bands match exactly.
                level_idx = steps - 1 - i
                t = level_idx / (steps - 1) if steps > 1 else 0.5
            else:
                # Color stop t: top slice (i=0) -> high (t=1); bottom (i=steps-1) -> low (t=0).
                t = 1.0 - (i + 0.5) / steps
            r = low_rgb[0] + t * (high_rgb[0] - low_rgb[0])
            g = low_rgb[1] + t * (high_rgb[1] - low_rgb[1])
            b = low_rgb[2] + t * (high_rgb[2] - low_rgb[2])
            script.append(
                f"var cmapGroup{i} = cmapStripContents.addProperty('ADBE Vector Group');\n"
            )
            script.append(
                f"var cmapGroupContents{i} = cmapGroup{i}.property('ADBE Vectors Group');\n"
            )
            script.append(
                f"var cmapRect{i} = cmapGroupContents{i}.addProperty('ADBE Vector Shape - Rect');\n"
            )
            # Slight 1px overlap removes hairline seams between slices when AE rasterizes.
            script.append(
                f"cmapRect{i}.property('ADBE Vector Rect Size').setValue([{strip_w}, {slice_h + 1.0}]);\n"
            )
            script.append(
                f"cmapRect{i}.property('ADBE Vector Rect Position').setValue([0, {slice_offset_y}]);\n"
            )
            script.append(
                f"var cmapFill{i} = cmapGroupContents{i}.addProperty('ADBE Vector Graphic - Fill');\n"
            )
            script.append(
                f"cmapFill{i}.property('ADBE Vector Fill Color').setValue([{r}, {g}, {b}]);\n"
            )

        # Strip border (thin outline so it reads against any background).
        script.append(
            "var cmapBorder = cmapStripContents.addProperty('ADBE Vector Group');\n"
        )
        script.append(
            "var cmapBorderContents = cmapBorder.property('ADBE Vectors Group');\n"
        )
        script.append(
            "var cmapBorderRect = cmapBorderContents.addProperty('ADBE Vector Shape - Rect');\n"
        )
        script.append(
            f"cmapBorderRect.property('ADBE Vector Rect Size').setValue([{strip_w}, {strip_height}]);\n"
        )
        script.append(
            "cmapBorderRect.property('ADBE Vector Rect Position').setValue([0, 0]);\n"
        )
        script.append(
            "var cmapBorderStroke = cmapBorderContents.addProperty('ADBE Vector Graphic - Stroke');\n"
        )
        script.append(
            f"cmapBorderStroke.property('ADBE Vector Stroke Color').setValue({ui_color_js});\n"
        )
        # Match plot frame/spine stroke width so the colorbar reads as part of the chart frame.
        script.append("cmapBorderStroke.property('ADBE Vector Stroke Width').setValue(3);\n")

        # Strip fade-in.
        if self.animate_opacity:
            script.append(
                "cmapStrip.property('Transform').property('Opacity').setValueAtTime(0, 0);\n"
            )
            script.append(
                f"cmapStrip.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 0.6}, 100);\n"
            )
            if self.easy_ease:
                script.append(
                    f"applyEasyEase(cmapStrip.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n"
                )
        else:
            script.append(
                "cmapStrip.property('Transform').property('Opacity').setValue(100);\n"
            )

        # Ticks + tick labels. Each entry is (t_from_bottom, label_text).
        if discrete:
            # One tick centered on each band, labeled with the integer level.
            n_lv = len(levels)
            tick_specs = [((k + 0.5) / n_lv, str(int(levels[k]))) for k in range(n_lv)]
        else:
            tick_positions = self._nice_ticks(vmin, vmax, n_ticks)
            tick_positions = self._sanitize_tick_positions(tick_positions, "linear")
            tick_specs = []
            for val in tick_positions:
                t = 0.5 if vmax == vmin else (val - vmin) / (vmax - vmin)
                t = max(0.0, min(1.0, t))
                tick_specs.append((t, self._format_tick_label(val, "linear")))
        tick_len = 8.0  # half-length on each side of the strip's right edge
        tick_label_gap = max(6.0, 8.0 * self.font_scale)
        tick_font = int(24 * self.font_scale)
        max_label_chars = 1
        for ti, (t, label_text) in enumerate(tick_specs):
            t = max(0.0, min(1.0, t))
            # Top of strip = vmax (t=1); bottom = vmin (t=0). In comp coords y grows down.
            tick_y = strip_bottom - t * strip_height
            tick_x0 = strip_left + strip_w  # at right edge of strip
            tick_x1 = tick_x0 + tick_len
            max_label_chars = max(max_label_chars, len(str(label_text)))

            script.append(f"var cmapTick{ti} = comp.layers.addShape();\n")
            script.append(f"cmapTick{ti}.name = 'cmapTick_{ti}';\n")
            script.append(
                f"cmapTick{ti}.property('Transform').property('Position').setValue([{strip_cx}, {strip_cy}]);\n"
            )
            script.append(f"cmapTick{ti}.parent = cmapNull;\n")
            script.append(
                f"var cmapTickContents{ti} = cmapTick{ti}.property('ADBE Root Vectors Group');\n"
            )
            script.append(
                f"var cmapTickPathG{ti} = cmapTickContents{ti}.addProperty('ADBE Vector Shape - Group');\n"
            )
            script.append(
                f"var cmapTickPath{ti} = cmapTickPathG{ti}.property('ADBE Vector Shape');\n"
            )
            script.append(f"var cmapTickShape{ti} = new Shape();\n")
            # Coordinates in the shape layer's local space (anchor at strip center).
            ty_local = tick_y - strip_cy
            x0_local = (tick_x0) - strip_cx
            x1_local = (tick_x1) - strip_cx
            script.append(
                f"cmapTickShape{ti}.vertices = [[{x0_local}, {ty_local}], [{x1_local}, {ty_local}]];\n"
            )
            script.append(f"cmapTickShape{ti}.closed = false;\n")
            script.append(f"cmapTickPath{ti}.setValue(cmapTickShape{ti});\n")
            script.append(
                f"var cmapTickStroke{ti} = cmapTickContents{ti}.addProperty('ADBE Vector Graphic - Stroke');\n"
            )
            script.append(
                f"cmapTickStroke{ti}.property('ADBE Vector Stroke Color').setValue({ui_color_js});\n"
            )
            script.append(
                f"cmapTickStroke{ti}.property('ADBE Vector Stroke Width').setValue(2);\n"
            )
            if self.animate_opacity:
                script.append(
                    f"cmapTick{ti}.property('Transform').property('Opacity').setValueAtTime(0, 0);\n"
                )
                script.append(
                    f"cmapTick{ti}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 0.7}, 100);\n"
                )
                if self.easy_ease:
                    script.append(
                        f"applyEasyEase(cmapTick{ti}.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n"
                    )
            else:
                script.append(
                    f"cmapTick{ti}.property('Transform').property('Opacity').setValue(100);\n"
                )

            # Tick label to the right of the tick.
            label_safe = str(label_text).replace('"', '\\"')
            script.append(f"var cmapTickLabel{ti} = comp.layers.addText(\"{label_safe}\");\n")
            label_x = tick_x1 + tick_label_gap
            script.append(
                f"cmapTickLabel{ti}.property('Transform').property('Position').setValue([{label_x}, {tick_y}]);\n"
            )
            script.append(f"cmapTickLabel{ti}.parent = cmapNull;\n")
            script.append(
                f"var cmapTickLabelProp{ti} = cmapTickLabel{ti}.property('Source Text');\n"
            )
            script.append(f"var cmapTickLabelDoc{ti} = cmapTickLabelProp{ti}.value;\n")
            script.append(f"cmapTickLabelDoc{ti}.fontSize = {tick_font};\n")
            script.append(f"cmapTickLabelDoc{ti}.font = \"{self.font_tick}\";\n")
            script.append(
                f"cmapTickLabelDoc{ti}.fillColor = {ui_color_js};\n"
            )
            script.append(
                f"cmapTickLabelDoc{ti}.justification = ParagraphJustification.LEFT_JUSTIFY;\n"
            )
            script.append(f"cmapTickLabelProp{ti}.setValue(cmapTickLabelDoc{ti});\n")
            script.append(
                f"var cmapTickLabelSR{ti} = cmapTickLabel{ti}.sourceRectAtTime(0, false);\n"
            )
            script.append(
                f"var cmapTickLabelAP{ti} = cmapTickLabel{ti}.property('Transform').property('Anchor Point');\n"
            )
            script.append(
                f"cmapTickLabelAP{ti}.setValue([cmapTickLabelSR{ti}.left, cmapTickLabelSR{ti}.top + cmapTickLabelSR{ti}.height/2]);\n"
            )
            # Note: tick labels intentionally use a plain opacity fade (below)
            # rather than the per-character slide-in animator the title/axis
            # labels use — keeps the small numbers crisp instead of flickering
            # in character-by-character.
            if self.animate_opacity:
                script.append(
                    f"cmapTickLabel{ti}.property('Transform').property('Opacity').setValueAtTime(0, 0);\n"
                )
                script.append(
                    f"cmapTickLabel{ti}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 0.8}, 100);\n"
                )
                if self.easy_ease:
                    script.append(
                        f"applyEasyEase(cmapTickLabel{ti}.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n"
                    )
            else:
                script.append(
                    f"cmapTickLabel{ti}.property('Transform').property('Opacity').setValue(100);\n"
                )

        # Rotated axis label (gradient column name) to the right of the tick labels.
        if name:
            label_text = str(name).replace('"', '\\"').replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
            # Estimate horizontal room used by tick labels (avg glyph width ~ 0.6 * fontSize).
            estimated_label_w = max_label_chars * tick_font * 0.6
            # Smaller than the main axis labels (28 vs 41) so the colorbar reads as a
            # secondary axis, with a generous extra gap so the rotated label can't
            # collide with the tick numbers.
            axis_label_font = int(28 * self.font_scale)
            extra_gap = max(18, 32.0 * self.font_scale)
            axis_label_x = (
                strip_left + strip_w + tick_len + tick_label_gap
                + estimated_label_w + extra_gap + axis_label_font / 2
            )
            axis_label_y = strip_cy
            script.append(f"var cmapAxisLabel = comp.layers.addText(\"{label_text}\");\n")
            script.append(
                f"cmapAxisLabel.property('Transform').property('Position').setValue([{axis_label_x}, {axis_label_y}]);\n"
            )
            script.append("cmapAxisLabel.parent = cmapNull;\n")
            script.append("cmapAxisLabel.property('Transform').property('Rotation').setValue(-90);\n")
            script.append("var cmapAxisLabelProp = cmapAxisLabel.property('Source Text');\n")
            script.append("var cmapAxisLabelDoc = cmapAxisLabelProp.value;\n")
            script.append(
                f"cmapAxisLabelDoc.fontSize = {axis_label_font};\n"
            )
            script.append(f"cmapAxisLabelDoc.font = \"{self.font_label}\";\n")
            script.append(
                f"cmapAxisLabelDoc.fillColor = {ui_color_js};\n"
            )
            script.append(
                "cmapAxisLabelDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n"
            )
            script.append("cmapAxisLabelProp.setValue(cmapAxisLabelDoc);\n")
            script.append(
                self._generate_text_slide_in_jsx(
                    "cmapAxisLabel", "CmapAxisLabel", ANIM_DURATION
                )
            )
            if self.animate_opacity:
                script.append(
                    "cmapAxisLabel.property('Transform').property('Opacity').setValueAtTime(0, 0);\n"
                )
                script.append(
                    f"cmapAxisLabel.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 0.9}, 100);\n"
                )
                if self.easy_ease:
                    script.append(
                        f"applyEasyEase(cmapAxisLabel.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n"
                    )
            else:
                script.append(
                    "cmapAxisLabel.property('Transform').property('Opacity').setValue(100);\n"
                )

    def _generate_drop_shadow_jsx(self, layer_name: str, index: str = "") -> str:
        """Generate JSX for drop shadow effect (caller decides whether to apply)."""
        jsx = []
        jsx.append(f"var fx{index} = {layer_name}.property('Effects').addProperty('ADBE Drop Shadow');\n")
        jsx.append(f"fx{index}.property('Direction').setValue({DEFAULT_DROP_SHADOW['direction']});\n")
        jsx.append(f"fx{index}.property('Distance').setValue({DEFAULT_DROP_SHADOW['distance']});\n")
        jsx.append(f"fx{index}.property('Softness').setValue({DEFAULT_DROP_SHADOW['softness']});\n")
        jsx.append(f"fx{index}.property('Shadow Color').setValue({color_to_js(DEFAULT_DROP_SHADOW['color'])});\n")
        jsx.append(f"fx{index}.property('Opacity').setValue({255 * DEFAULT_DROP_SHADOW['opacity']});\n")

        return "".join(jsx)

    def _generate_text_slide_in_jsx(self, layer_name: str, index: str, ANIM_DURATION: float, y_offset: int | None = None, delay: float = 0.0) -> str:
        """Generate JSX for a text animator slide-in effect on a text layer."""
        if y_offset is None:
            y_offset = max(10, int(15 * self.font_scale))
        # x_offset = max(10, int(15 * self.font_scale))
        x_offset = 0
        jsx = []
        jsx.append(f"var {layer_name}Animator{index} = addTextAnimator({layer_name});\n")
        jsx.append(f"var {layer_name}Selector{index} = addTextSelector({layer_name}Animator{index});\n")
        # jsx.append(f"if ({layer_name}Selector{index}) {{\n")
        jsx.append(f"    maybeSetValue({layer_name}Selector{index}, 'ADBE Text Percent Start', 0);\n")
        jsx.append(f"    maybeSetValue({layer_name}Selector{index}, 'ADBE Text Percent End', 100);\n")
        jsx.append(f"    maybeSetValue({layer_name}Selector{index}, 'ADBE Text Range Units', 1);\n")
        jsx.append(f"    maybeSetValue({layer_name}Selector{index}, 'ADBE Text Selector Mode', 1);\n")
        jsx.append(f"    maybeSetValue({layer_name}Selector{index}, 'ADBE Text Range Shape', 2);\n")
        jsx.append(f"    maybeSetValueDeep({layer_name}Selector{index}, 'ADBE Text Range Shape', 2);\n")
        jsx.append(f"    maybeSetValue({layer_name}Selector{index}, 'ADBE Text Selector Smoothness', 100);\n")
        jsx.append(f"    var {layer_name}Offset{index} = {layer_name}Selector{index}.property('ADBE Text Percent Offset');\n")
        jsx.append(f"    {layer_name}Offset{index}.setValueAtTime({delay}, -100);\n")
        jsx.append(f"    {layer_name}Offset{index}.setValueAtTime({delay + ANIM_DURATION}, 100);\n")
        # jsx.append(f"}}\n")
        jsx.append(f"if ({layer_name}Animator{index}) {{\n")
        # Position animator
        jsx.append(f"    var {layer_name}Pos{index} = {layer_name}Animator{index}.property('ADBE Text Animator Properties').addProperty('ADBE Text Position 3D');\n")
        jsx.append(f"    if ({layer_name}Pos{index}) {layer_name}Pos{index}.setValue([{x_offset}, {y_offset}, 0]);\n")

        # Opacity animator
        jsx.append(f"    var {layer_name}Opacity{index} = {layer_name}Animator{index}.property('ADBE Text Animator Properties').addProperty('ADBE Text Opacity');\n")
        jsx.append(f"    if ({layer_name}Opacity{index}) {layer_name}Opacity{index}.setValue(0);\n")
        if self.easy_ease:
            jsx.append(f"    if ({layer_name}Offset{index}) applyEasyEase({layer_name}Offset{index}, {self.ease_speed}, {self.ease_influence});\n")
        jsx.append(f"}}\n")
        return "".join(jsx)

    def _axis_trim_reach_time(self, tick_pos, axis_min, axis_max, anim_duration, scale: str):
        """Seconds when an axis trim path (0→100% over anim_duration) reaches tick_pos."""
        progress = max(0.0, min(1.0, self._data_norm(tick_pos, axis_min, axis_max, scale)))
        return anim_duration * progress

    def _generate_tick_opacity_jsx(self, script, prop_expr, appear_time, anim_duration, fallback_fraction):
        """Opacity 0→100 fade; when animate_axes, fade starts as trim path reaches the tick."""
        # tick_fade = min(1, max(0.18, anim_duration * 0.36))
        tick_fade = 0.5
        if self.animate_axes:
            # setValue (not keyframed) keeps ticks hidden before appear_time without a stray t=0 key
            if appear_time > 0:
                script.append(f"{prop_expr}.setValue(0);\n")
            script.append(f"{prop_expr}.setValueAtTime({appear_time}, 0);\n")
            script.append(f"{prop_expr}.setValueAtTime({appear_time + tick_fade}, 100);\n")
        else:
            script.append(f"{prop_expr}.setValueAtTime(0, 0);\n")
            script.append(f"{prop_expr}.setValueAtTime({anim_duration * fallback_fraction}, 100);\n")
        if self.easy_ease:
            script.append(f"applyEasyEase({prop_expr}, {self.ease_speed}, {self.ease_influence});\n")

    def _generate_spine_jsx(
        self,
        script,
        layer_var: str,
        layer_label: str,
        x0s: float,
        y0s: float,
        x1s: float,
        y1s: float,
        center_x: float,
        center_y: float,
        ANIM_DURATION: float,
        animate_trim: bool = False,
        shadow_suffix: str = "",
        view_geom_fn=None,
    ) -> None:
        """Single axis/frame spine segment in shape-local coordinates."""
        script.append(f"var {layer_var} = comp.layers.addShape();\n")
        script.append(f"{layer_var}.name = \"{layer_label}\";\n")
        script.append(f"{layer_var}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
        script.append(f"{layer_var}.parent = PlotAnchor;\n")
        script.append(f"var {layer_var}Contents = {layer_var}.property('ADBE Root Vectors Group');\n")
        script.append(f"var {layer_var}PathGroup = {layer_var}Contents.addProperty('ADBE Vector Shape - Group');\n")
        script.append(f"var {layer_var}Path = {layer_var}PathGroup.property('ADBE Vector Shape');\n")
        script.append(f"var {layer_var}Shape = new Shape();\n")
        script.append(f"{layer_var}Shape.vertices = [[{x0s}, {y0s}], [{x1s}, {y1s}]];\n")
        script.append(f"{layer_var}Shape.closed = false;\n")
        script.append(f"{layer_var}Path.setValue({layer_var}Shape);\n")
        if self._view_animated and view_geom_fn is not None:
            self._emit_view_path_kf(script, f"{layer_var}Path", f"spine_{layer_var}", view_geom_fn, closed=False)
        script.append(f"var {layer_var}Stroke = {layer_var}Contents.addProperty('ADBE Vector Graphic - Stroke');\n")
        script.append(f"{layer_var}Stroke.property('ADBE Vector Stroke Color').setValue({color_to_js(self.ui_color)});\n")
        script.append(f"{layer_var}Stroke.property('ADBE Vector Stroke Width').setValue(3);\n")
        script.append(f"{layer_var}Stroke.property('ADBE Vector Stroke Opacity').setValue(100);\n")
        if animate_trim:
            script.append(f"var {layer_var}Trim = {layer_var}Contents.addProperty('ADBE Vector Filter - Trim');\n")
            script.append(f"var {layer_var}TrimEnd = {layer_var}Trim.property('ADBE Vector Trim End');\n")
            script.append(f"{layer_var}TrimEnd.setValueAtTime(0, 0);\n")
            script.append(f"{layer_var}TrimEnd.setValueAtTime({ANIM_DURATION}, 100);\n")
            if self.easy_ease:
                script.append(
                    f"applyEasyEase({layer_var}TrimEnd, {self.ease_speed}, {self.ease_influence});\n"
                )
        if self.drop_shadow:
            script.append(self._generate_drop_shadow_jsx(layer_var, shadow_suffix or layer_label))

    def _generate_axes_jsx(self, script, center_x, center_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad, ANIM_DURATION, has_barh=False):
        """Helper function to generate axes and ticks JSX code."""
        y_axis_x = self._resolve_yaxis_x(xmin_pad, xmax_pad)
        x_axis_y = self._resolve_xaxis_y(ymin_pad, ymax_pad, has_barh)
        x_labels_below = self._x_tick_labels_below_axis(x_axis_y, ymin_pad, ymax_pad)
        y_labels_left = self._y_tick_labels_left_of_axis(y_axis_x, xmin_pad, xmax_pad)
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
        span_y = max(abs(ymax_pad - ymin_pad), 1e-12)
        span_x = max(abs(xmax_pad - xmin_pad), 1e-12)

        def _xspine_geom(xmn, xmx, ymn, ymx):
            xay = self._resolve_xaxis_y(ymn, ymx, has_barh)
            return [self._data_to_shape(xmn, xay, xmn, xmx, ymn, ymx),
                    self._data_to_shape(xmx, xay, xmn, xmx, ymn, ymx)]

        def _yspine_geom(xmn, xmx, ymn, ymx):
            yax = self._resolve_yaxis_x(xmn, xmx)
            return [self._data_to_shape(yax, ymn, xmn, xmx, ymn, ymx),
                    self._data_to_shape(yax, ymx, xmn, xmx, ymn, ymx)]

        if self.show_xaxis:
            self._generate_spine_jsx(
                script, "axesLayer", "Axes", x0s, y0s, x1s, y1s,
                center_x, center_y, ANIM_DURATION,
                animate_trim=self.animate_axes, shadow_suffix="Axes",
                view_geom_fn=_xspine_geom if self._view_animated else None,
            )

        if self.show_yaxis:
            self._generate_spine_jsx(
                script, "yAxisLayer", "Y-Axis", yx0s, yy0s, yx1s, yy1s,
                center_x, center_y, ANIM_DURATION,
                animate_trim=self.animate_axes, shadow_suffix="YAxis",
                view_geom_fn=_yspine_geom if self._view_animated else None,
            )

        if self.plot_frame:
            x_at_bottom = abs(x_axis_y - ymin_pad) <= span_y * 1e-6
            y_at_left = abs(y_axis_x - xmin_pad) <= span_x * 1e-6
            frame_top_y = ymax_pad if x_at_bottom else ymin_pad
            frame_right_x = xmax_pad if y_at_left else xmin_pad
            # Trim paths start at the corner opposite where the primary axes meet
            join_x, join_y = y_axis_x, x_axis_y
            frame_origin_x = xmax_pad if abs(join_x - xmin_pad) <= span_x * 1e-6 else xmin_pad
            frame_origin_y = ymax_pad if abs(join_y - ymin_pad) <= span_y * 1e-6 else ymin_pad

            if abs(x_axis_y - frame_top_y) > span_y * 1e-6:
                tx_lo, ty_lo = self._data_to_shape(
                    xmin_pad, frame_top_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad
                )
                tx_hi, ty_hi = self._data_to_shape(
                    xmax_pad, frame_top_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad
                )
                if abs(frame_origin_x - xmax_pad) <= span_x * 1e-6:
                    tx0s, ty0s, tx1s, ty1s = tx_hi, ty_hi, tx_lo, ty_lo
                else:
                    tx0s, ty0s, tx1s, ty1s = tx_lo, ty_lo, tx_hi, ty_hi
                self._generate_spine_jsx(
                    script, "frameTopLayer", "Frame-Top", tx0s, ty0s, tx1s, ty1s,
                    center_x, center_y, ANIM_DURATION,
                    animate_trim=self.animate_axes, shadow_suffix="FrameTop",
                )

            if abs(y_axis_x - frame_right_x) > span_x * 1e-6:
                rx_lo, ry_lo = self._data_to_shape(
                    frame_right_x, ymin_pad, xmin_pad, xmax_pad, ymin_pad, ymax_pad
                )
                rx_hi, ry_hi = self._data_to_shape(
                    frame_right_x, ymax_pad, xmin_pad, xmax_pad, ymin_pad, ymax_pad
                )
                if abs(frame_origin_y - ymax_pad) <= span_y * 1e-6:
                    rx0s, ry0s, rx1s, ry1s = rx_hi, ry_hi, rx_lo, ry_lo
                else:
                    rx0s, ry0s, rx1s, ry1s = rx_lo, ry_lo, rx_hi, ry_hi
                self._generate_spine_jsx(
                    script, "frameRightLayer", "Frame-Right", rx0s, ry0s, rx1s, ry1s,
                    center_x, center_y, ANIM_DURATION,
                    animate_trim=self.animate_axes, shadow_suffix="FrameRight",
                )

        # X Ticks
        if self.show_xaxis and self.xticks:
            for idx, (pos, label) in enumerate(self.xticks):
                if self._normalize_scale(self.xscale) == "log" and pos <= 0:
                    continue
                if self._xtick_labels_auto:
                    label = self._format_tick_label(pos, self.xscale)
                if not str(label).strip():
                    continue
                pos_var = str(pos).replace('-', 'm').replace('.', '_')
                tick_xs, tick_ys0, _, tick_ys1 = self._x_tick_shape_coords(
                    pos, x_axis_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad
                )
                tick_label_gap = self._tick_label_gap()
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
                if self._view_animated:
                    def _xtick_geom(xmn, xmx, ymn, ymx, _pos=pos):
                        xay = self._resolve_xaxis_y(ymn, ymx, has_barh)
                        txs, tys0, _u, tys1 = self._x_tick_shape_coords(_pos, xay, xmn, xmx, ymn, ymx)
                        return [(txs, tys0), (txs, tys1)]
                    self._emit_view_path_kf(script, f"xtickPath{pos_var}", f"xt{pos_var}", _xtick_geom, closed=False)
                script.append(f"var xtickStroke{pos_var} = xtickContents{pos_var}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"xtickStroke{pos_var}.property('ADBE Vector Stroke Color').setValue({color_to_js(self.ui_color)});\n")
                script.append(f"xtickStroke{pos_var}.property('ADBE Vector Stroke Width').setValue(2);\n")
                # Opacity: view-window clip when bounds animate; else axis-trim fade
                _xtick_stroke_op = f"xtickStroke{pos_var}.property('ADBE Vector Stroke Opacity')"
                if self._view_animated:
                    _xt_entrance = _xt_fade = None
                    if self.animate_opacity:
                        if self.animate_axes:
                            _xt_entrance = self._axis_trim_reach_time(
                                pos, xmin_pad, xmax_pad, ANIM_DURATION, self.xscale
                            ) * 0.8
                            _xt_fade = 0.5
                        else:
                            _xt_entrance = 0.0
                            _xt_fade = ANIM_DURATION * 0.8
                    self._emit_axis_visibility_kf(
                        script, _xtick_stroke_op, pos, "x",
                        entrance_start=_xt_entrance, entrance_fade=_xt_fade or 0.5,
                    )
                elif self.animate_opacity:
                    x_appear = self._axis_trim_reach_time(pos, xmin_pad, xmax_pad, ANIM_DURATION, self.xscale)
                    self._generate_tick_opacity_jsx(
                        script, _xtick_stroke_op, x_appear * 0.8, ANIM_DURATION, 0.8,
                    )
                else:
                    script.append(f"{_xtick_stroke_op}.setValue(100);\n")
                # Tick label beside the tick (below or above spine depending on xaxis_location)
                if self.show_tick_labels:
                    lx, _ = self._data_to_shape(pos, x_axis_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    if x_labels_below:
                        label_ly = max(tick_ys0, tick_ys1) + tick_label_gap
                        anchor_js = f"[xsr{pos_var}.left + xsr{pos_var}.width/2, xsr{pos_var}.top]"
                    else:
                        label_ly = min(tick_ys0, tick_ys1) - tick_label_gap
                        anchor_js = f"[xsr{pos_var}.left + xsr{pos_var}.width/2, xsr{pos_var}.top + xsr{pos_var}.height]"
                    script.append(f"var xtickLabel{pos_var} = comp.layers.addText(\"{label}\");\n")
                    script.append(f"xtickLabel{pos_var}.property('Transform').property('Position').setValue([{center_x + lx}, {center_y + label_ly}]);\n")
                    script.append(f"xtickLabel{pos_var}.parent = PlotAnchor;\n")
                    if self._view_animated:
                        script.append(f"var xtickLabelPos{pos_var} = xtickLabel{pos_var}.property('Transform').property('Position');\n")
                        def _xlabel_geom(xmn, xmx, ymn, ymx, _pos=pos, _below=x_labels_below, _gap=tick_label_gap):
                            xay = self._resolve_xaxis_y(ymn, ymx, has_barh)
                            txs, tys0, _u, tys1 = self._x_tick_shape_coords(_pos, xay, xmn, xmx, ymn, ymx)
                            llx = self._data_to_shape(_pos, xay, xmn, xmx, ymn, ymx)[0]
                            lly = (max(tys0, tys1) + _gap) if _below else (min(tys0, tys1) - _gap)
                            return (llx, lly)
                        self._emit_view_pos_kf(script, f"xtickLabelPos{pos_var}", _xlabel_geom)
                    script.append(f"var xtickLabelProp{pos_var} = xtickLabel{pos_var}.property('Source Text');\n")
                    script.append(f"var xtickLabelDoc{pos_var} = xtickLabelProp{pos_var}.value;\n")
                    script.append(f"xtickLabelDoc{pos_var}.fontSize = {int(27 * self.font_scale)};\n")
                    script.append(f"xtickLabelDoc{pos_var}.font = \"{self.font_tick}\";\n")
                    script.append(f"xtickLabelDoc{pos_var}.fillColor = {color_to_js(self.ui_color)};\n")
                    script.append(f"xtickLabelDoc{pos_var}.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
                    script.append(f"xtickLabelProp{pos_var}.setValue(xtickLabelDoc{pos_var});\n")
                    script.append(f"var xsr{pos_var} = xtickLabel{pos_var}.sourceRectAtTime(0, false);\n")
                    script.append(f"var xap{pos_var} = xtickLabel{pos_var}.property('Transform').property('Anchor Point');\n")
                    script.append(f"xap{pos_var}.setValue({anchor_js});\n")
                    # script.append(self._generate_text_slide_in_jsx(f"xtickLabel{pos_var}", f"XTickLabel{pos_var}", ANIM_DURATION))
                    _xtick_label_op = f"xtickLabel{pos_var}.property('Transform').property('Opacity')"
                    if self._view_animated:
                        _xt_entrance = _xt_fade = None
                        if self.animate_opacity:
                            if self.animate_axes:
                                _xt_entrance = self._axis_trim_reach_time(
                                    pos, xmin_pad, xmax_pad, ANIM_DURATION, self.xscale
                                ) * 0.8
                                _xt_fade = 0.5
                            else:
                                _xt_entrance = 0.0
                                _xt_fade = ANIM_DURATION * 0.9
                        self._emit_axis_visibility_kf(
                            script, _xtick_label_op, pos, "x",
                            entrance_start=_xt_entrance, entrance_fade=_xt_fade or 0.5,
                        )
                    elif self.animate_opacity:
                        self._generate_tick_opacity_jsx(
                            script, _xtick_label_op, x_appear * 0.8, ANIM_DURATION, 0.9,
                        )
                    else:
                        script.append(f"{_xtick_label_op}.setValue(100);\n")
        # Y Ticks
        if self.show_yaxis and self.yticks:
            for idx, (pos, label) in enumerate(self.yticks):
                if self._normalize_scale(self.yscale) == "log" and pos <= 0:
                    continue
                if self._ytick_labels_auto:
                    label = self._format_tick_label(pos, self.yscale)
                if not str(label).strip():
                    continue
                pos_var = str(pos).replace('-', 'm').replace('.', '_')
                tick_xs0, tick_ys, tick_xs1, _ = self._y_tick_shape_coords(
                    y_axis_x, pos, xmin_pad, xmax_pad, ymin_pad, ymax_pad
                )
                tick_label_gap = self._tick_label_gap()
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
                if self._view_animated:
                    def _ytick_geom(xmn, xmx, ymn, ymx, _pos=pos):
                        yax = self._resolve_yaxis_x(xmn, xmx)
                        txs0, tys, txs1, _u = self._y_tick_shape_coords(yax, _pos, xmn, xmx, ymn, ymx)
                        return [(txs0, tys), (txs1, tys)]
                    self._emit_view_path_kf(script, f"ytickPath{pos_var}", f"yt{pos_var}", _ytick_geom, closed=False)
                script.append(f"var ytickStroke{pos_var} = ytickContents{pos_var}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"ytickStroke{pos_var}.property('ADBE Vector Stroke Color').setValue({color_to_js(self.ui_color)});\n")
                script.append(f"ytickStroke{pos_var}.property('ADBE Vector Stroke Width').setValue(2);\n")
                _ytick_stroke_op = f"ytickStroke{pos_var}.property('ADBE Vector Stroke Opacity')"
                if self._view_animated:
                    _yt_entrance = _yt_fade = None
                    if self.animate_opacity:
                        if self.animate_axes:
                            _yt_entrance = self._axis_trim_reach_time(
                                pos, ymin_pad, ymax_pad, ANIM_DURATION, self.yscale
                            ) * 0.8
                            _yt_fade = 0.5
                        else:
                            _yt_entrance = 0.0
                            _yt_fade = ANIM_DURATION * 0.8
                    self._emit_axis_visibility_kf(
                        script, _ytick_stroke_op, pos, "y",
                        entrance_start=_yt_entrance, entrance_fade=_yt_fade or 0.5,
                    )
                elif self.animate_opacity:
                    y_appear = self._axis_trim_reach_time(pos, ymin_pad, ymax_pad, ANIM_DURATION, self.yscale)
                    self._generate_tick_opacity_jsx(
                        script, _ytick_stroke_op, y_appear * 0.8, ANIM_DURATION, 0.8,
                    )
                else:
                    script.append(f"{_ytick_stroke_op}.setValue(100);\n")
                # Tick label beside the tick (left or right of spine depending on yaxis_location)
                if self.show_tick_labels:
                    _, ly = self._data_to_shape(y_axis_x, pos, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    if y_labels_left:
                        label_lx = min(tick_xs0, tick_xs1) - tick_label_gap
                        anchor_js = f"[ysr{pos_var}.left + ysr{pos_var}.width, ysr{pos_var}.top + ysr{pos_var}.height/2]"
                        justify = "ParagraphJustification.RIGHT_JUSTIFY"
                    else:
                        label_lx = max(tick_xs0, tick_xs1) + tick_label_gap
                        anchor_js = f"[ysr{pos_var}.left, ysr{pos_var}.top + ysr{pos_var}.height/2]"
                        justify = "ParagraphJustification.LEFT_JUSTIFY"
                    script.append(f"var ytickLabel{pos_var} = comp.layers.addText(\"{label}\");\n")
                    script.append(f"ytickLabel{pos_var}.property('Transform').property('Position').setValue([{center_x + label_lx}, {center_y + ly}]);\n")
                    script.append(f"ytickLabel{pos_var}.parent = PlotAnchor;\n")
                    if self._view_animated:
                        script.append(f"var ytickLabelPos{pos_var} = ytickLabel{pos_var}.property('Transform').property('Position');\n")
                        def _ylabel_geom(xmn, xmx, ymn, ymx, _pos=pos, _left=y_labels_left, _gap=tick_label_gap):
                            yax = self._resolve_yaxis_x(xmn, xmx)
                            txs0, tys, txs1, _u = self._y_tick_shape_coords(yax, _pos, xmn, xmx, ymn, ymx)
                            lly = self._data_to_shape(yax, _pos, xmn, xmx, ymn, ymx)[1]
                            llx = (min(txs0, txs1) - _gap) if _left else (max(txs0, txs1) + _gap)
                            return (llx, lly)
                        self._emit_view_pos_kf(script, f"ytickLabelPos{pos_var}", _ylabel_geom)
                    script.append(f"var ytickLabelProp{pos_var} = ytickLabel{pos_var}.property('Source Text');\n")
                    script.append(f"var ytickLabelDoc{pos_var} = ytickLabelProp{pos_var}.value;\n")
                    script.append(f"ytickLabelDoc{pos_var}.fontSize = {int(27 * self.font_scale)};\n")
                    script.append(f"ytickLabelDoc{pos_var}.font = \"{self.font_tick}\";\n")
                    script.append(f"ytickLabelDoc{pos_var}.fillColor = {color_to_js(self.ui_color)};\n")
                    script.append(f"ytickLabelDoc{pos_var}.justification = {justify};\n")
                    script.append(f"ytickLabelProp{pos_var}.setValue(ytickLabelDoc{pos_var});\n")
                    script.append(f"var ysr{pos_var} = ytickLabel{pos_var}.sourceRectAtTime(0, false);\n")
                    script.append(f"var yap{pos_var} = ytickLabel{pos_var}.property('Transform').property('Anchor Point');\n")
                    script.append(f"yap{pos_var}.setValue({anchor_js});\n")
                    # script.append(self._generate_text_slide_in_jsx(f"ytickLabel{pos_var}", f"YTickLabel{pos_var}", ANIM_DURATION))
                    _ytick_label_op = f"ytickLabel{pos_var}.property('Transform').property('Opacity')"
                    if self._view_animated:
                        _yt_entrance = _yt_fade = None
                        if self.animate_opacity:
                            if self.animate_axes:
                                _yt_entrance = self._axis_trim_reach_time(
                                    pos, ymin_pad, ymax_pad, ANIM_DURATION, self.yscale
                                ) * 0.8
                                _yt_fade = 0.5
                            else:
                                _yt_entrance = 0.0
                                _yt_fade = ANIM_DURATION * 0.9
                        self._emit_axis_visibility_kf(
                            script, _ytick_label_op, pos, "y",
                            entrance_start=_yt_entrance, entrance_fade=_yt_fade or 0.5,
                        )
                    elif self.animate_opacity:
                        self._generate_tick_opacity_jsx(
                            script, _ytick_label_op, y_appear * 0.8, ANIM_DURATION, 0.9,
                        )
                    else:
                        script.append(f"{_ytick_label_op}.setValue(100);\n")

    # Per-preset look for each distress texture: (filename, opacity, blend, scale_mult).
    _DISTRESS_PRESETS = {
        1: ("grunge1.jpg", 10, "OVERLAY", 1.0),
        2: ("grunge2.png", 40, "MULTIPLY", 1.0),
        3: ("grunge3.jpg", 25, "OVERLAY", 1.6),
        4: ("grunge4.jpg", 20, None, 1.0),
        5: ("grunge5.jpg", 50, None, 1.0),
        6: ("grunge6.jpg", 60, "OVERLAY", 1.0),
        7: ("grunge7.jpg", 60, "OVERLAY", 1.0),
        8: ("grunge8.png", 30, "MULTIPLY", 1.0),
    }

    def _emit_scatter_image_point_jsx(
        self, script, elem_idx, pt_idx, image_path, center_x, center_y, sx, sy,
        px, py, radius, marker, scatter_opacity, outline, outline_width,
        outline_color, outline_opacity, fallback_color, delay, start_time,
        anim_time, elem_ease_speed, elem_ease_influence, drop_shadow,
        clip_to_view=True, move_start=None, reveal_duration=0.3,
    ):
        """Emit one scatter marker whose fill is an image clipped to ``marker``.

        move_start: Optional ``(sx, sy)`` shape-space starting coordinate. When
        given, the marker's null starts there and slides to ``(sx, sy)`` over
        its entrance window instead of appearing directly at its final spot.
        """
        i, j = elem_idx, pt_idx
        path_js = json.dumps(os.path.abspath(image_path).replace("\\", "/"))
        d = radius * 2
        null_name = f"scatterNull{i}_{j}"
        init_sx, init_sy = move_start if move_start is not None else (sx, sy)
        script.append(f"var {null_name} = comp.layers.addNull();\n")
        script.append(f"{null_name}.name = \"Scatter_{i}_{j}\";\n")
        script.append(
            f"{null_name}.property('Transform').property('Position')"
            f".setValue([{center_x + init_sx}, {center_y + init_sy}]);\n"
        )
        script.append(f"{null_name}.parent = PlotAnchor;\n")
        if self._view_animated:
            script.append(
                f"var scatterPos{i}_{j} = {null_name}.property('Transform')"
                f".property('Position');\n"
            )
            self._emit_view_pos_kf(
                script, f"scatterPos{i}_{j}",
                lambda xmn, xmx, ymn, ymx, _x=px, _y=py: self._data_to_shape(
                    _x, _y, xmn, xmx, ymn, ymx
                ),
            )
        elif move_start is not None:
            script.append(
                f"var scatterPos{i}_{j} = {null_name}.property('Transform')"
                f".property('Position');\n"
            )
            # Once parented, AE has already folded the center offset into the
            # base Position value (it preserves world position on parenting),
            # so subsequent explicit keyframes must use the raw shape-space
            # coordinates (sx, sy) -- matching `_emit_view_pos_kf` above --
            # not `center_x + sx` again.
            script.extend(self._emit_point_move_kf(
                f"scatterPos{i}_{j}",
                (init_sx, init_sy),
                (sx, sy),
                entrance_start=delay + start_time, entrance_dur=anim_time or 0.0,
                ease_speed=elem_ease_speed, ease_influence=elem_ease_influence,
            ))
            script.append(
                f"var scatterScale{i}_{j} = {null_name}.property('Transform')"
                f".property('Scale');\n"
            )
            script.extend(self._emit_point_hide_until_kf(
                f"scatterScale{i}_{j}", delay + start_time,
                reveal_duration=reveal_duration,
                ease_speed=elem_ease_speed, ease_influence=elem_ease_influence,
            ))

        script.append(f"var scatterImg{i}_{j} = comp.layers.add(__aegraphImportFootage({path_js}));\n")
        script.append(f"if (scatterImg{i}_{j}) {{\n")
        script.append(f"    scatterImg{i}_{j}.name = \"ScatterImg_{i}_{j}\";\n")
        script.append(f"    scatterImg{i}_{j}.parent = {null_name};\n")
        script.append(f"    scatterImg{i}_{j}.property('Transform').property('Position').setValue([0, 0]);\n")
        script.append(
            f"    if (scatterImg{i}_{j}.source && scatterImg{i}_{j}.source.width > 0 "
            f"&& scatterImg{i}_{j}.source.height > 0) {{\n"
        )
        script.append(
            f"        var __imgSc{i}_{j} = Math.max({d} / scatterImg{i}_{j}.source.width, "
            f"{d} / scatterImg{i}_{j}.source.height) * 100;\n"
        )
        script.append(
            f"        scatterImg{i}_{j}.property('Transform').property('Scale')"
            f".setValue([__imgSc{i}_{j}, __imgSc{i}_{j}]);\n"
        )
        script.append("    }\n")
        if scatter_opacity < 100:
            script.append(
                f"    scatterImg{i}_{j}.property('Transform').property('Opacity')"
                f".setValue({scatter_opacity});\n"
            )

        script.append(f"    var scatterMatte{i}_{j} = comp.layers.addShape();\n")
        script.append(f"    scatterMatte{i}_{j}.name = \"ScatterMatte_{i}_{j}\";\n")
        script.append(f"    scatterMatte{i}_{j}.parent = {null_name};\n")
        script.append(f"    scatterMatte{i}_{j}.property('Transform').property('Position').setValue([0, 0]);\n")
        for _mask_line in self._screen_space_mask_lines(f"scatterMatte{i}_{j}"):
            script.append("    " + _mask_line)
        script.append(f"    var scatterMatteC{i}_{j} = scatterMatte{i}_{j}.property('ADBE Root Vectors Group');\n")
        for line in _marker_static_jsx(
            f"scatterMatteC{i}_{j}", f"scatterMatteS{i}_{j}", marker, radius
        ).splitlines(keepends=True):
            script.append("    " + line)
        script.append(f"    var scatterMatteF{i}_{j} = scatterMatteC{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
        script.append(f"    scatterMatteF{i}_{j}.property('ADBE Vector Fill Color').setValue([1, 1, 1]);\n")
        script.append(f"    scatterImg{i}_{j}.trackMatteType = TrackMatteType.ALPHA;\n")
        if outline and outline_width > 0:
            # NOTE: the matte layer above is consumed as a track-matte source
            # and never actually renders, so its stroke would be invisible.
            # Emit the outline on its own plain shape layer stacked on top
            # instead, so it's actually visible over the clipped image.
            oc = color_to_js(outline_color or self._darken_color(fallback_color))
            script.append(f"    var scatterOutline{i}_{j} = comp.layers.addShape();\n")
            script.append(f"    scatterOutline{i}_{j}.name = \"ScatterOutline_{i}_{j}\";\n")
            script.append(f"    scatterOutline{i}_{j}.parent = {null_name};\n")
            script.append(f"    scatterOutline{i}_{j}.property('Transform').property('Position').setValue([0, 0]);\n")
            for _mask_line in self._screen_space_mask_lines(f"scatterOutline{i}_{j}"):
                script.append("    " + _mask_line)
            script.append(f"    var scatterOutlineC{i}_{j} = scatterOutline{i}_{j}.property('ADBE Root Vectors Group');\n")
            for line in _marker_static_jsx(
                f"scatterOutlineC{i}_{j}", f"scatterOutlineS{i}_{j}", marker, radius
            ).splitlines(keepends=True):
                script.append("    " + line)
            script.append(f"    var scatterOutlineSt{i}_{j} = scatterOutlineC{i}_{j}.addProperty('ADBE Vector Graphic - Stroke');\n")
            script.append(f"    scatterOutlineSt{i}_{j}.property('ADBE Vector Stroke Color').setValue({oc});\n")
            script.append(f"    scatterOutlineSt{i}_{j}.property('ADBE Vector Stroke Width').setValue({outline_width});\n")
            script.append(f"    scatterOutlineSt{i}_{j}.property('ADBE Vector Stroke Opacity').setValue({outline_opacity});\n")
        script.append("} else {\n")
        script.append(f"    var scatterFallback{i}_{j} = comp.layers.addShape();\n")
        script.append(f"    scatterFallback{i}_{j}.name = \"Scatter_{i}_{j}\";\n")
        script.append(f"    scatterFallback{i}_{j}.parent = {null_name};\n")
        script.append(f"    scatterFallback{i}_{j}.property('Transform').property('Position').setValue([0, 0]);\n")
        for _mask_line in self._screen_space_mask_lines(f"scatterFallback{i}_{j}"):
            script.append("    " + _mask_line)
        script.append(f"    var scatterFallbackC{i}_{j} = scatterFallback{i}_{j}.property('ADBE Root Vectors Group');\n")
        for line in _marker_static_jsx(
            f"scatterFallbackC{i}_{j}", f"scatterFallbackS{i}_{j}", marker, radius
        ).splitlines(keepends=True):
            script.append("    " + line)
        script.append(f"    var scatterFallbackF{i}_{j} = scatterFallbackC{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
        script.append(f"    scatterFallbackF{i}_{j}.property('ADBE Vector Fill Color').setValue({color_to_js(fallback_color)});\n")
        if scatter_opacity < 100:
            script.append(f"    scatterFallbackF{i}_{j}.property('ADBE Vector Fill Opacity').setValue({scatter_opacity});\n")
        if outline and outline_width > 0:
            oc = color_to_js(outline_color or self._darken_color(fallback_color))
            script.append(f"    var scatterFallbackSt{i}_{j} = scatterFallbackC{i}_{j}.addProperty('ADBE Vector Graphic - Stroke');\n")
            script.append(f"    scatterFallbackSt{i}_{j}.property('ADBE Vector Stroke Color').setValue({oc});\n")
            script.append(f"    scatterFallbackSt{i}_{j}.property('ADBE Vector Stroke Width').setValue({outline_width});\n")
            script.append(f"    scatterFallbackSt{i}_{j}.property('ADBE Vector Stroke Opacity').setValue({outline_opacity});\n")
        script.append("}\n")

        if (anim_time and anim_time > 0) or (clip_to_view and self._view_animated):
            if move_start is None:
                scale_lines = self._emit_point_scale_kf(
                    f"scatterScale{i}_{j}", px, py,
                    entrance_start=delay + start_time, entrance_dur=anim_time or 0.0,
                    ease_speed=elem_ease_speed, ease_influence=elem_ease_influence,
                    clip_to_view=clip_to_view,
                )
                if scale_lines:
                    script.append(
                        f"var scatterScale{i}_{j} = {null_name}.property('Transform')"
                        f".property('Scale');\n"
                    )
                    script.extend(scale_lines)

        if drop_shadow:
            script.append(self._generate_drop_shadow_jsx(null_name, f"{i}_{j}img"))

    def _distress_block_jsx(self, center_x, center_y, move_line, parent_var="PlotAnchor"):
        """Emit the JSX for the configured distress (grunge) texture, if any.

        Imports the footage once (reusing it if already in the project),
        scales it to cover the comp, applies the preset's opacity/blend, and
        stacks it just above the background via ``move_line``. Returns a list
        of script lines (empty when no texture is configured).
        """
        preset = self._DISTRESS_PRESETS.get(self.distress_texture)
        if preset is None:
            return []
        fname, opacity, blend, scale_mult = preset
        path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "distress_textures", fname)
        ).replace("\\", "/").replace('"', '\\"')
        mult = "" if scale_mult == 1.0 else f" * {scale_mult}"
        s = [
            f'var distressFile = new File("{path}");\n',
            "if (distressFile.exists) {\n",
            "    var distressFootage = null;\n",
            "    var distressPath = distressFile.fsName.toLowerCase();\n",
            "    for (var i = 1; i <= app.project.numItems; i++) {\n",
            "        var projectItem = app.project.item(i);\n",
            "        if (projectItem instanceof FootageItem && projectItem.mainSource && projectItem.mainSource.file) {\n",
            "            if (projectItem.mainSource.file.fsName.toLowerCase() === distressPath) {\n",
            "                distressFootage = projectItem;\n",
            "                break;\n",
            "            }\n",
            "        }\n",
            "    }\n",
            "    if (!distressFootage) {\n",
            "        var distressImportOptions = new ImportOptions(distressFile);\n",
            "        distressFootage = app.project.importFile(distressImportOptions);\n",
            "    }\n",
            "    var distressLayer = comp.layers.add(distressFootage);\n",
            f"    distressLayer.name = '{fname}';\n",
            f"    distressLayer.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n",
            "    if (distressLayer.source && distressLayer.source.width > 0 && distressLayer.source.height > 0) {\n",
            f"        var distressScale = Math.max(({self.comp_width} / distressLayer.source.width) * 100{mult}, ({self.comp_height} / distressLayer.source.height) * 100{mult});\n",
            "        distressLayer.property('Transform').property('Scale').setValue([distressScale, distressScale]);\n",
            "    }\n",
            f"    distressLayer.property('Transform').property('Opacity').setValue({opacity});\n",
        ]
        if blend:
            s.append(f"    distressLayer.blendingMode = BlendingMode.{blend};\n")
        s.append(move_line)
        if parent_var:
            s.append(f"    distressLayer.parent = {parent_var};\n")
        s.append("}\n")
        return s

    # ------------------------------------------------------------------
    # Experimental "film style" post-processing pass. Each helper below emits
    # one piece of the treatment; see ``aegraph_config.config.film_style_*``
    # for the tunable knobs (and ``film_style_parameters()`` for per-graph
    # overrides) and ``_generate_jsx`` / ``AEFigure._build_jsx`` for how the
    # pieces are stacked and parented.
    # ------------------------------------------------------------------

    def film_style_parameters(self, **kwargs):
        """Override film-style knobs for just this graph (or figure, via
        ``AEFigure.film_style_parameters``), without touching the global
        ``aegraph_config.config`` defaults. Returns ``self`` for chaining.

        Pass any ``aegraph_config.config.film_style_*`` field name *without*
        the ``film_style_`` prefix, e.g.::

            g.film_style_parameters(
                roughen_border=1.0,          # numeric knobs
                element_blur=True,           # bools toggle a whole piece on/off
                element_roughen_kinds={"pie": False, "grid": 0.5},  # per-kind override
                edge_blur_margin=0.08,
                zoom_amount=2.0,             # multiply the push-in distance (1.0 = default, 0.0 = no zoom)
            )

        The ``*_kinds`` dicts (``element_roughen_kinds``, ``element_blur_kinds``,
        ``element_multiply_kinds``) are merged into the existing map rather than
        replacing it, so ``film_style_parameters(element_roughen_kinds={"pie": False})``
        adds an exclusion without losing the default ``{"scatter": False}``.

        For ``element_roughen_kinds``, each value controls Roughen Edges for
        that kind: omit the key to use ``roughen_border``; ``True`` also uses
        ``roughen_border``; ``False`` or ``0`` turns roughen off; any positive
        number sets a custom border width for just that kind (e.g.
        ``{"grid": 0.5, "text": False}``).

        ``element_blur_kinds`` and ``element_multiply_kinds`` still use booleans
        only (``False`` = off, otherwise on with the global amount/mode).
        Recognized kinds: "scatter", "line", "bar", "pie", "heatmap", "quiver",
        "annotation", "evolving_text", "grid", "tick", "errorbar", "refline",
        "band", "colorbar", "text" (titles/labels/legend/ticks), "other".

        Whole-piece toggles: ``paper``, ``light_leak``, ``temporal``,
        ``posterize_time``, ``vignette``, ``edge_blur``, ``zoom``,
        ``element_roughen``, ``element_blur``, ``element_multiply``.
        ``temporal=False`` disables the whole adjustment layer (posterize
        time + exposure flicker); ``posterize_time=False`` keeps exposure
        flicker but skips Posterize Time only.
        """
        overrides = getattr(self, "_film_style_overrides", None)
        if overrides is None:
            overrides = {}
            self._film_style_overrides = overrides
        for key, value in kwargs.items():
            if key.endswith("_kinds") and isinstance(value, dict):
                merged = dict(overrides.get(key, self._fp(key) or {}))
                merged.update(value)
                overrides[key] = merged
            else:
                overrides[key] = value
        return self

    def _fp(self, key):
        """Resolve a film-style parameter: per-graph override (set via
        ``film_style_parameters()``) if present, else the module-level
        ``aegraph_config.config.film_style_<key>`` default."""
        overrides = getattr(self, "_film_style_overrides", None)
        if overrides and key in overrides:
            return overrides[key]
        return getattr(config, f"film_style_{key}")

    def _film_style_min_zoom_fraction(self):
        """The smallest fraction (of the layer's base 100% scale) the push-in
        zoom will ever multiply a layer down to, accounting for
        ``zoom_amount``. Returns ``1.0`` (no shrinkage) when the ``zoom``
        piece is disabled. Used to pre-inflate any layer that must always
        fully cover the frame (e.g. the paper background) even at the
        zoom's most-zoomed-out moment.
        """
        if not self._fp("zoom"):
            return 1.0
        start = float(self._fp("zoom_start"))
        end = float(self._fp("zoom_end"))
        amount = float(self._fp("zoom_amount"))
        eff_start = end - (end - start) * amount
        frac = min(eff_start, end, 1.0)
        return frac if frac > 0.01 else 1.0

    def _film_style_zoom_keyframes_jsx(self, layer_var, scale_expr="100"):
        """Front-heavy push-in: keyframe ``layer_var``'s Scale from
        ``zoom_start`` to ``zoom_end`` (as a fraction of ``scale_expr``, a
        JS expression evaluating to the layer's "no zoom" base scale -- pass
        e.g. an already-declared variable like ``"paperScale"`` for a layer
        that has its own cover-the-frame scale, so the push-in multiplies
        that instead of clobbering it with a bare 87/100) over
        ``zoom_duration`` seconds. ``zoom_amount`` multiplies the push-in
        distance around the fixed ``zoom_end``, so dialing it up/down
        doesn't shift where the graph settles. Eased asymmetrically so the
        layer moves quickly at first and settles with a long, gentle
        deceleration (most motion happens "up front"). Returns ``[]`` when
        the ``zoom`` piece is disabled.
        """
        if not self._fp("zoom"):
            return []
        start = float(self._fp("zoom_start"))
        end = float(self._fp("zoom_end"))
        amount = float(self._fp("zoom_amount"))
        # Scale the push-in distance around the fixed end point, so the graph
        # still settles at its intended framing regardless of `amount`.
        start = end - (end - start) * amount
        dur = float(self._fp("zoom_duration"))
        out_inf = float(self._fp("zoom_ease_out_influence"))
        in_inf = float(self._fp("zoom_ease_in_influence"))
        v = f"__filmZoom_{layer_var}"
        # AE's Transform > Scale property is always 3-dimensional internally
        # (x/y/z) even on a 2D layer, so setValueAtTime/setTemporalEaseAtKey
        # require 3-element arrays -- passing 2 throws "Value array does not
        # have 3 elements".
        out_ease = f"[new KeyframeEase(0,{out_inf}), new KeyframeEase(0,{out_inf}), new KeyframeEase(0,{out_inf})]"
        in_ease = f"[new KeyframeEase(0,{in_inf}), new KeyframeEase(0,{in_inf}), new KeyframeEase(0,{in_inf})]"
        return [
            f"var {v} = {layer_var}.property('Transform').property('Scale');\n",
            f"var {v}_base = ({scale_expr});\n",
            f"{v}.setValueAtTime(0, [{v}_base * {start}, {v}_base * {start}, 100]);\n",
            f"{v}.setValueAtTime({dur}, [{v}_base * {end}, {v}_base * {end}, 100]);\n",
            f"{v}.setTemporalEaseAtKey(1, {out_ease}, {out_ease});\n",
            f"{v}.setTemporalEaseAtKey(2, {in_ease}, {in_ease});\n",
        ]

    def _film_style_paper_jsx(self, center_x, center_y, move_line, parent_var="PlotAnchor", apply_zoom=False):
        """Emit the paper-texture background layer for the film style.

        Mirrors ``_distress_block_jsx``: imports the texture once, scales it
        to cover the comp, tints it, and stacks it just above the
        background. When ``apply_zoom`` is True (figures, where this layer
        has no single panel to parent to), the push-in zoom keyframes are
        added directly to it instead of relying on a parent's zoom. Returns
        ``[]`` when the ``paper`` piece is disabled.
        """
        if not self._fp("paper"):
            return []
        fname = self._fp("paper_file")
        path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "film_elements", fname)
        ).replace("\\", "/").replace('"', '\\"')
        tint_black = list(self._fp("paper_tint_black"))
        tint_white = list(self._fp("paper_tint_white"))
        s = [
            f'var paperFile = new File("{path}");\n',
            "if (paperFile.exists) {\n",
            "    var paperFootage = null;\n",
            "    var paperPath = paperFile.fsName.toLowerCase();\n",
            "    for (var __pi = 1; __pi <= app.project.numItems; __pi++) {\n",
            "        var __pItem = app.project.item(__pi);\n",
            "        if (__pItem instanceof FootageItem && __pItem.mainSource && __pItem.mainSource.file) {\n",
            "            if (__pItem.mainSource.file.fsName.toLowerCase() === paperPath) {\n",
            "                paperFootage = __pItem;\n",
            "                break;\n",
            "            }\n",
            "        }\n",
            "    }\n",
            "    if (!paperFootage) {\n",
            "        var paperImportOptions = new ImportOptions(paperFile);\n",
            "        paperFootage = app.project.importFile(paperImportOptions);\n",
            "    }\n",
            "    var paperLayer = comp.layers.add(paperFootage);\n",
            "    paperLayer.name = 'AEGraph_FilmPaper';\n",
            f"    paperLayer.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n",
            "    var paperScale = 100;\n",
            "    if (paperLayer.source && paperLayer.source.width > 0 && paperLayer.source.height > 0) {\n",
            f"        paperScale = Math.max(({self.comp_width} / paperLayer.source.width) * 100, ({self.comp_height} / paperLayer.source.height) * 100) / {self._film_style_min_zoom_fraction()};\n",
            "        paperLayer.property('Transform').property('Scale').setValue([paperScale, paperScale]);\n",
            "    }\n",
            "    paperLayer.property('Transform').property('Opacity').setValue(100);\n",
            "    var paperTint = paperLayer.property('Effects').addProperty('ADBE Tint');\n",
            "    if (paperTint) {\n",
            f"        maybeSetValue(paperTint, 'ADBE Tint-0001', {tint_black});\n",
            f"        maybeSetValue(paperTint, 'ADBE Tint-0002', {tint_white});\n",
            "        maybeSetValue(paperTint, 'ADBE Tint-0003', 100);\n",
            "    }\n",
            move_line,
        ]
        if parent_var:
            s.append(f"    paperLayer.parent = {parent_var};\n")
        if apply_zoom:
            for line in self._film_style_zoom_keyframes_jsx("paperLayer", scale_expr="paperScale"):
                s.append("    " + line)
        s.append("}\n")
        return s

    def _film_style_light_leak_jsx(self, center_x, center_y, parent_var="PlotAnchor", apply_zoom=False):
        """Emit a looping light-leak video overlay above all chart content.
        Returns ``[]`` when the ``light_leak`` piece is disabled."""
        if not self._fp("light_leak"):
            return []
        fname = self._fp("light_leak_file")
        path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "film_elements", fname)
        ).replace("\\", "/").replace('"', '\\"')
        opacity = float(self._fp("light_leak_opacity"))
        blend = self._fp("light_leak_blend")
        s = [
            f'var leakFile = new File("{path}");\n',
            "if (leakFile.exists) {\n",
            "    var leakFootage = null;\n",
            "    var leakPath = leakFile.fsName.toLowerCase();\n",
            "    for (var __li = 1; __li <= app.project.numItems; __li++) {\n",
            "        var __lItem = app.project.item(__li);\n",
            "        if (__lItem instanceof FootageItem && __lItem.mainSource && __lItem.mainSource.file) {\n",
            "            if (__lItem.mainSource.file.fsName.toLowerCase() === leakPath) {\n",
            "                leakFootage = __lItem;\n",
            "                break;\n",
            "            }\n",
            "        }\n",
            "    }\n",
            "    if (!leakFootage) {\n",
            "        var leakImportOptions = new ImportOptions(leakFile);\n",
            "        leakFootage = app.project.importFile(leakImportOptions);\n",
            "    }\n",
            "    var leakLayer = comp.layers.add(leakFootage);\n",
            "    leakLayer.name = 'AEGraph_FilmLightLeak';\n",
            f"    leakLayer.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n",
            "    var leakScale = 100;\n",
            "    if (leakLayer.source && leakLayer.source.width > 0 && leakLayer.source.height > 0) {\n",
            f"        leakScale = Math.max(({self.comp_width} / leakLayer.source.width) * 100, ({self.comp_height} / leakLayer.source.height) * 100) / {self._film_style_min_zoom_fraction()};\n",
            "        leakLayer.property('Transform').property('Scale').setValue([leakScale, leakScale]);\n",
            "    }\n",
            f"    leakLayer.property('Transform').property('Opacity').setValue({opacity});\n",
            f"    leakLayer.blendingMode = BlendingMode.{blend};\n",
            "    try {\n",
            "        leakLayer.timeRemapEnabled = true;\n",
            "        leakLayer.property('Time Remap').expression = 'loopOut(\"cycle\")';\n",
            "    } catch (e) {}\n",
            "    leakLayer.moveToBeginning();\n",
        ]
        if parent_var:
            s.append(f"    leakLayer.parent = {parent_var};\n")
        if apply_zoom:
            for line in self._film_style_zoom_keyframes_jsx("leakLayer", scale_expr="leakScale"):
                s.append("    " + line)
        s.append("}\n")
        return s

    def _film_style_temporal_adjustment_jsx(self, center_x, center_y, parent_var=None, apply_zoom=False):
        """Global adjustment layer: Posterize Time (choppy frame-rate look)
        plus an Exposure flicker driven by a wiggle expression. Unparented
        and full-comp-sized by default so it always covers the whole frame,
        regardless of the push-in zoom applied to chart content. Returns
        ``[]`` when the ``temporal`` piece is disabled."""
        if not self._fp("temporal"):
            return []
        wiggle_expr = self._fp("exposure_wiggle")
        s = [
            f'var filmTimeAdj = comp.layers.addSolid([1,1,1], "AEGraph_FilmTime", {self.comp_width}, {self.comp_height}, 1.0);\n',
            "filmTimeAdj.adjustmentLayer = true;\n",
            f"filmTimeAdj.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n",
            "filmTimeAdj.moveToBeginning();\n",
        ]
        if self._fp("posterize_time"):
            fps_val = float(self._fp("posterize_fps"))
            s.extend([
                "var filmPosterize = filmTimeAdj.property('Effects').addProperty('ADBE Posterize Time');\n",
                "if (filmPosterize) {\n",
                f"    maybeSetValue(filmPosterize, 'ADBE Posterize Time-0001', {fps_val});\n",
                "}\n",
            ])
        s.extend([
            "var filmExposure = filmTimeAdj.property('Effects').addProperty('ADBE Exposure2');\n",
            "if (filmExposure) {\n",
            f"    try {{ filmExposure.property('ADBE Exposure2-0003').expression = {json.dumps(wiggle_expr)}; }} catch (e) {{}}\n",
            "}\n",
        ])
        if parent_var:
            s.append(f"filmTimeAdj.parent = {parent_var};\n")
        if apply_zoom:
            s.extend(self._film_style_zoom_keyframes_jsx("filmTimeAdj"))
        return s

    def _film_style_vignette_adjustment_jsx(self, center_x, center_y, parent_var=None, apply_zoom=False):
        """Global adjustment layer: Lumetri Color vignette. Unparented and
        full-comp-sized by default so it always covers the whole frame,
        regardless of the push-in zoom applied to chart content. Returns
        ``[]`` when the ``vignette`` piece is disabled."""
        if not self._fp("vignette"):
            return []
        amt = float(self._fp("vignette_amount"))
        mid = float(self._fp("vignette_midpoint"))
        rnd = float(self._fp("vignette_roundness"))
        feather = float(self._fp("vignette_feather"))
        s = [
            f'var filmVigAdj = comp.layers.addSolid([1,1,1], "AEGraph_FilmVignette", {self.comp_width}, {self.comp_height}, 1.0);\n',
            "filmVigAdj.adjustmentLayer = true;\n",
            f"filmVigAdj.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n",
            "filmVigAdj.moveToBeginning();\n",
            "var filmLumetri = filmVigAdj.property('Effects').addProperty('ADBE Lumetri');\n",
            "if (filmLumetri) {\n",
            "    try {\n",
            "        maybeSetValueDeep(filmLumetri, 'ADBE Lumetri-0050', 1);\n",
            f"        maybeSetValueDeep(filmLumetri, 'ADBE Lumetri-0051', {amt});\n",
            f"        maybeSetValueDeep(filmLumetri, 'ADBE Lumetri-0052', {mid});\n",
            f"        maybeSetValueDeep(filmLumetri, 'ADBE Lumetri-0053', {rnd});\n",
            f"        maybeSetValueDeep(filmLumetri, 'ADBE Lumetri-0054', {feather});\n",
            "    } catch (e) {}\n",
            "}\n",
        ]
        if parent_var:
            s.append(f"filmVigAdj.parent = {parent_var};\n")
        if apply_zoom:
            s.extend(self._film_style_zoom_keyframes_jsx("filmVigAdj"))
        return s

    def _film_style_edge_blur_adjustment_jsx(self, center_x, center_y, parent_var=None, apply_zoom=False):
        """Global adjustment layer: an *inverted*, feathered rectangular mask
        plus Gaussian Blur, so the blur is confined to a thin band near the
        true frame edge and the center stays sharp. ``margin``/``feather``
        are fractions of the comp's shorter side, so the look is consistent
        across comp sizes. Unparented and full-comp-sized by default so the
        blurred band always sits at the true frame edge, regardless of the
        push-in zoom applied to chart content. Returns ``[]`` when the
        ``edge_blur`` piece is disabled.
        """
        if not self._fp("edge_blur"):
            return []
        blur_amt = float(self._fp("edge_blur_amount"))
        margin_frac = float(self._fp("edge_blur_margin"))
        feather_frac = float(self._fp("edge_blur_feather"))
        w, h = self.comp_width, self.comp_height
        short_side = min(w, h)
        margin_px = margin_frac * short_side
        feather_px = feather_frac * short_side
        s = [
            f'var filmBlurAdj = comp.layers.addSolid([1,1,1], "AEGraph_FilmEdgeBlur", {w}, {h}, 1.0);\n',
            "filmBlurAdj.adjustmentLayer = true;\n",
            f"filmBlurAdj.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n",
            "filmBlurAdj.moveToBeginning();\n",
            "var filmBlurMask = filmBlurAdj.property('ADBE Mask Parade').addProperty('ADBE Mask Atom');\n",
            "if (filmBlurMask) {\n",
            f"    var __mw = {w} - 2 * {margin_px}, __mh = {h} - 2 * {margin_px}, __mx = {w} / 2, __my = {h} / 2;\n",
            "    var __maskShape = new Shape();\n",
            "    __maskShape.vertices = [[__mx - __mw/2, __my - __mh/2], [__mx + __mw/2, __my - __mh/2], [__mx + __mw/2, __my + __mh/2], [__mx - __mw/2, __my + __mh/2]];\n",
            "    __maskShape.inTangents = [[0,0],[0,0],[0,0],[0,0]];\n",
            "    __maskShape.outTangents = [[0,0],[0,0],[0,0],[0,0]];\n",
            "    __maskShape.closed = true;\n",
            "    filmBlurMask.property('ADBE Mask Shape').setValue(__maskShape);\n",
            f"    filmBlurMask.property('ADBE Mask Feather').setValue([{feather_px}, {feather_px}]);\n",
            "    filmBlurMask.inverted = true;\n",  # blur OUTSIDE the mask (i.e. near the true edge), not inside
            "}\n",
            "var filmEdgeBlur = filmBlurAdj.property('Effects').addProperty('ADBE Gaussian Blur 2');\n",
            "if (filmEdgeBlur) {\n",
            f"    maybeSetValue(filmEdgeBlur, 'ADBE Gaussian Blur 2-0001', {blur_amt});\n",
            "    maybeSetValue(filmEdgeBlur, 'ADBE Gaussian Blur 2-0003', 1);\n",
            "}\n",
        ]
        if parent_var:
            s.append(f"filmBlurAdj.parent = {parent_var};\n")
        if apply_zoom:
            s.extend(self._film_style_zoom_keyframes_jsx("filmBlurAdj"))
        return s

    def _film_style_element_pass_jsx(self):
        """Runtime pass (see ``_JSX_FILM_STYLE_FN``): applies Roughen Edges
        and/or Gaussian Blur and/or Multiply blending to every shape/text
        content layer currently in the comp, honoring the per-kind override
        dicts. Returns ``""`` when every piece is disabled."""
        roughen_on = bool(self._fp("element_roughen"))
        blur_on = bool(self._fp("element_blur"))
        multiply_on = bool(self._fp("element_multiply"))
        if not (roughen_on or blur_on or multiply_on):
            return ""
        opts = {
            "roughen": roughen_on,
            "roughenKinds": dict(self._fp("element_roughen_kinds") or {}),
            "roughenEdgeType": self._fp("roughen_edge_type"),
            "roughenEdgeColor": list(self._fp("roughen_edge_color")),
            "roughenBorder": self._fp("roughen_border"),
            "roughenSharpness": self._fp("roughen_sharpness"),
            "roughenScale": self._fp("roughen_scale"),
            "blur": blur_on,
            "blurKinds": dict(self._fp("element_blur_kinds") or {}),
            "blurAmount": self._fp("element_blur_amount"),
            "multiply": multiply_on,
            "multiplyKinds": dict(self._fp("element_multiply_kinds") or {}),
        }
        return f"__runFilmStyleElementPass(comp, {json.dumps(opts)});\n"

    def _generate_jsx(self, include_preamble: bool = True, origin=None,
                      wrap_iife: bool = False, ns: str = "",
                      draw_bg: bool = True, draw_distress: bool = True,
                      draw_film_style: bool = True) -> str:
        """Generate the JSX for this graph.

        Args:
            include_preamble: Emit the shared helper functions and the
                find-or-create-comp header. AEFigure sets this False for panels
                (it emits the preamble once for the whole figure).
            origin: Optional ``(x, y)`` comp-pixel center for this graph's
                plotting region. Defaults to ``self.position`` when set, else
                the comp center. This is the choke point that lets a figure
                place panels anywhere in a shared comp.
            wrap_iife: Wrap the body in ``(function(){ ... })();`` so all the
                generated ``var`` names are function-scoped and never collide
                with sibling panels in the same comp.
            ns: Namespace suffix appended to AE layer *names* (e.g. PlotAnchor)
                so panels are distinguishable in the timeline.
            draw_bg: Draw this graph's background rectangle. AEFigure sets this
                False for panels and instead paints one comp-wide background, so
                tiled panels don't cut each other's backgrounds off.
            draw_distress: Emit this graph's distress/grunge texture layer.
                AEFigure sets this False for panels and instead lays down a
                single shared texture, so a multi-panel comp doesn't stack one
                grunge layer per panel.
            draw_film_style: Emit this graph's own paper background, light-leak
                overlay, global adjustment stack, and per-element pass. AEFigure
                sets this False for panels and instead emits those pieces once,
                comp-wide (see ``AEFigure._figure_film_style_jsx``); each panel
                still gets its own push-in zoom keyframes regardless of this flag.
        """
        ANIM_DURATION = self.text_animate  # seconds for text slide-in animations
        preamble = []
        script = []

        if include_preamble:
            if self.easy_ease:
                preamble.append(_JSX_EASY_EASE_FN)
            preamble.append(_JSX_HELPER_FUNCTIONS)
            if self.film_style:
                preamble.append(_JSX_FILM_STYLE_FN)
            preamble.append(_jsx_comp_header(self.comp_name, self.comp_width, self.comp_height, self.fps))

        # Ensure the comp is long enough to contain every evolving keyframe.
        evolving_end = 0.0
        for _elem in self.elements:
            if _elem.get("type") in ("line_evolving", "scatter_evolving", "quiver_evolving", "heatmap_evolving"):
                _times = _elem.get("frame_times") or [0.0]
                evolving_end = max(evolving_end, max(_times) + _elem.get("fade_in", 0.0))
            elif _elem.get("type") == "barh_evolving":
                _times = _elem.get("frame_times") or [0.0]
                evolving_end = max(evolving_end, max(_times))
            elif _elem.get("type") == "evolving_text":
                _times = _elem.get("frame_times") or [0.0]
                evolving_end = max(evolving_end, max(_times) + _elem.get("fade_in", 0.0))
        if self._view_animated and self._view_kf:
            evolving_end = max(evolving_end, self._view_kf[-1][0])
        if evolving_end > 0:
            needed_duration = round(evolving_end + 1.0, 3)
            script.append(
                f"if (comp.duration < {needed_duration}) comp.duration = {needed_duration};\n"
            )
        # Remove comp background color setting - let it be transparent
        # script.append(f"comp.bgColor = {color_to_js(self.bg_color)};\n")

        # --- Plot region center ---
        # Layers are placed at (center_x + sx, center_y + sy) in comp pixels and
        # then parented to PlotAnchor. Because AE preserves world position on
        # parenting, the *region* is defined by center_x/center_y here (not by
        # PlotAnchor's position). `origin` lets a figure offset a panel; falling
        # back to self.position keeps a standalone graph's position meaningful.
        if origin is not None:
            center_x, center_y = float(origin[0]), float(origin[1])
        elif self.position is not None:
            center_x, center_y = float(self.position[0]), float(self.position[1])
        else:
            center_x = self.comp_width / 2
            center_y = self.comp_height / 2

        # --- PlotAnchor Null Layer ---
        anchor_x, anchor_y = center_x, center_y
        script.append(f"var PlotAnchor = comp.layers.addNull();\n")
        script.append(f"PlotAnchor.name = 'PlotAnchor{ns}';\n")
        script.append(f"PlotAnchor.property('Transform').property('Position').setValue([{anchor_x}, {anchor_y}]);\n")
        script.append(f"PlotAnchor.moveToBeginning();\n")
        self._current_plotanchor_name = f"PlotAnchor{ns}"
        self._screen_mask_counter = 0

        # How the distress layer is restacked above the background. Standalone
        # graphs draw their own `bgLayer` (declared just above) and move before
        # it. Figure panels have no per-panel background, so they move relative
        # to the shared 'FigureBG' looked up in *this* comp. We deliberately do
        # NOT use `typeof bgLayer` here: After Effects' persistent ExtendScript
        # engine keeps `bgLayer` from a previous render as a global, and that
        # stale layer lives in another comp -> "unable to move a layer in
        # another composition".
        if draw_bg:
            distress_move_line = "    distressLayer.moveBefore(bgLayer);\n"
            paper_move_line = "    paperLayer.moveBefore(bgLayer);\n"
        else:
            distress_move_line = (
                "    var __figBG = null;\n"
                "    try { __figBG = comp.layer('FigureBG'); } catch (e) { __figBG = null; }\n"
                "    if (__figBG) { distressLayer.moveBefore(__figBG); }\n"
            )
            paper_move_line = (
                "    var __figBG = null;\n"
                "    try { __figBG = comp.layer('FigureBG'); } catch (e) { __figBG = null; }\n"
                "    if (__figBG) { paperLayer.moveBefore(__figBG); }\n"
            )

        # Optional background rectangle (skipped when bg_color == 'none', or
        # when draw_bg is False because a figure paints one shared background)
        if draw_bg and not (isinstance(self.bg_color, str) and self.bg_color.lower() == "none"):
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

        if self.film_style and draw_film_style:
            script.extend(self._film_style_paper_jsx(center_x, center_y, paper_move_line))

        if draw_distress:
            script.extend(self._distress_block_jsx(center_x, center_y, distress_move_line))


        # In _generate_jsx, calculate global data limits with NO padding
        all_x = []
        all_y = []
        for elem in self.elements:
            if elem["type"] in ["line", "scatter", "line_evolving", "scatter_evolving", "quiver", "quiver_evolving", "heatmap", "heatmap_evolving"]:
                all_x.extend(elem["x"])
                all_y.extend(elem["y"])
            elif elem["type"] in ["histogram", "bar_graph", "bar_stacked", "bar_evolving"]:
                # Use bin edges for x, heights for y
                all_x.extend(elem["bin_centers"])
                all_x.extend(elem["bin_left"])
                all_x.extend(elem["bin_right"])
                _base = elem.get("baseline")
                if _base:
                    tops = [b + h for b, h in zip(_base, elem["heights"])]
                    all_y.extend(tops)
                    all_y.extend(_base)
                else:
                    all_y.extend(elem["heights"])
                all_y.append(0)  # include baseline
                # Pad each side of the x-axis by the inter-bar gap so the space
                # between the outermost bars and the frame matches the spacing
                # between bars (mirrors the barh logic below). Histograms have
                # contiguous bins (bar_width == spacing), so the gap is 0 and
                # their bars stay flush, as expected.
                if len(elem["bin_left"]) > 1:
                    centers = sorted(elem["bin_centers"])
                    spacing = centers[1] - centers[0]  # distance between bar centers
                    bar_width = elem["bin_right"][0] - elem["bin_left"][0]
                    gap = spacing - bar_width  # gap between adjacent bars
                    # Honor per-element x_pad override; fall back to the
                    # inter-bar gap so edge spacing mirrors bar spacing.
                    edge = gap if elem.get("x_pad") is None else float(elem["x_pad"])
                    all_x.append(min(elem["bin_left"]) - edge)
                    all_x.append(max(elem["bin_right"]) + edge)
            elif elem["type"] in ("barh", "barh_evolving"):
                # Use widths for x, bin edges for y
                _base = elem.get("baseline")
                if _base:
                    all_x.extend([b + w for b, w in zip(_base, elem["widths"])])
                    all_x.extend(_base)
                else:
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
            elif elem["type"] == "band":
                all_x.extend(elem["x"])
                all_y.extend(elem["y_low"])
                all_y.extend(elem["y_high"])
            elif elem["type"] == "errorbar":
                all_x.extend(elem["x"])
                ys = np.asarray(elem["y"], dtype=float)
                yerr = elem.get("yerr")
                if yerr is not None and len(ys):
                    yerr = np.asarray(yerr, dtype=float)
                    all_y.extend(list(ys - yerr))
                    all_y.extend(list(ys + yerr))
                else:
                    all_y.extend(list(ys))
                xerr = elem.get("xerr")
                if xerr is not None and len(elem["x"]):
                    xs = np.asarray(elem["x"], dtype=float)
                    xerr = np.asarray(xerr, dtype=float)
                    all_x.extend(list(xs - xerr))
                    all_x.extend(list(xs + xerr))
            elif elem["type"] == "refline":
                if elem.get("orient") == "v":
                    all_x.append(elem["value"])
                else:
                    all_y.append(elem["value"])
        if self.xlim:
            xmin, xmax = self.xlim
        elif all_x:
            xmin, xmax = min(all_x), max(all_x)
        else:
            xmin, xmax = (1.0, 10.0) if self._normalize_scale(self.xscale) == "log" else (0.0, 1.0)
        if self.ylim:
            ymin, ymax = self.ylim
        elif all_y:
            ymin, ymax = min(all_y), max(all_y)
        else:
            ymin, ymax = (1.0, 10.0) if self._normalize_scale(self.yscale) == "log" else (0.0, 1.0)
        if self._normalize_scale(self.xscale) == "log":
            positive_x = [v for v in all_x if v > 0]
            xmin, xmax = self._sanitize_log_limits(xmin, xmax, positive_x or all_x)
        if self._normalize_scale(self.yscale) == "log":
            positive_y = [v for v in all_y if v > 0]
            ymin, ymax = self._sanitize_log_limits(ymin, ymax, positive_y or all_y)
        # Auto-pad limits when caller didn't pin them. Linear: expand the span
        # by `lim_factor` (10% each side at 1.2). Log: equivalent multiplicative
        # expansion in log space (sqrt(lim_factor) factor each side).
        #
        # Padding for bar-like plots is asymmetric: the side that bars anchor
        # to (y=0 for histograms / bar_graphs, x=0 for barh) stays tight so
        # bars kiss the axis, but the opposite "tip" side gets the same
        # per-side breathing room ``_auto_pad_limits`` would normally apply
        # — i.e. ``(lim_factor - 1) / 2 * span``. For scatter-only / line-only
        # plots, the original bidirectional padding is kept.
        has_bars_or_hist = any(
            elem["type"] in ("histogram", "bar_graph", "bar_stacked", "bar_evolving", "barh", "barh_evolving") for elem in self.elements
        )
        # Heatmaps fill the plot area exactly, so they should sit flush against
        # the frame (no scatter-style breathing room). Reuse the tight-padding
        # branch; the bar-specific padding loops below are no-ops without bars.
        has_heatmap = any(
            elem["type"] in ("heatmap", "heatmap_evolving") for elem in self.elements
        )
        if has_bars_or_hist or has_heatmap:
            xmin_pad, xmax_pad = xmin, xmax
            ymin_pad, ymax_pad = ymin, ymax

            auto_pad_on = getattr(config, "auto_limits", True)

            # Vertical bars / histograms: pad the y-axis tip side.
            if auto_pad_on and self.ylim is None:
                v_heights = []
                for elem in self.elements:
                    if elem["type"] in ("histogram", "bar_graph", "bar_stacked", "bar_evolving"):
                        v_heights.extend(elem["heights"])
                if v_heights:
                    full_ymin, full_ymax = self._auto_pad_limits(
                        ymin, ymax, None, self.yscale
                    )
                    if max(v_heights) >= abs(min(v_heights)):
                        ymax_pad = full_ymax  # bars grow up → pad top
                    else:
                        ymin_pad = full_ymin  # bars grow down → pad bottom

            # Horizontal bars (barh): pad the x-axis tip side.
            if auto_pad_on and self.xlim is None:
                h_widths = []
                for elem in self.elements:
                    if elem["type"] in ("barh", "barh_evolving"):
                        h_widths.extend(elem["widths"])
                if h_widths:
                    full_xmin, full_xmax = self._auto_pad_limits(
                        xmin, xmax, None, self.xscale
                    )
                    if max(h_widths) >= abs(min(h_widths)):
                        xmax_pad = full_xmax  # bars grow right → pad right
                    else:
                        xmin_pad = full_xmin  # bars grow left → pad left
        else:
            xmin_pad, xmax_pad = self._auto_pad_limits(xmin, xmax, self.xlim, self.xscale)
            ymin_pad, ymax_pad = self._auto_pad_limits(ymin, ymax, self.ylim, self.yscale)
        if self._view_animated:
            # Animated view window: ignore data-extent padding entirely. The
            # padded limits become the *reference* (first-keyframe) view used
            # by any static fallback emission; every coordinate-driven layer is
            # additionally keyframed via _view_limits_at() during the element
            # loop. Tick values are fixed over the union of all keyframe views.
            xmin_pad, xmax_pad, ymin_pad, ymax_pad = self._view_limits_at(self._view_kf[0][0])
            self._view_set_fixed_ticks()
            _unsupported = {
                "bar_stacked", "histogram", "barh", "bar_evolving",
                "barh_evolving", "heatmap", "heatmap_evolving", "quiver",
                "quiver_evolving", "pie",
            }
            _present_unsupported = sorted({e["type"] for e in self.elements if e["type"] in _unsupported})
            if _present_unsupported:
                raise NotImplementedError(
                    "Animated view bounds (view_keyframes) are not yet supported for "
                    f"these plot types: {', '.join(_present_unsupported)}. Supported v1 "
                    "types: line, scatter, line_evolving, scatter_evolving, bar_graph, "
                    "band, refline, errorbar, and annotations."
                )
        # Auto-generate ticks over the SAME padded range the grid uses so every
        # grid line has a matching tick + label (and vice versa).
        if self.xticks is None:
            self._auto_set_ticks_from_padded("x", xmin_pad, xmax_pad)
        if self.yticks is None:
            self._auto_set_ticks_from_padded("y", ymin_pad, ymax_pad)
        # Check what plot types we have to decide on axes placement
        has_scatter = any(elem["type"] in ("scatter", "scatter_evolving") for elem in self.elements)
        has_barh = any(elem["type"] in ("barh", "barh_evolving") for elem in self.elements)
        # Pie/donut charts are polar and carry no x/y axes, ticks, or grid.
        pie_only = bool(self.elements) and all(elem["type"] == "pie" for elem in self.elements)

        # For grid lines, fade in with opacity (GENERATED FIRST so grid appears behind plot elements)
        if self.show_grid and not pie_only:
            grid_color_js = color_to_js(self.grid_color)

            # Vertical grid lines
            if not self.hide_vertical:
                _grid_v_vals = ([pos for pos, _ in self.xticks] if self._view_animated and self.xticks
                                else self._nice_ticks_for_axis(xmin_pad, xmax_pad, scale=self.xscale))
                for idx, i in enumerate(_grid_v_vals):
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
                    if self._view_animated:
                        self._emit_view_path_kf(
                            script, f"gridVPath{grid_var}", f"gv{grid_var}",
                            lambda xmn, xmx, ymn, ymx, _i=i: [
                                self._data_to_shape(_i, ymn, xmn, xmx, ymn, ymx),
                                self._data_to_shape(_i, ymx, xmn, xmx, ymn, ymx),
                            ],
                            closed=False,
                        )
                    script.append(f"var gridVStroke{grid_var} = gridVContents{grid_var}.addProperty('ADBE Vector Graphic - Stroke');\n")
                    script.append(f"gridVStroke{grid_var}.property('ADBE Vector Stroke Color').setValue({grid_color_js});\n")
                    script.append(f"gridVStroke{grid_var}.property('ADBE Vector Stroke Width').setValue({float(self.grid_linewidth)});\n")
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
                    _grid_v_op = f"gridVStroke{grid_var}.property('ADBE Vector Stroke Opacity')"
                    if self._view_animated:
                        _grid_entrance = ANIM_DURATION * (0.5 + 0.5 * idx / 10)
                        self._emit_axis_visibility_kf(
                            script, _grid_v_op, i, "x", int(self.grid_alpha * 100),
                            entrance_start=0.0, entrance_fade=_grid_entrance,
                        )
                    else:
                        script.append(f"{_grid_v_op}.setValue(0);\n")
                        script.append(f"{_grid_v_op}.setValueAtTime(0, 0);\n")
                        script.append(f"{_grid_v_op}.setValueAtTime({ANIM_DURATION * (0.5 + 0.5 * idx / 10)}, {int(self.grid_alpha * 100)});\n")
                        if self.easy_ease:
                            script.append(f"applyEasyEase({_grid_v_op}, {self.ease_speed}, {self.ease_influence});\n")

            # Horizontal grid lines
            if not self.hide_horizontal:
                _grid_h_vals = ([pos for pos, _ in self.yticks] if self._view_animated and self.yticks
                                else self._nice_ticks_for_axis(ymin_pad, ymax_pad, scale=self.yscale))
                for idx, i in enumerate(_grid_h_vals):
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
                    if self._view_animated:
                        self._emit_view_path_kf(
                            script, f"gridHPath{grid_var}", f"gh{grid_var}",
                            lambda xmn, xmx, ymn, ymx, _i=i: [
                                self._data_to_shape(xmn, _i, xmn, xmx, ymn, ymx),
                                self._data_to_shape(xmx, _i, xmn, xmx, ymn, ymx),
                            ],
                            closed=False,
                        )
                    script.append(f"var gridHStroke{grid_var} = gridHContents{grid_var}.addProperty('ADBE Vector Graphic - Stroke');\n")
                    script.append(f"gridHStroke{grid_var}.property('ADBE Vector Stroke Color').setValue({grid_color_js});\n")
                    script.append(f"gridHStroke{grid_var}.property('ADBE Vector Stroke Width').setValue({float(self.grid_linewidth)});\n")
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
                    _grid_h_op = f"gridHStroke{grid_var}.property('ADBE Vector Stroke Opacity')"
                    if self._view_animated:
                        _grid_entrance = ANIM_DURATION * (0.5 + 0.5 * idx / 10)
                        self._emit_axis_visibility_kf(
                            script, _grid_h_op, i, "y", int(self.grid_alpha * 100),
                            entrance_start=0.0, entrance_fade=_grid_entrance,
                        )
                    else:
                        script.append(f"{_grid_h_op}.setValue(0);\n")
                        script.append(f"{_grid_h_op}.setValueAtTime(0, 0);\n")
                        script.append(f"{_grid_h_op}.setValueAtTime({ANIM_DURATION * (0.5 + 0.5 * idx / 10)}, {int(self.grid_alpha * 100)});\n")
                        if self.easy_ease:
                            script.append(f"applyEasyEase({_grid_h_op}, {self.ease_speed}, {self.ease_influence});\n")

        # Generate axes BEFORE plot elements ONLY for scatter plots (so axes appear behind points)
        # For histograms and bar graphs, axes will be generated AFTER plot elements (so they appear on top)
        if has_scatter and not has_bars_or_hist and not pie_only:
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
                self._emit_screen_space_mask_jsx(script, f"lineLayer{i}")
                script.append(f"var contents{i} = lineLayer{i}.property('ADBE Root Vectors Group');\n")
                script.append(f"var pathGroup{i} = contents{i}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var path{i} = pathGroup{i}.property('ADBE Vector Shape');\n")
                script.append(f"var shape{i} = new Shape();\n")
                script.append(f"shape{i}.vertices = [{points_js}];\n")
                script.append(f"shape{i}.closed = false;\n")
                script.append(f"path{i}.setValue(shape{i});\n")
                if self._view_animated:
                    self._emit_view_path_kf(
                        script, f"path{i}", f"line{i}",
                        lambda xmn, xmx, ymn, ymx, _px=px, _py=py: [
                            self._data_to_shape(x, y, xmn, xmx, ymn, ymx) for x, y in zip(_px, _py)
                        ],
                        closed=False,
                    )
                script.append(f"var stroke{i} = contents{i}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"stroke{i}.property('ADBE Vector Stroke Color').setValue({color_js});\n")
                script.append(f"stroke{i}.property('ADBE Vector Stroke Width').setValue({elem['linewidth']});\n")
                line_alpha = float(elem.get("alpha", 1.0))
                if line_alpha < 1.0:
                    script.append(f"stroke{i}.property('ADBE Vector Stroke Opacity').setValue({line_alpha * 100});\n")
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
            elif elem["type"] == "band":
                bx = elem["x"]
                y_low = elem["y_low"]; y_high = elem["y_high"]
                # Closed polygon: upper boundary left->right, then lower right->left.
                upper = [self._data_to_shape(x, yh, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                         for x, yh in zip(bx, y_high)]
                lower = [self._data_to_shape(x, yl, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                         for x, yl in zip(bx, y_low)]
                verts = upper + lower[::-1]
                pts_js = ",".join(f"[{vx},{vy}]" for vx, vy in verts)
                fill_js = color_to_js(elem["color"])
                script.append(f"var bandLayer{i} = comp.layers.addShape();\n")
                script.append(f"bandLayer{i}.name = 'Band_{i}';\n")
                script.append(f"bandLayer{i}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"bandLayer{i}.parent = PlotAnchor;\n")
                self._emit_screen_space_mask_jsx(script, f"bandLayer{i}")
                script.append(f"var bandContents{i} = bandLayer{i}.property('ADBE Root Vectors Group');\n")
                script.append(f"var bandGrp{i} = bandContents{i}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var bandPath{i} = bandGrp{i}.property('ADBE Vector Shape');\n")
                script.append(f"var bandShp{i} = new Shape();\n")
                script.append(f"bandShp{i}.vertices = [{pts_js}];\n")
                script.append(f"bandShp{i}.closed = true;\n")
                script.append(f"bandPath{i}.setValue(bandShp{i});\n")
                if self._view_animated:
                    self._emit_view_path_kf(
                        script, f"bandPath{i}", f"band{i}",
                        lambda xmn, xmx, ymn, ymx, _bx=bx, _yl=y_low, _yh=y_high: (
                            [self._data_to_shape(x, yh, xmn, xmx, ymn, ymx) for x, yh in zip(_bx, _yh)]
                            + [self._data_to_shape(x, yl, xmn, xmx, ymn, ymx) for x, yl in zip(_bx, _yl)][::-1]
                        ),
                        closed=True,
                    )
                script.append(f"var bandFill{i} = bandContents{i}.addProperty('ADBE Vector Graphic - Fill');\n")
                script.append(f"bandFill{i}.property('ADBE Vector Fill Color').setValue({fill_js});\n")
                script.append(f"bandFill{i}.property('ADBE Vector Fill Opacity').setValue({int(float(elem['alpha']) * 100)});\n")
                if elem.get("edge"):
                    edge_js = color_to_js(elem["edge_color"])
                    script.append(f"var bandStroke{i} = bandContents{i}.addProperty('ADBE Vector Graphic - Stroke');\n")
                    script.append(f"bandStroke{i}.property('ADBE Vector Stroke Color').setValue({edge_js});\n")
                    script.append(f"bandStroke{i}.property('ADBE Vector Stroke Width').setValue({elem.get('edge_width', 2)});\n")
                anim = elem.get("animate") or 0.0
                if anim and anim > 0:
                    op = f"bandLayer{i}.property('Transform').property('Opacity')"
                    script.append(f"{op}.setValueAtTime({delay}, 0);\n")
                    script.append(f"{op}.setValueAtTime({delay + anim}, 100);\n")
                    if self.easy_ease:
                        script.append(f"applyEasyEase({op}, {self.ease_speed}, {self.ease_influence});\n")
            elif elem["type"] == "gradient_area":
                gx = elem["x"]
                gy_top, gy_bottom = elem["y_top"], elem["y_bottom"]
                grad_top_val, grad_bot_val = elem["gradient_top"], elem["gradient_bottom"]
                # Shape geometry follows the curve (upper boundary left->right,
                # then lower boundary back right->left) -- identical to
                # `band`/`fill_between`. The gradient itself stays a plain
                # straight vertical ramp anchored to fixed data y-values
                # (gradient_top/gradient_bottom), independent of the curve.
                upper = [self._data_to_shape(x, yt, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                         for x, yt in zip(gx, gy_top)]
                lower = [self._data_to_shape(x, yb, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                         for x, yb in zip(gx, gy_bottom)]
                verts = upper + lower[::-1]
                pts_js = ",".join(f"[{vx},{vy}]" for vx, vy in verts)
                top_js = color_to_js(elem["top_color"])
                bottom_js = color_to_js(elem["bottom_color"])
                grad_mid_x = (min(gx) + max(gx)) / 2.0
                ramp_start = self._data_to_shape(grad_mid_x, grad_top_val, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                ramp_end = self._data_to_shape(grad_mid_x, grad_bot_val, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                script.append(f"var gradLayer{i} = comp.layers.addShape();\n")
                script.append(f"gradLayer{i}.name = 'GradientArea_{i}';\n")
                script.append(f"gradLayer{i}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"gradLayer{i}.parent = PlotAnchor;\n")
                self._emit_screen_space_mask_jsx(script, f"gradLayer{i}")
                script.append(f"var gradContents{i} = gradLayer{i}.property('ADBE Root Vectors Group');\n")
                script.append(f"var gradGrp{i} = gradContents{i}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var gradPath{i} = gradGrp{i}.property('ADBE Vector Shape');\n")
                script.append(f"var gradShp{i} = new Shape();\n")
                script.append(f"gradShp{i}.vertices = [{pts_js}];\n")
                script.append(f"gradShp{i}.closed = true;\n")
                script.append(f"gradPath{i}.setValue(gradShp{i});\n")
                if self._view_animated:
                    self._emit_view_path_kf(
                        script, f"gradPath{i}", f"grad{i}",
                        lambda xmn, xmx, ymn, ymx, _gx=gx, _yt=gy_top, _yb=gy_bottom: (
                            [self._data_to_shape(x, yt, xmn, xmx, ymn, ymx) for x, yt in zip(_gx, _yt)]
                            + [self._data_to_shape(x, yb, xmn, xmx, ymn, ymx) for x, yb in zip(_gx, _yb)][::-1]
                        ),
                        closed=True,
                    )
                # A plain vector fill just establishes the shape's alpha
                # silhouette; the Ramp effect below paints over it with the
                # actual gradient colors (AE's Generate effects render only
                # within the layer's existing alpha).
                script.append(f"var gradFill{i} = gradContents{i}.addProperty('ADBE Vector Graphic - Fill');\n")
                script.append(f"gradFill{i}.property('ADBE Vector Fill Color').setValue({top_js});\n")
                script.append(f"gradFill{i}.property('ADBE Vector Fill Opacity').setValue(100);\n")
                script.append(f"var gradRamp{i} = gradLayer{i}.property('Effects').addProperty('ADBE Ramp');\n")
                script.append(f"gradRamp{i}.property('Start Color').setValue({top_js});\n")
                script.append(f"gradRamp{i}.property('End Color').setValue({bottom_js});\n")
                script.append(f"gradRamp{i}.property('Ramp Shape').setValue(1);\n")
                if float(elem.get("ramp_scatter", 0.0)):
                    script.append(f"gradRamp{i}.property('Ramp Scatter').setValue({float(elem['ramp_scatter'])});\n")
                script.append(f"gradRamp{i}.property('Start of Ramp').setValue([{ramp_start[0]}, {ramp_start[1]}]);\n")
                script.append(f"gradRamp{i}.property('End of Ramp').setValue([{ramp_end[0]}, {ramp_end[1]}]);\n")
                if self._view_animated:
                    self._emit_view_pos_kf(
                        script, f"gradRamp{i}.property('Start of Ramp')",
                        lambda xmn, xmx, ymn, ymx, _mx=grad_mid_x, _gt=grad_top_val: (
                            self._data_to_shape(_mx, _gt, xmn, xmx, ymn, ymx)
                        ),
                    )
                    self._emit_view_pos_kf(
                        script, f"gradRamp{i}.property('End of Ramp')",
                        lambda xmn, xmx, ymn, ymx, _mx=grad_mid_x, _gb=grad_bot_val: (
                            self._data_to_shape(_mx, _gb, xmn, xmx, ymn, ymx)
                        ),
                    )
                grad_alpha = float(elem.get("alpha", 1.0))
                anim = elem.get("animate") or 0.0
                op = f"gradLayer{i}.property('Transform').property('Opacity')"
                if anim and anim > 0:
                    script.append(f"{op}.setValueAtTime({delay}, 0);\n")
                    script.append(f"{op}.setValueAtTime({delay + anim}, {grad_alpha * 100});\n")
                    if self.easy_ease:
                        script.append(f"applyEasyEase({op}, {self.ease_speed}, {self.ease_influence});\n")
                elif grad_alpha < 1.0:
                    script.append(f"{op}.setValue({grad_alpha * 100});\n")
            elif elem["type"] == "errorbar":
                ex = elem["x"]; ey = elem["y"]
                yerr = elem.get("yerr"); xerr = elem.get("xerr")
                cap = float(elem.get("capsize", 6.0)) / 2.0
                color_js = color_to_js(elem["color"])
                script.append(f"var errLayer{i} = comp.layers.addShape();\n")
                script.append(f"errLayer{i}.name = 'ErrorBar_{i}';\n")
                script.append(f"errLayer{i}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"errLayer{i}.parent = PlotAnchor;\n")
                self._emit_screen_space_mask_jsx(script, f"errLayer{i}")
                script.append(f"var errContents{i} = errLayer{i}.property('ADBE Root Vectors Group');\n")
                seg = 0

                # Each segment is described in a view-independent way so it can
                # be re-mapped per keyframe: two data-space endpoints plus fixed
                # pixel offsets (caps) applied after the data->shape mapping.
                # (da, db) are data points; (oa, ob) are (dx, dy) pixel offsets.
                def _err_seg_verts(da, db, oa, ob, xmn, xmx, ymn, ymx):
                    sa = self._data_to_shape(da[0], da[1], xmn, xmx, ymn, ymx)
                    sb = self._data_to_shape(db[0], db[1], xmn, xmx, ymn, ymx)
                    return (sa[0] + oa[0], sa[1] + oa[1]), (sb[0] + ob[0], sb[1] + ob[1])

                segments = []  # list of (da, db, oa, ob)
                for j in range(len(ex)):
                    if yerr is not None:
                        e = yerr[j]
                        lo = (ex[j], ey[j] - e); hi = (ex[j], ey[j] + e)
                        segments.append((lo, hi, (0, 0), (0, 0)))
                        segments.append((lo, lo, (-cap, 0), (cap, 0)))
                        segments.append((hi, hi, (-cap, 0), (cap, 0)))
                    if xerr is not None:
                        e = xerr[j]
                        lo = (ex[j] - e, ey[j]); hi = (ex[j] + e, ey[j])
                        segments.append((lo, hi, (0, 0), (0, 0)))
                        segments.append((lo, lo, (0, -cap), (0, cap)))
                        segments.append((hi, hi, (0, -cap), (0, cap)))

                for da, db, oa, ob in segments:
                    gtag = f"{i}_{seg}"
                    script.append(f"var eg{gtag} = errContents{i}.addProperty('ADBE Vector Shape - Group');\n")
                    script.append(f"var ep{gtag} = eg{gtag}.property('ADBE Vector Shape');\n")
                    a, b = _err_seg_verts(da, db, oa, ob, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    script.append(f"var es{gtag} = new Shape();\n")
                    script.append(f"es{gtag}.vertices = [[{a[0]},{a[1]}],[{b[0]},{b[1]}]];\n")
                    script.append(f"es{gtag}.closed = false;\n")
                    script.append(f"ep{gtag}.setValue(es{gtag});\n")
                    if self._view_animated:
                        self._emit_view_path_kf(
                            script, f"ep{gtag}", f"err{gtag}",
                            lambda xmn, xmx, ymn, ymx, _da=da, _db=db, _oa=oa, _ob=ob: list(
                                _err_seg_verts(_da, _db, _oa, _ob, xmn, xmx, ymn, ymx)
                            ),
                            closed=False,
                        )
                    seg += 1
                script.append(f"var errStroke{i} = errContents{i}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"errStroke{i}.property('ADBE Vector Stroke Color').setValue({color_js});\n")
                script.append(f"errStroke{i}.property('ADBE Vector Stroke Width').setValue({elem.get('linewidth', 2)});\n")
                err_alpha = float(elem.get("alpha", 1.0))
                if err_alpha < 1.0:
                    script.append(f"errStroke{i}.property('ADBE Vector Stroke Opacity').setValue({err_alpha * 100});\n")
                anim = elem.get("animate") or 0.0
                if anim and anim > 0:
                    op = f"errLayer{i}.property('Transform').property('Opacity')"
                    script.append(f"{op}.setValueAtTime({delay}, 0);\n")
                    script.append(f"{op}.setValueAtTime({delay + anim}, 100);\n")
                    if self.easy_ease:
                        script.append(f"applyEasyEase({op}, {self.ease_speed}, {self.ease_influence});\n")
            elif elem["type"] == "refline":
                orient = elem.get("orient", "h")
                val = elem["value"]
                if orient == "h":
                    p0 = self._data_to_shape(xmin_pad, val, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    p1 = self._data_to_shape(xmax_pad, val, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                else:
                    p0 = self._data_to_shape(val, ymin_pad, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    p1 = self._data_to_shape(val, ymax_pad, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                color_js = color_to_js(elem["color"])
                script.append(f"var refLayer{i} = comp.layers.addShape();\n")
                script.append(f"refLayer{i}.name = 'RefLine_{i}';\n")
                script.append(f"refLayer{i}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"refLayer{i}.parent = PlotAnchor;\n")
                self._emit_screen_space_mask_jsx(script, f"refLayer{i}")
                script.append(f"var refContents{i} = refLayer{i}.property('ADBE Root Vectors Group');\n")
                script.append(f"var refGrp{i} = refContents{i}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var refPath{i} = refGrp{i}.property('ADBE Vector Shape');\n")
                script.append(f"var refShp{i} = new Shape();\n")
                script.append(f"refShp{i}.vertices = [[{p0[0]},{p0[1]}],[{p1[0]},{p1[1]}]];\n")
                script.append(f"refShp{i}.closed = false;\n")
                script.append(f"refPath{i}.setValue(refShp{i});\n")
                if self._view_animated:
                    def _refline_verts(xmn, xmx, ymn, ymx, _orient=orient, _val=val):
                        if _orient == "h":
                            return [self._data_to_shape(xmn, _val, xmn, xmx, ymn, ymx),
                                    self._data_to_shape(xmx, _val, xmn, xmx, ymn, ymx)]
                        return [self._data_to_shape(_val, ymn, xmn, xmx, ymn, ymx),
                                self._data_to_shape(_val, ymx, xmn, xmx, ymn, ymx)]
                    self._emit_view_path_kf(script, f"refPath{i}", f"ref{i}", _refline_verts, closed=False)
                script.append(f"var refStroke{i} = refContents{i}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"refStroke{i}.property('ADBE Vector Stroke Color').setValue({color_js});\n")
                script.append(f"refStroke{i}.property('ADBE Vector Stroke Width').setValue({elem.get('linewidth', 3)});\n")
                ref_alpha = float(elem.get("alpha", 1.0))
                if ref_alpha < 1.0:
                    script.append(f"refStroke{i}.property('ADBE Vector Stroke Opacity').setValue({ref_alpha * 100});\n")
                linestyle = elem.get("linestyle", "dashed")
                if linestyle in ("dashed", "--", "dotted", ":"):
                    dash_val, gap_val = self._get_dash_values(
                        "dashed" if linestyle in ("dashed", "--") else "dotted", 1.0)
                    script.append(f"refStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({dash_val});\n")
                    script.append(f"refStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({gap_val});\n")
                anim = elem.get("animate") or 0.0
                if anim and anim > 0:
                    script.append(f"var refTrim{i} = refContents{i}.addProperty('ADBE Vector Filter - Trim');\n")
                    script.append(f"var refEnd{i} = refTrim{i}.property('ADBE Vector Trim End');\n")
                    script.append(f"refEnd{i}.setValueAtTime({delay}, 0);\n")
                    script.append(f"refEnd{i}.setValueAtTime({delay + anim}, 100);\n")
                    if self.easy_ease:
                        script.append(f"applyEasyEase(refEnd{i}, {self.ease_speed}, {self.ease_influence});\n")
            elif elem["type"] == "pie":
                fracs = elem["fracs"]
                colors = elem["colors"]
                labels = elem.get("labels")
                donut = float(elem.get("donut", 0.0))
                base_r = elem.get("radius") or 0.40 * min(self.width, self.height)
                inner_r = donut * base_r
                start = math.radians(elem.get("start_angle", 0.0))
                total_anim = elem.get("animate") or 0.0
                alpha = float(elem.get("alpha", 1.0))
                n = len(fracs)
                label_color = elem.get("label_color")
                label_js = color_to_js(label_color) if label_color is not None else color_to_js(self.ui_color)
                gap_px = float(elem.get("gap", 0.0))
                roundness = float(elem.get("roundness", 0.0))
                leader_lines = bool(elem.get("leader_lines", False))
                # Overall easy-ease for the whole reveal: warp the keyframe
                # *times* with a smoothstep so the leading edge eases in and out
                # across the entire circle (slow -> fast -> slow), seamlessly
                # spanning all wedges. We sample the eased global progress
                # densely in time (gs[j] = fraction of the full pie revealed at
                # time tsamp[j]) and slice each wedge out of that shared
                # schedule, so wedge boundaries stay continuous and the arc
                # stays smooth. Interpolation between samples is linear; the
                # ease lives in the sample timing, not in per-keyframe beziers.
                if total_anim and total_anim > 0:
                    n_samp = 72
                    us = [j / n_samp for j in range(n_samp + 1)]
                    gs = [u * u * (3 - 2 * u) for u in us]      # smoothstep ease-in-out
                    tsamp = [delay + u * total_anim for u in us]
                # Labels/leaders all fade in over the SAME constant duration; the
                # pie only dictates *when* each one starts (its wedge's in-point),
                # not how long it takes, so small slices don't get a snappy label
                # while big slices get a slow one.
                label_fade = min(0.45, total_anim) if total_anim else 0.0
                # Gap layout (absolute pixels). Reserve a constant-width gap
                # between slices and split the *remaining* circle proportionally,
                # so turning on a gap shrinks every slice in proportion instead
                # of eating fixed chunks off the small ones. delta_o is the
                # half-gap angle at the rim (asin so the rim gap is exactly
                # gap_px wide); the edges are then drawn as parallel offset lines
                # so the gap keeps that width all the way in.
                half_g = gap_px / 2.0
                if half_g > 0 and base_r > 0:
                    delta_o = math.asin(min(half_g / base_r, 0.999))
                else:
                    delta_o = 0.0
                gap_ang = 2.0 * delta_o
                avail = 2.0 * math.pi - n * gap_ang
                if avail < 2.0 * math.pi * 0.25:        # absurdly large gaps
                    avail = 2.0 * math.pi * 0.25
                    gap_ang = (2.0 * math.pi - avail) / n
                    delta_o = gap_ang / 2.0
                half_g_eff = base_r * math.sin(delta_o)

                # --- Label de-clumping ------------------------------------
                # Each label normally sits on its wedge's angular bisector. When
                # several thin slices are adjacent, those bisectors crowd
                # together and the labels overlap. To avoid that, pre-compute
                # every wedge's bisector, split the labels onto the left/right
                # half of the circle, and push apart any that fall within a
                # minimum vertical spacing -- then recenter each side so the
                # cluster keeps its original mean position. Leader lines (when
                # enabled) bridge the gap from each rim back to its moved label.
                label_lr_factor = 1.16 if leader_lines else 1.18
                _mids = []
                _gc = start
                for _k in range(n):
                    _a0 = _gc + gap_ang / 2.0
                    _a1 = _a0 + fracs[_k] * avail
                    _gc = _a1 + gap_ang / 2.0
                    _mids.append((_a0 + _a1) / 2.0)
                _sides = [1.0 if math.sin(m) >= 0 else -1.0 for m in _mids]
                _base_ly = [-base_r * label_lr_factor * math.cos(m) for m in _mids]
                _adj_ly = list(_base_ly)
                _min_gap = 1.3 * 28 * self.font_scale
                for _sgn in (1.0, -1.0):
                    _idxs = [k for k in range(n) if _sides[k] == _sgn]
                    if len(_idxs) < 2:
                        continue
                    _idxs.sort(key=lambda k: _base_ly[k])
                    for _a in range(1, len(_idxs)):
                        _i, _j = _idxs[_a - 1], _idxs[_a]
                        if _adj_ly[_j] - _adj_ly[_i] < _min_gap:
                            _adj_ly[_j] = _adj_ly[_i] + _min_gap
                    _shift = (sum(_base_ly[k] for k in _idxs)
                              - sum(_adj_ly[k] for k in _idxs)) / len(_idxs)
                    for k in _idxs:
                        _adj_ly[k] += _shift

                cum = 0.0
                geom_cursor = start
                for k in range(n):
                    cum_before = cum
                    cum += fracs[k]
                    cum_after = cum
                    # Geometry angles carry the reserved gaps and keep slices
                    # proportional; the eased reveal still keys off the
                    # data-fraction cumulatives (cum_before / cum_after).
                    a0 = geom_cursor + gap_ang / 2.0
                    span = fracs[k] * avail
                    a1 = a0 + span
                    geom_cursor = a1 + gap_ang / 2.0
                    sweep = a1 - a0
                    o0, o1 = a0, a1
                    apex_pt = (0.0, 0.0)
                    if inner_r > 0:
                        # Donut: inset the inner arc more than the rim so the
                        # straight edge between them is the constant-gap offset
                        # line (gap stays gap_px wide at every radius).
                        delta_i = math.asin(min(half_g_eff / inner_r, 0.999)) if half_g_eff > 0 else 0.0
                        extra = min(max(0.0, delta_i - delta_o), 0.45 * sweep)
                        i0, i1 = a0 + extra, a1 - extra
                    else:
                        # Pie: the two offset edges meet at an apex on the
                        # bisector, so the gap stays a constant pixel width down
                        # to the point instead of pinching to zero at the center.
                        if half_g_eff > 0 and sweep > 1e-6:
                            r_apex = min(half_g_eff / math.sin(sweep / 2.0 + delta_o), base_r * 0.9)
                        else:
                            r_apex = 0.0
                        bis = (a0 + a1) / 2.0
                        apex_pt = (r_apex * math.sin(bis), -r_apex * math.cos(bis))
                    steps = max(2, int(math.ceil(abs(o1 - o0) / math.radians(3))))

                    # A wedge filled to fraction f in [0, 1]: the outer arc sweeps
                    # from o0 toward o1 and (for a donut) the inner arc mirrors it
                    # from i0 toward i1, with a FIXED vertex count so the path can
                    # be keyframed. Every outer point stays on the circle at
                    # radius base_r (inner at inner_r), so each keyframe is a real
                    # circular sector that fans open along the arc instead of
                    # straight-line morphing into a bulging blob.
                    def wedge_verts(f):
                        oe = o0 + (o1 - o0) * f
                        vs = []
                        for s in range(steps + 1):
                            t = o0 + (oe - o0) * (s / steps)
                            vs.append((base_r * math.sin(t), -base_r * math.cos(t)))
                        if inner_r > 0:
                            ie = i0 + (i1 - i0) * f
                            for s in range(steps + 1):
                                t = ie - (ie - i0) * (s / steps)
                                vs.append((inner_r * math.sin(t), -inner_r * math.cos(t)))
                        else:
                            vs.append(apex_pt)
                        return ",".join(f"[{vx},{vy}]" for vx, vy in vs)

                    full_js = wedge_verts(1.0)
                    col_js = color_to_js(colors[k])
                    pie_op = f"pieLayer{i}_{k}.property('Transform').property('Opacity')"
                    pie_opacity = int(alpha * 100)
                    script.append(f"var pieLayer{i}_{k} = comp.layers.addShape();\n")
                    script.append(f"pieLayer{i}_{k}.name = 'Pie_{i}_{k}';\n")
                    script.append(f"pieLayer{i}_{k}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                    script.append(f"pieLayer{i}_{k}.parent = PlotAnchor;\n")
                    script.append(f"var pieContents{i}_{k} = pieLayer{i}_{k}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var pieGrp{i}_{k} = pieContents{i}_{k}.addProperty('ADBE Vector Shape - Group');\n")
                    script.append(f"var piePath{i}_{k} = pieGrp{i}_{k}.property('ADBE Vector Shape');\n")
                    script.append(f"var pieShpFull{i}_{k} = new Shape();\n")
                    script.append(f"pieShpFull{i}_{k}.vertices = [{full_js}];\n")
                    script.append(f"pieShpFull{i}_{k}.closed = true;\n")
                    # Slice this wedge out of the shared eased schedule: it holds
                    # folded until the sweep reaches its start edge, fans open
                    # while the eased progress crosses [cum_before, cum_after],
                    # then holds full. Sharing the global sample times with its
                    # neighbors keeps the unfold continuous across wedges, and
                    # the smoothstep timing gives the whole pie one easy-ease.
                    lbl_t_enter = delay + cum_before * total_anim if total_anim else 0.0
                    lbl_t_exit = delay + cum_after * total_anim if total_anim else 0.0
                    if total_anim and total_anim > 0:
                        width = cum_after - cum_before
                        j_enter = 0
                        for j in range(n_samp + 1):
                            if gs[j] <= cum_before:
                                j_enter = j
                            else:
                                break
                        j_exit = n_samp
                        for j in range(n_samp + 1):
                            if gs[j] >= cum_after:
                                j_exit = j
                                break
                        for j in range(j_enter, j_exit + 1):
                            frac_in = (min(max(gs[j] - cum_before, 0.0), width) / width) if width > 0 else 1.0
                            verts_js = wedge_verts(frac_in)
                            script.append(f"var pieShp{i}_{k}_{j} = new Shape();\n")
                            script.append(f"pieShp{i}_{k}_{j}.vertices = [{verts_js}];\n")
                            script.append(f"pieShp{i}_{k}_{j}.closed = true;\n")
                            script.append(f"piePath{i}_{k}.setValueAtTime({tsamp[j]}, pieShp{i}_{k}_{j});\n")
                        # In-point comes from the pie (when this wedge starts
                        # unfolding); the fade length is the shared constant.
                        lbl_t_enter = tsamp[j_enter]
                        lbl_t_exit = lbl_t_enter + label_fade
                        # Collapsed path geometry can still anti-alias into visible
                        # stripes; keep the whole wedge layer invisible until its
                        # reveal in-point, then snap to full opacity (no fade).
                        script.append(f"{pie_op}.setValueAtTime(0, 0);\n")
                        script.append(f"{pie_op}.setValueAtTime({tsamp[j_enter]}, {pie_opacity});\n")
                        script.append(f"setKeyInterp({pie_op}, true);\n")
                    else:
                        script.append(f"piePath{i}_{k}.setValue(pieShpFull{i}_{k});\n")
                        script.append(f"{pie_op}.setValue({pie_opacity});\n")
                    # Round Corners modifies the path above it in the group, so it
                    # must sit between the path and the fill to soften the wedge's
                    # sharp corners.
                    if roundness > 0:
                        script.append(f"var pieRound{i}_{k} = pieContents{i}_{k}.addProperty('ADBE Vector Filter - RC');\n")
                        script.append(f"pieRound{i}_{k}.property('ADBE Vector RoundCorner Radius').setValue({roundness});\n")
                    script.append(f"var pieFill{i}_{k} = pieContents{i}_{k}.addProperty('ADBE Vector Graphic - Fill');\n")
                    script.append(f"pieFill{i}_{k}.property('ADBE Vector Fill Color').setValue({col_js});\n")
                    script.append(f"pieFill{i}_{k}.property('ADBE Vector Fill Opacity').setValue({int(alpha * 100)});\n")
                    if elem.get("stroke_color") is not None and elem.get("stroke_width", 0) > 0:
                        sc = color_to_js(elem["stroke_color"])
                        script.append(f"var pieStroke{i}_{k} = pieContents{i}_{k}.addProperty('ADBE Vector Graphic - Stroke');\n")
                        script.append(f"pieStroke{i}_{k}.property('ADBE Vector Stroke Color').setValue({sc});\n")
                        script.append(f"pieStroke{i}_{k}.property('ADBE Vector Stroke Width').setValue({elem['stroke_width']});\n")
                    if elem.get("show_labels") and labels is not None and k < len(labels):
                        mid = (a0 + a1) / 2
                        side = _sides[k]
                        ly = _adj_ly[k]   # de-clumped vertical position
                        if leader_lines:
                            # Stub from the rim to a vertical column at the
                            # de-clumped label height, then a short horizontal
                            # run into the label -- drawn in the slice's color.
                            # The elbow rides the moved label's y, so the line
                            # tracks the spread-out label instead of the bisector.
                            rimx, rimy = base_r * 1.02 * math.sin(mid), -base_r * 1.02 * math.cos(mid)
                            col_x = side * base_r * 1.14
                            endx = col_x + side * (base_r * 0.4)
                            lx = endx + side * (base_r * 0.02)
                            script.append(f"var pieLead{i}_{k} = comp.layers.addShape();\n")
                            script.append(f"pieLead{i}_{k}.name = 'PieLeader_{i}_{k}';\n")
                            script.append(f"pieLead{i}_{k}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                            script.append(f"pieLead{i}_{k}.parent = PlotAnchor;\n")
                            script.append(f"var pieLeadC{i}_{k} = pieLead{i}_{k}.property('ADBE Root Vectors Group');\n")
                            script.append(f"var pieLeadG{i}_{k} = pieLeadC{i}_{k}.addProperty('ADBE Vector Shape - Group');\n")
                            script.append(f"var pieLeadP{i}_{k} = pieLeadG{i}_{k}.property('ADBE Vector Shape');\n")
                            script.append(f"var pieLeadS{i}_{k} = new Shape();\n")
                            script.append(f"pieLeadS{i}_{k}.vertices = [[{rimx},{rimy}],[{col_x},{ly}],[{endx},{ly}]];\n")
                            script.append(f"pieLeadS{i}_{k}.closed = false;\n")
                            script.append(f"pieLeadP{i}_{k}.setValue(pieLeadS{i}_{k});\n")
                            script.append(f"var pieLeadStroke{i}_{k} = pieLeadC{i}_{k}.addProperty('ADBE Vector Graphic - Stroke');\n")
                            script.append(f"pieLeadStroke{i}_{k}.property('ADBE Vector Stroke Color').setValue({col_js});\n")
                            script.append(f"pieLeadStroke{i}_{k}.property('ADBE Vector Stroke Width').setValue(2.5);\n")
                            if total_anim and total_anim > 0:
                                ldop = f"pieLead{i}_{k}.property('Transform').property('Opacity')"
                                script.append(f"{ldop}.setValueAtTime({lbl_t_enter}, 0);\n")
                                script.append(f"{ldop}.setValueAtTime({lbl_t_exit}, 100);\n")
                            justify = "ParagraphJustification.LEFT_JUSTIFY" if side >= 0 else "ParagraphJustification.RIGHT_JUSTIFY"
                        else:
                            lr = base_r * 1.18
                            lx = lr * math.sin(mid)
                            justify = "ParagraphJustification.LEFT_JUSTIFY" if lx >= 0 else "ParagraphJustification.RIGHT_JUSTIFY"
                        txt = str(labels[k])
                        if elem.get("show_percent"):
                            txt = f"{txt} {fracs[k] * 100:.0f}%"
                        txt = txt.replace("\\", "\\\\").replace('"', '\\"')
                        script.append(f"var pieLbl{i}_{k} = comp.layers.addText(\"{txt}\");\n")
                        script.append(f"pieLbl{i}_{k}.property('Transform').property('Position').setValue([{center_x + lx}, {center_y + ly}]);\n")
                        script.append(f"pieLbl{i}_{k}.parent = PlotAnchor;\n")
                        script.append(f"var pieLblProp{i}_{k} = pieLbl{i}_{k}.property('Source Text');\n")
                        script.append(f"var pieLblDoc{i}_{k} = pieLblProp{i}_{k}.value;\n")
                        script.append(f"pieLblDoc{i}_{k}.fontSize = {int(28 * self.font_scale)};\n")
                        script.append(f"pieLblDoc{i}_{k}.font = \"{self.font_label}\";\n")
                        script.append(f"pieLblDoc{i}_{k}.fillColor = {label_js};\n")
                        script.append(f"pieLblDoc{i}_{k}.justification = {justify};\n")
                        script.append(f"pieLblProp{i}_{k}.setValue(pieLblDoc{i}_{k});\n")
                        # Anchor the text at the vertical center of its near edge so
                        # the leader line points to the middle of the label instead
                        # of a corner.
                        script.append(f"var pieLblSR{i}_{k} = pieLbl{i}_{k}.sourceRectAtTime(0, false);\n")
                        script.append(f"var pieLblAP{i}_{k} = pieLbl{i}_{k}.property('Transform').property('Anchor Point');\n")
                        _sr = f"pieLblSR{i}_{k}"
                        if justify == "ParagraphJustification.LEFT_JUSTIFY":
                            _anchor_x = f"{_sr}.left"
                        else:  # RIGHT_JUSTIFY
                            _anchor_x = f"{_sr}.left + {_sr}.width"
                        script.append(f"pieLblAP{i}_{k}.setValue([{_anchor_x}, {_sr}.top + {_sr}.height/2]);\n")
                        # Fade the label in as its wedge finishes unfolding so the
                        # text rides along with the sweep instead of appearing up
                        # front.
                        if total_anim and total_anim > 0:
                            lop = f"pieLbl{i}_{k}.property('Transform').property('Opacity')"
                            script.append(f"{lop}.setValueAtTime({lbl_t_enter}, 0);\n")
                            script.append(f"{lop}.setValueAtTime({lbl_t_exit}, 100);\n")
            elif elem["type"] == "scatter":
                px, py = elem["x"], elem["y"]
                n_points = len(px)
                point_colors = _prepare_scatter_colors(elem["color"], n_points)
                point_radii = _prepare_scatter_radii(elem["radius"], n_points)
                outline = bool(elem.get("outline", getattr(config, "scatter_outline", True)))
                outline_width = float(elem.get("outline_width", getattr(config, "scatter_outline_width", 1.0)))
                marker = elem.get("marker", "circle")
                scatter_alpha = float(elem.get("alpha", 1.0))
                scatter_opacity = max(0, min(100, scatter_alpha * 100))
                outline_opacity = max(0, min(100, float(elem.get("outline_alpha", 1.0)) * 100))
                # per-point animation durations: prefer `point_anim_times`,
                # fallback to legacy `bar_anim_times` or single `point_duration`/`point_anim_times`
                point_anim_times = elem.get("point_anim_times")
                if point_anim_times is None:
                    point_anim_times = elem.get("bar_anim_times")
                # support single-value alias `point_duration`
                if point_anim_times is None and elem.get("point_duration") is not None:
                    point_anim_times = elem.get("point_duration")

                total_anim = elem["animate"] if elem["animate"] else 1.0
                elem_ease_speed = self.ease_speed if elem.get("ease_speed") is None else elem.get("ease_speed")
                elem_ease_influence = self.ease_influence if elem.get("ease_influence") is None else elem.get("ease_influence")
                # Accept a single float for per-point durations and apply to all points
                if point_anim_times is None:
                    point_anim_times = [total_anim / n_points] * n_points
                elif isinstance(point_anim_times, (int, float)):
                    point_anim_times = [float(point_anim_times)] * n_points
                # Defensive: if point_anim_times is not a list of correct length, fallback to default
                if not hasattr(point_anim_times, '__iter__') or len(point_anim_times) != n_points:
                    point_anim_times = [total_anim / n_points] * n_points

                # Overlapping animation: distribute start times evenly within total_anim
                # Start times span 0..total_anim so entrances are spread across the animate period,
                # unless the caller supplied explicit per-point start times (e.g. to line up
                # exactly with a paired bar/stem animation).
                explicit_start_times = elem.get("point_start_times")
                if explicit_start_times is not None and len(explicit_start_times) == n_points:
                    start_times = list(explicit_start_times)
                else:
                    start_times = np.linspace(0, total_anim, n_points)
                    # Ease the sequence of point entrances so the sweep itself eases in/out.
                    start_times = self._apply_meta_ease(start_times, elem)

                scatter_px, scatter_py = zip(*[self._data_to_shape(x, y, xmin_pad, xmax_pad, ymin_pad, ymax_pad) for x, y in zip(px, py)])
                point_images = elem.get("images")
                move_from_x_list = elem.get("move_from_x")
                move_from_y_list = elem.get("move_from_y")
                delay = float(elem.get("delay", 0.0) or 0.0)
                reveal_duration = float(elem.get("point_reveal_duration", 0.3))
                clip_to_view = bool(elem.get("clip_to_view", getattr(config, "scatter_clip_to_view", True)))
                for j, (sx, sy) in enumerate(zip(scatter_px, scatter_py)):
                    anim_time = point_anim_times[j] if elem["animate"] and elem["animate"] > 0 else 0.0
                    start_time = start_times[j]
                    move_start = None
                    if move_from_x_list is not None or move_from_y_list is not None:
                        fx = move_from_x_list[j] if move_from_x_list is not None else px[j]
                        fy = move_from_y_list[j] if move_from_y_list is not None else py[j]
                        move_start = self._data_to_shape(fx, fy, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    img_path = point_images[j] if point_images else None
                    if img_path:
                        self._emit_scatter_image_point_jsx(
                            script, i, j, img_path, center_x, center_y, sx, sy,
                            px[j], py[j], point_radii[j], marker, scatter_opacity,
                            outline, outline_width, elem.get("outline_color"),
                            outline_opacity, point_colors[j], delay, start_time,
                            anim_time, elem_ease_speed, elem_ease_influence,
                            elem.get("drop_shadow", False), clip_to_view,
                            move_start=move_start,
                            reveal_duration=reveal_duration,
                        )
                        continue
                    init_sx, init_sy = move_start if move_start is not None else (sx, sy)
                    script.append(f"var scatterLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"scatterLayer{i}_{j}.name = \"Scatter_{i}_{j}\";\n")
                    script.append(f"scatterLayer{i}_{j}.property('Transform').property('Position').setValue([{center_x + init_sx}, {center_y + init_sy}]);\n")
                    script.append(f"scatterLayer{i}_{j}.parent = PlotAnchor;\n")
                    self._emit_screen_space_mask_jsx(script, f"scatterLayer{i}_{j}")
                    if self._view_animated:
                        # After parenting, Position is parent-relative shape coords.
                        script.append(f"var scatterPos{i}_{j} = scatterLayer{i}_{j}.property('Transform').property('Position');\n")
                        self._emit_view_pos_kf(
                            script, f"scatterPos{i}_{j}",
                            lambda xmn, xmx, ymn, ymx, _x=px[j], _y=py[j]: self._data_to_shape(_x, _y, xmn, xmx, ymn, ymx),
                        )
                    elif move_start is not None:
                        script.append(f"var scatterPos{i}_{j} = scatterLayer{i}_{j}.property('Transform').property('Position');\n")
                        # Post-parent keyframes must use raw shape coords (sx, sy),
                        # not `center_x + sx` -- see comment in
                        # `_emit_scatter_image_point_jsx` for why.
                        script.extend(self._emit_point_move_kf(
                            f"scatterPos{i}_{j}",
                            (init_sx, init_sy),
                            (sx, sy),
                            entrance_start=delay + start_time, entrance_dur=anim_time,
                            ease_speed=elem_ease_speed, ease_influence=elem_ease_influence,
                        ))
                        script.append(
                            f"var scatterScale{i}_{j} = scatterLayer{i}_{j}.property('Transform')"
                            f".property('Scale');\n"
                        )
                        script.extend(self._emit_point_hide_until_kf(
                            f"scatterScale{i}_{j}", delay + start_time,
                            reveal_duration=reveal_duration,
                            ease_speed=elem_ease_speed, ease_influence=elem_ease_influence,
                        ))
                    script.append(f"var contents{i}_{j} = scatterLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(_marker_static_jsx(f"contents{i}_{j}", f"markerGroup{i}_{j}", marker, point_radii[j]))
                    script.append(f"var fill{i}_{j} = contents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    script.append(f"fill{i}_{j}.property('ADBE Vector Fill Color').setValue({color_to_js(point_colors[j])});\n")
                    if scatter_opacity < 100:
                        script.append(f"fill{i}_{j}.property('ADBE Vector Fill Opacity').setValue({scatter_opacity});\n")
                    if outline and outline_width > 0:
                        outline_color = elem.get("outline_color") or self._darken_color(point_colors[j])
                        script.append(f"var stroke{i}_{j} = contents{i}_{j}.addProperty('ADBE Vector Graphic - Stroke');\n")
                        script.append(f"stroke{i}_{j}.property('ADBE Vector Stroke Color').setValue({color_to_js(outline_color)});\n")
                        script.append(f"stroke{i}_{j}.property('ADBE Vector Stroke Width').setValue({outline_width});\n")
                        script.append(f"stroke{i}_{j}.property('ADBE Vector Stroke Opacity').setValue({outline_opacity});\n")

                    # Sequential animation for scatter points, plus (optionally)
                    # hiding the marker whenever the animated view window no
                    # longer contains its data point.
                    anim_time = point_anim_times[j] if elem["animate"] and elem["animate"] > 0 else 0.0
                    start_time = start_times[j]
                    if move_start is None and (
                        anim_time > 0 or (clip_to_view and self._view_animated)
                    ):
                        scale_lines = self._emit_point_scale_kf(
                            f"scale{i}_{j}", px[j], py[j],
                            entrance_start=delay + start_time, entrance_dur=anim_time,
                            ease_speed=elem_ease_speed, ease_influence=elem_ease_influence,
                            clip_to_view=clip_to_view,
                        )
                        if scale_lines:
                            script.append(f"var scale{i}_{j} = scatterLayer{i}_{j}.property('Transform').property('Scale');\n")
                            script.extend(scale_lines)

                    # Add drop shadow if specified
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"scatterLayer{i}_{j}", f"{i}_{j}"))
            elif elem["type"] == "quiver":
                bx = elem["qx"]; by = elem["qy"]
                vx = elem["qvx"]; vy = elem["qvy"]
                arrow_colors = elem.get("arrow_colors")
                base_color_js = color_to_js(elem["color"])
                width = elem.get("width", 3.0)
                hw_mult = elem.get("headwidth", 3.0)
                hl_px = elem.get("headlength", 11.0)
                alpha = elem.get("alpha", 1.0)
                pivot = elem.get("pivot", "tail")
                scale_mode = elem.get("scale_mode", "data")
                qscale = elem.get("qscale", 1.0)
                qrel = elem.get("qrel")
                plot_w = float(self.width)
                total_anim = elem.get("animate") or 0.0
                n_arrows = len(bx)
                for j in range(n_arrows):
                    if scale_mode == "comp":
                        # Length is a fraction of the plot width (sized only
                        # relative to other arrows); direction comes from the
                        # on-screen mapping of the (u, v) step. Pivot is applied
                        # in pixel space so arrows stay a fixed length.
                        bsx, bsy = self._data_to_shape(bx[j], by[j], xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        tsx, tsy = self._data_to_shape(bx[j] + vx[j], by[j] + vy[j], xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        ddx, ddy = tsx - bsx, tsy - bsy
                        Ld = float(np.hypot(ddx, ddy))
                        if Ld < 1e-9:
                            continue
                        ux, uy = ddx / Ld, ddy / Ld
                        Lp = qscale * plot_w * (qrel[j] if qrel is not None else 1.0)
                        if Lp < 1e-6:
                            continue
                        if pivot == "mid":
                            sx0, sy0 = bsx - ux * Lp / 2.0, bsy - uy * Lp / 2.0
                            sx1, sy1 = bsx + ux * Lp / 2.0, bsy + uy * Lp / 2.0
                        elif pivot == "tip":
                            sx0, sy0 = bsx - ux * Lp, bsy - uy * Lp
                            sx1, sy1 = bsx, bsy
                        else:  # tail
                            sx0, sy0 = bsx, bsy
                            sx1, sy1 = bsx + ux * Lp, bsy + uy * Lp
                    else:
                        if pivot == "mid":
                            x0, y0 = bx[j] - vx[j] / 2.0, by[j] - vy[j] / 2.0
                            x1, y1 = bx[j] + vx[j] / 2.0, by[j] + vy[j] / 2.0
                        elif pivot == "tip":
                            x0, y0 = bx[j] - vx[j], by[j] - vy[j]
                            x1, y1 = bx[j], by[j]
                        else:  # tail
                            x0, y0 = bx[j], by[j]
                            x1, y1 = bx[j] + vx[j], by[j] + vy[j]
                        sx0, sy0 = self._data_to_shape(x0, y0, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        sx1, sy1 = self._data_to_shape(x1, y1, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        dxp, dyp = sx1 - sx0, sy1 - sy0
                        Lp = float(np.hypot(dxp, dyp))
                        if Lp < 1e-6:
                            continue
                        ux, uy = dxp / Lp, dyp / Lp        # unit direction (shape px)
                    ppx, ppy = -uy, ux                  # unit perpendicular
                    sw = width / 2.0                    # shaft half-width
                    hw = (width * hw_mult) / 2.0        # head half-width
                    hl = min(hl_px, Lp * 0.85)          # head length (clamped)
                    hbx, hby = sx1 - ux * hl, sy1 - uy * hl
                    verts = [
                        (sx0 + ppx * sw, sy0 + ppy * sw),
                        (hbx + ppx * sw, hby + ppy * sw),
                        (hbx + ppx * hw, hby + ppy * hw),
                        (sx1, sy1),
                        (hbx - ppx * hw, hby - ppy * hw),
                        (hbx - ppx * sw, hby - ppy * sw),
                        (sx0 - ppx * sw, sy0 - ppy * sw),
                    ]
                    pts_js = ",".join(f"[{vxp},{vyp}]" for vxp, vyp in verts)
                    col_js = (f"[{arrow_colors[j][0]}, {arrow_colors[j][1]}, {arrow_colors[j][2]}]"
                              if arrow_colors else base_color_js)
                    script.append(f"var quiverLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"quiverLayer{i}_{j}.name = 'Quiver_{i}_{j}';\n")
                    script.append(f"quiverLayer{i}_{j}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                    script.append(f"quiverLayer{i}_{j}.parent = PlotAnchor;\n")
                    self._emit_screen_space_mask_jsx(script, f"quiverLayer{i}_{j}")
                    script.append(f"var qContents{i}_{j} = quiverLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var qGroup{i}_{j} = qContents{i}_{j}.addProperty('ADBE Vector Shape - Group');\n")
                    script.append(f"var qPath{i}_{j} = qGroup{i}_{j}.property('ADBE Vector Shape');\n")
                    script.append(f"var qShp{i}_{j} = new Shape();\n")
                    script.append(f"qShp{i}_{j}.vertices = [{pts_js}];\n")
                    script.append(f"qShp{i}_{j}.closed = true;\n")
                    script.append(f"qPath{i}_{j}.setValue(qShp{i}_{j});\n")
                    script.append(f"var qFill{i}_{j} = qContents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    script.append(f"qFill{i}_{j}.property('ADBE Vector Fill Color').setValue({col_js});\n")
                    script.append(f"qFill{i}_{j}.property('ADBE Vector Fill Opacity').setValue({int(alpha * 100)});\n")
                    if total_anim and total_anim > 0:
                        frac = j / max(n_arrows - 1, 1)
                        q_meta = elem.get("meta_easy_ease")
                        if q_meta is None:
                            q_meta = self.meta_easy_ease
                        if q_meta:
                            q_inf = elem.get("meta_ease_influence")
                            q_inf = self.meta_ease_influence if q_inf is None else q_inf
                            q_spd = elem.get("meta_ease_speed")
                            q_spd = self.meta_ease_speed if q_spd is None else q_spd
                            frac = self._cubic_bezier_ease(frac, q_inf, q_spd, invert=True)
                        t_start = delay + frac * total_anim * 0.6
                        op = f"quiverLayer{i}_{j}.property('Transform').property('Opacity')"
                        script.append(f"{op}.setValueAtTime({t_start}, 0);\n")
                        script.append(f"{op}.setValueAtTime({t_start + 0.4}, 100);\n")
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"quiverLayer{i}_{j}", f"q{i}_{j}"))
            elif elem["type"] == "heatmap":
                xe = elem["hx_edges"]; ye = elem["hy_edges"]
                cell_colors = elem["cell_colors"]
                nx = elem["nx"]; ny = elem["ny"]
                alpha = elem.get("alpha", 1.0)
                gap = elem.get("gap", 0.0)
                edge_color = elem.get("edge_color")
                edge_width = elem.get("edge_width", 0.0)
                total_anim = elem.get("animate") or 0.0
                reveal = elem.get("reveal", "fade")
                edge_js = color_to_js(edge_color) if edge_color is not None else None
                n_cells = nx * ny
                # Diagonal reveal: each cell fades in on its own, staggered from
                # the bottom-left corner (col 0, row 0) to the top-right corner.
                diag_reveal = reveal == "diagonal" and total_anim > 0
                denom_x = (nx - 1) or 1
                denom_y = (ny - 1) or 1
                cell_fade = max(0.12, total_anim * 0.18)
                stagger_window = max(total_anim - cell_fade, 0.0)
                # All cells live on one shape layer (fast + a single fade-in).
                script.append(f"var heatLayer{i} = comp.layers.addShape();\n")
                script.append(f"heatLayer{i}.name = 'Heatmap_{i}';\n")
                script.append(f"heatLayer{i}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"heatLayer{i}.parent = PlotAnchor;\n")
                self._emit_screen_space_mask_jsx(script, f"heatLayer{i}")
                script.append(f"var heatContents{i} = heatLayer{i}.property('ADBE Root Vectors Group');\n")
                cell_idx = 0
                for r in range(ny):
                    for col in range(nx):
                        color = cell_colors[r][col]
                        if color is None:
                            cell_idx += 1
                            continue
                        # Cell corners in data space, optionally inset by `gap`.
                        x0d, x1d = xe[col], xe[col + 1]
                        y0d, y1d = ye[r], ye[r + 1]
                        if gap > 0:
                            ix = (x1d - x0d) * gap / 2.0
                            iy = (y1d - y0d) * gap / 2.0
                            x0d, x1d = x0d + ix, x1d - ix
                            y0d, y1d = y0d + iy, y1d - iy
                        ax, ay = self._data_to_shape(x0d, y0d, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        bx, by = self._data_to_shape(x1d, y1d, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        verts = [(ax, ay), (bx, ay), (bx, by), (ax, by)]
                        pts_js = ",".join(f"[{vxp},{vyp}]" for vxp, vyp in verts)
                        col_js = f"[{color[0]}, {color[1]}, {color[2]}]"
                        gi = f"{i}_{cell_idx}"
                        script.append(f"var hGroup{gi} = heatContents{i}.addProperty('ADBE Vector Group');\n")
                        script.append(f"var hGroupC{gi} = hGroup{gi}.property('ADBE Vectors Group');\n")
                        script.append(f"var hGrp{gi} = hGroupC{gi}.addProperty('ADBE Vector Shape - Group');\n")
                        script.append(f"var hPath{gi} = hGrp{gi}.property('ADBE Vector Shape');\n")
                        script.append(f"var hShp{gi} = new Shape();\n")
                        script.append(f"hShp{gi}.vertices = [{pts_js}];\n")
                        script.append(f"hShp{gi}.closed = true;\n")
                        script.append(f"hPath{gi}.setValue(hShp{gi});\n")
                        script.append(f"var hFill{gi} = hGroupC{gi}.addProperty('ADBE Vector Graphic - Fill');\n")
                        script.append(f"hFill{gi}.property('ADBE Vector Fill Color').setValue({col_js});\n")
                        script.append(f"hFill{gi}.property('ADBE Vector Fill Opacity').setValue({int(alpha * 100)});\n")
                        if edge_js is not None and edge_width > 0:
                            script.append(f"var hStroke{gi} = hGroupC{gi}.addProperty('ADBE Vector Graphic - Stroke');\n")
                            script.append(f"hStroke{gi}.property('ADBE Vector Stroke Color').setValue({edge_js});\n")
                            script.append(f"hStroke{gi}.property('ADBE Vector Stroke Width').setValue({edge_width});\n")
                        if diag_reveal:
                            # Progress 0 at bottom-left (col 0, row 0) -> 1 at
                            # top-right. Each cell fades in over `cell_fade`.
                            d = (col / denom_x + r / denom_y) / 2.0
                            t0 = delay + d * stagger_window
                            gop = (f"hGroup{gi}.property('ADBE Vector Transform Group')"
                                   f".property('ADBE Vector Group Opacity')")
                            # Group opacity multiplies the fill's own alpha, so
                            # reveal to full (100) and let Fill Opacity carry alpha.
                            script.append(f"{gop}.setValueAtTime({t0}, 0);\n")
                            script.append(f"{gop}.setValueAtTime({t0 + cell_fade}, 100);\n")
                        cell_idx += 1
                if total_anim and total_anim > 0 and not diag_reveal:
                    # "fade" reveal: fade the whole grid in together via a single
                    # layer-opacity keyframe pair.
                    op = f"heatLayer{i}.property('Transform').property('Opacity')"
                    script.append(f"{op}.setValueAtTime({delay}, 0);\n")
                    script.append(f"{op}.setValueAtTime({delay + total_anim}, 100);\n")
                if elem.get("drop_shadow", False):
                    script.append(self._generate_drop_shadow_jsx(f"heatLayer{i}", f"h{i}"))
            elif elem["type"] == "heatmap_evolving":
                xe = elem["hx_edges"]; ye = elem["hy_edges"]
                color_frames = elem["color_frames"]
                cell_mask = elem["cell_mask"]
                times = elem["frame_times"]
                hold_js = "true" if elem["hold"] else "false"
                nx = elem["nx"]; ny = elem["ny"]
                alpha = elem.get("alpha", 1.0)
                gap = elem.get("gap", 0.0)
                fade = elem.get("fade_in", 0.0)
                script.append(f"var heatLayer{i} = comp.layers.addShape();\n")
                script.append(f"heatLayer{i}.name = 'HeatmapEvolving_{i}';\n")
                script.append(f"heatLayer{i}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"heatLayer{i}.parent = PlotAnchor;\n")
                self._emit_screen_space_mask_jsx(script, f"heatLayer{i}")
                script.append(f"var heatContents{i} = heatLayer{i}.property('ADBE Root Vectors Group');\n")
                cell_idx = 0
                for r in range(ny):
                    for col in range(nx):
                        if not cell_mask[r][col]:
                            cell_idx += 1
                            continue
                        x0d, x1d = xe[col], xe[col + 1]
                        y0d, y1d = ye[r], ye[r + 1]
                        if gap > 0:
                            ix = (x1d - x0d) * gap / 2.0
                            iy = (y1d - y0d) * gap / 2.0
                            x0d, x1d = x0d + ix, x1d - ix
                            y0d, y1d = y0d + iy, y1d - iy
                        ax, ay = self._data_to_shape(x0d, y0d, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        bx, by = self._data_to_shape(x1d, y1d, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        verts = [(ax, ay), (bx, ay), (bx, by), (ax, by)]
                        pts_js = ",".join(f"[{vxp},{vyp}]" for vxp, vyp in verts)
                        gi = f"{i}_{cell_idx}"
                        script.append(f"var hGroup{gi} = heatContents{i}.addProperty('ADBE Vector Group');\n")
                        script.append(f"var hGroupC{gi} = hGroup{gi}.property('ADBE Vectors Group');\n")
                        script.append(f"hGroupC{gi}.addProperty('ADBE Vector Shape - Group');\n")
                        script.append(f"hGroupC{gi}.addProperty('ADBE Vector Graphic - Fill');\n")
                        script.append(f"var hGrp{gi} = hGroupC{gi}.property('ADBE Vector Shape - Group');\n")
                        script.append(f"var hPath{gi} = hGrp{gi}.property('ADBE Vector Shape');\n")
                        script.append(f"var hFill{gi} = hGroupC{gi}.property('ADBE Vector Graphic - Fill');\n")
                        script.append(f"hFill{gi}.property('ADBE Vector Fill Opacity').setValue({int(alpha * 100)});\n")
                        script.append(f"var hShp{gi} = new Shape();\n")
                        script.append(f"hShp{gi}.vertices = [{pts_js}];\n")
                        script.append(f"hShp{gi}.closed = true;\n")
                        script.append(f"hPath{gi}.setValue(hShp{gi});\n")
                        script.append(f"var hFillColor{gi} = hFill{gi}.property('ADBE Vector Fill Color');\n")
                        for k, tk in enumerate(times):
                            c = color_frames[k][r][col]
                            script.append(f"hFillColor{gi}.setValueAtTime({tk}, [{c[0]}, {c[1]}, {c[2]}]);\n")
                        script.append(f"setKeyInterp(hFillColor{gi}, {hold_js});\n")
                        cell_idx += 1
                if fade and fade > 0:
                    # Fade in with everything else (un-delayed); only the data
                    # animation keyframes carry the ``delay`` offset.
                    t0 = times[0] - elem.get("delay", 0.0)
                    op = f"heatLayer{i}.property('Transform').property('Opacity')"
                    script.append(f"{op}.setValueAtTime({t0}, 0);\n")
                    script.append(f"{op}.setValueAtTime({t0 + fade}, 100);\n")
                if elem.get("drop_shadow", False):
                    script.append(self._generate_drop_shadow_jsx(f"heatLayer{i}", f"h{i}"))
            elif elem["type"] == "line_evolving":
                x_frames = elem["x_frames"]
                y_frames = elem["y_frames"]
                times = elem["frame_times"]
                hold_js = "true" if elem["hold"] else "false"
                color_js = color_to_js(elem["color"])
                fade = elem.get("fade_in", 0.0)
                script.append(f"var lineLayer{i} = comp.layers.addShape();\n")
                script.append(f"lineLayer{i}.name = \"LineEvolving_{i}\";\n")
                script.append(f"lineLayer{i}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"lineLayer{i}.parent = PlotAnchor;\n")
                self._emit_screen_space_mask_jsx(script, f"lineLayer{i}")
                script.append(f"var contents{i} = lineLayer{i}.property('ADBE Root Vectors Group');\n")
                script.append(f"var pathGroup{i} = contents{i}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var path{i} = pathGroup{i}.property('ADBE Vector Shape');\n")
                if self._view_animated:
                    # Compose the data morph with the animated view window: sample
                    # densely, interpolating both data and limits at each sample.
                    samp_times = self._view_composite_times(times)
                    for k, tk in enumerate(samp_times):
                        xr = self._interp_frame_rows(x_frames, times, tk, elem["hold"])
                        yr = self._interp_frame_rows(y_frames, times, tk, elem["hold"])
                        vxmn, vxmx, vymn, vymx = self._view_limits_at(tk)
                        shape_pts = [self._data_to_shape(xx, yy, vxmn, vxmx, vymn, vymx)
                                     for xx, yy in zip(xr, yr)]
                        pts_js = ",".join(f"[{sx},{sy}]" for sx, sy in shape_pts)
                        script.append(f"var shp{i}_{k} = new Shape();\n")
                        script.append(f"shp{i}_{k}.vertices = [{pts_js}];\n")
                        script.append(f"shp{i}_{k}.closed = false;\n")
                        script.append(f"path{i}.setValueAtTime({tk}, shp{i}_{k});\n")
                    # Densely sampled -> always LINEAR between samples.
                    script.append(f"setKeyInterp(path{i}, false);\n")
                else:
                    for k, (xr, yr, tk) in enumerate(zip(x_frames, y_frames, times)):
                        shape_pts = [self._data_to_shape(xx, yy, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                                     for xx, yy in zip(xr, yr)]
                        pts_js = ",".join(f"[{sx},{sy}]" for sx, sy in shape_pts)
                        script.append(f"var shp{i}_{k} = new Shape();\n")
                        script.append(f"shp{i}_{k}.vertices = [{pts_js}];\n")
                        script.append(f"shp{i}_{k}.closed = false;\n")
                        script.append(f"path{i}.setValueAtTime({tk}, shp{i}_{k});\n")
                    script.append(f"setKeyInterp(path{i}, {hold_js});\n")
                if elem.get("loop"):
                    script.append(f"loopOutCycle(path{i});\n")
                script.append(f"var stroke{i} = contents{i}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"stroke{i}.property('ADBE Vector Stroke Color').setValue({color_js});\n")
                script.append(f"stroke{i}.property('ADBE Vector Stroke Width').setValue({elem['linewidth']});\n")
                script.append(f"stroke{i}.property('ADBE Vector Stroke Line Cap').setValue(2);\n")
                script.append(f"stroke{i}.property('ADBE Vector Stroke Line Join').setValue(2);\n")
                linestyle = elem.get("linestyle", "solid")
                dash_size = elem.get("dash_size", 1.0)
                if linestyle in ["dashed", "--", "dotted", ":"]:
                    dash_val, gap_val = self._get_dash_values(linestyle, dash_size)
                    script.append(f"stroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({dash_val});\n")
                    script.append(f"stroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({gap_val});\n")
                if fade and fade > 0:
                    # Fade in with everything else (un-delayed); only the morph
                    # keyframes carry the ``delay`` offset.
                    t0 = times[0] - elem.get("delay", 0.0)
                    script.append(f"var lineOp{i} = lineLayer{i}.property('Transform').property('Opacity');\n")
                    script.append(f"lineOp{i}.setValueAtTime({t0}, 0);\n")
                    script.append(f"lineOp{i}.setValueAtTime({t0 + fade}, 100);\n")
                if elem.get("drop_shadow", False):
                    script.append(self._generate_drop_shadow_jsx(f"lineLayer{i}", f"{i}"))
            elif elem["type"] == "scatter_evolving":
                x_frames = elem["x_frames"]
                y_frames = elem["y_frames"]
                radius_frames = elem["radius_frames"]
                color_frames = elem.get("color_frames")
                times = elem["frame_times"]
                hold_js = "true" if elem["hold"] else "false"
                n_pts = elem["n_points"]
                alpha = elem.get("alpha", 1.0)
                radius_varies = elem.get("radius_varies", False)
                base_color_js = color_to_js(elem["color"])
                fade = elem.get("fade_in", 0.0)
                outline = bool(elem.get("outline", getattr(config, "scatter_outline", True)))
                outline_width = float(elem.get("outline_width", getattr(config, "scatter_outline_width", 1.0)))
                draw_outline = outline and outline_width > 0
                marker = elem.get("marker", "circle")
                marker_spec = _MARKER_SHAPES[marker]
                is_circle = marker_spec is None
                shape_match = "ADBE Vector Shape - Ellipse" if is_circle else "ADBE Vector Shape - Star"
                inner_ratio = None if is_circle else marker_spec["inner_ratio"]
                for j in range(n_pts):
                    script.append(f"var scatterLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"scatterLayer{i}_{j}.name = \"ScatterEvolving_{i}_{j}\";\n")
                    script.append(f"scatterLayer{i}_{j}.parent = PlotAnchor;\n")
                    self._emit_screen_space_mask_jsx(script, f"scatterLayer{i}_{j}")
                    script.append(f"var contents{i}_{j} = scatterLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    # Add every child property FIRST, then re-fetch references.
                    # Calling addProperty on a group invalidates ALL previously
                    # captured references to that group's children (the group
                    # handles too), so anything used later must be looked up after
                    # the group structure is fully built.
                    script.append(f"contents{i}_{j}.addProperty('{shape_match}');\n")
                    script.append(f"contents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    if draw_outline:
                        script.append(f"contents{i}_{j}.addProperty('ADBE Vector Graphic - Stroke');\n")
                    script.append(f"var markerGroup{i}_{j} = contents{i}_{j}.property('{shape_match}');\n")
                    script.append(f"var fill{i}_{j} = contents{i}_{j}.property('ADBE Vector Graphic - Fill');\n")
                    script.append(f"fill{i}_{j}.property('ADBE Vector Fill Opacity').setValue({int(alpha*100)});\n")
                    if is_circle:
                        script.append(f"var markerSize{i}_{j} = markerGroup{i}_{j}.property('ADBE Vector Ellipse Size');\n")
                    else:
                        # Static polystar params (sharp polygon/star pointing up).
                        script.append(f"markerGroup{i}_{j}.property('ADBE Vector Star Type').setValue({marker_spec['star_type']});\n")
                        script.append(f"markerGroup{i}_{j}.property('ADBE Vector Star Points').setValue({marker_spec['points']});\n")
                        script.append(f"markerGroup{i}_{j}.property('ADBE Vector Star Rotation').setValue({marker_spec['rotation']});\n")
                        script.append(f"var markerSize{i}_{j} = markerGroup{i}_{j}.property('ADBE Vector Star Outer Radius');\n")
                        if inner_ratio is not None:
                            script.append(f"var markerInner{i}_{j} = markerGroup{i}_{j}.property('ADBE Vector Star Inner Radius');\n")
                    script.append(f"var fillColor{i}_{j} = fill{i}_{j}.property('ADBE Vector Fill Color');\n")
                    if draw_outline:
                        script.append(f"var stroke{i}_{j} = contents{i}_{j}.property('ADBE Vector Graphic - Stroke');\n")
                        script.append(f"stroke{i}_{j}.property('ADBE Vector Stroke Opacity').setValue({int(alpha*100)});\n")
                        script.append(f"stroke{i}_{j}.property('ADBE Vector Stroke Width').setValue({outline_width});\n")
                        script.append(f"var strokeColor{i}_{j} = stroke{i}_{j}.property('ADBE Vector Stroke Color');\n")
                    script.append(f"var pos{i}_{j} = scatterLayer{i}_{j}.property('Transform').property('Position');\n")
                    # Position keyframes (always — this is the motion). The layer
                    # is parented to PlotAnchor (which sits at the plot origin), so
                    # keyframe values must be parent-relative shape coords [sx, sy]
                    # — NOT world coords. Using world coords here would double-offset
                    # every point by the PlotAnchor position.
                    if self._view_animated:
                        samp_times = self._view_composite_times(times)
                        xrow_j = [x_frames[k][j] for k in range(len(times))]
                        yrow_j = [y_frames[k][j] for k in range(len(times))]
                        for tk in samp_times:
                            dx = self._interp_frame_rows([[v] for v in xrow_j], times, tk, elem["hold"])[0]
                            dy = self._interp_frame_rows([[v] for v in yrow_j], times, tk, elem["hold"])[0]
                            vxmn, vxmx, vymn, vymx = self._view_limits_at(tk)
                            sx, sy = self._data_to_shape(dx, dy, vxmn, vxmx, vymn, vymx)
                            script.append(f"pos{i}_{j}.setValueAtTime({tk}, [{sx}, {sy}]);\n")
                        script.append(f"setKeyInterp(pos{i}_{j}, false);\n")
                    else:
                        for k, tk in enumerate(times):
                            sx, sy = self._data_to_shape(x_frames[k][j], y_frames[k][j],
                                                         xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                            script.append(f"pos{i}_{j}.setValueAtTime({tk}, [{sx}, {sy}]);\n")
                        script.append(f"setKeyInterp(pos{i}_{j}, {hold_js});\n")
                    if elem.get("loop"):
                        script.append(f"loopOutCycle(pos{i}_{j});\n")
                    # Radius (size) — keyframe only when it varies. Circles use
                    # the ellipse [w, h] size; polystars use a scalar outer radius
                    # (plus a proportional inner radius for stars).
                    def _size_val(r, _circle=is_circle):
                        return f"[{r * 2}, {r * 2}]" if _circle else f"{r}"
                    if radius_varies:
                        for k, tk in enumerate(times):
                            r = radius_frames[k][j]
                            script.append(f"markerSize{i}_{j}.setValueAtTime({tk}, {_size_val(r)});\n")
                        script.append(f"setKeyInterp(markerSize{i}_{j}, {hold_js});\n")
                        if elem.get("loop"):
                            script.append(f"loopOutCycle(markerSize{i}_{j});\n")
                        if inner_ratio is not None:
                            for k, tk in enumerate(times):
                                r = radius_frames[k][j]
                                script.append(f"markerInner{i}_{j}.setValueAtTime({tk}, {r * inner_ratio});\n")
                            script.append(f"setKeyInterp(markerInner{i}_{j}, {hold_js});\n")
                            if elem.get("loop"):
                                script.append(f"loopOutCycle(markerInner{i}_{j});\n")
                    else:
                        r = radius_frames[0][j]
                        script.append(f"markerSize{i}_{j}.setValue({_size_val(r)});\n")
                        if inner_ratio is not None:
                            script.append(f"markerInner{i}_{j}.setValue({r * inner_ratio});\n")
                    # Color — keyframe when color_frames given, else static.
                    if color_frames is not None:
                        for k, tk in enumerate(times):
                            c = color_frames[k][j]
                            script.append(f"fillColor{i}_{j}.setValueAtTime({tk}, [{c[0]}, {c[1]}, {c[2]}]);\n")
                        script.append(f"setKeyInterp(fillColor{i}_{j}, {hold_js});\n")
                        if elem.get("loop"):
                            script.append(f"loopOutCycle(fillColor{i}_{j});\n")
                        if draw_outline:
                            for k, tk in enumerate(times):
                                oc = self._darken_color(color_frames[k][j])
                                script.append(f"strokeColor{i}_{j}.setValueAtTime({tk}, [{oc[0]}, {oc[1]}, {oc[2]}]);\n")
                            script.append(f"setKeyInterp(strokeColor{i}_{j}, {hold_js});\n")
                            if elem.get("loop"):
                                script.append(f"loopOutCycle(strokeColor{i}_{j});\n")
                    else:
                        script.append(f"fillColor{i}_{j}.setValue({base_color_js});\n")
                        if draw_outline:
                            outline_color = elem.get("outline_color") or self._darken_color(elem["color"])
                            script.append(f"strokeColor{i}_{j}.setValue({color_to_js(outline_color)});\n")
                    if fade and fade > 0:
                        # Fade in with everything else (un-delayed); only the
                        # movement / size / color keyframes carry the ``delay``.
                        t0 = times[0] - elem.get("delay", 0.0)
                        script.append(f"var scOp{i}_{j} = scatterLayer{i}_{j}.property('Transform').property('Opacity');\n")
                        script.append(f"scOp{i}_{j}.setValueAtTime({t0}, 0);\n")
                        script.append(f"scOp{i}_{j}.setValueAtTime({t0 + fade}, 100);\n")
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"scatterLayer{i}_{j}", f"{i}_{j}"))
            elif elem["type"] == "quiver_evolving":
                bx = elem["qx"]; by = elem["qy"]
                u_frames = elem["u_frames"]; v_frames = elem["v_frames"]
                qrel = elem["qrel"]
                color_frames = elem.get("color_frames")
                times = elem["frame_times"]
                hold_js = "true" if elem["hold"] else "false"
                n_arrows = elem["n_points"]
                scale_mode = elem.get("scale_mode", "data")
                qscale = elem.get("qscale", 1.0)
                width = elem.get("width", 3.0)
                hw_mult = elem.get("headwidth", 3.0)
                hl_px = elem.get("headlength", 11.0)
                alpha = elem.get("alpha", 1.0)
                pivot = elem.get("pivot", "tail")
                base_color_js = color_to_js(elem["color"])
                fade = elem.get("fade_in", 0.0)
                plot_w = float(self.width)

                def _arrow_verts(bxj, byj, uu, vv, rel):
                    """7-vertex arrow polygon (shape coords) for one frame."""
                    if scale_mode == "comp":
                        bsx, bsy = self._data_to_shape(bxj, byj, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        tsx, tsy = self._data_to_shape(bxj + uu, byj + vv, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        ddx, ddy = tsx - bsx, tsy - bsy
                        Ld = float(np.hypot(ddx, ddy))
                        ux, uy = (1.0, 0.0) if Ld < 1e-9 else (ddx / Ld, ddy / Ld)
                        Lp = max(qscale * plot_w * rel, 1e-4)
                        if pivot == "mid":
                            sx0, sy0 = bsx - ux * Lp / 2.0, bsy - uy * Lp / 2.0
                            sx1, sy1 = bsx + ux * Lp / 2.0, bsy + uy * Lp / 2.0
                        elif pivot == "tip":
                            sx0, sy0 = bsx - ux * Lp, bsy - uy * Lp
                            sx1, sy1 = bsx, bsy
                        else:
                            sx0, sy0 = bsx, bsy
                            sx1, sy1 = bsx + ux * Lp, bsy + uy * Lp
                    else:
                        vx, vy = uu * qscale, vv * qscale
                        if pivot == "mid":
                            x0, y0 = bxj - vx / 2.0, byj - vy / 2.0
                            x1, y1 = bxj + vx / 2.0, byj + vy / 2.0
                        elif pivot == "tip":
                            x0, y0 = bxj - vx, byj - vy
                            x1, y1 = bxj, byj
                        else:
                            x0, y0 = bxj, byj
                            x1, y1 = bxj + vx, byj + vy
                        sx0, sy0 = self._data_to_shape(x0, y0, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        sx1, sy1 = self._data_to_shape(x1, y1, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        Lp = float(np.hypot(sx1 - sx0, sy1 - sy0))
                        if Lp < 1e-9:
                            sx1, sy1, Lp = sx0 + 0.01, sy0, 0.01
                        ux, uy = (sx1 - sx0) / Lp, (sy1 - sy0) / Lp
                    ppx, ppy = -uy, ux
                    sw = width / 2.0
                    hw = (width * hw_mult) / 2.0
                    hl = min(hl_px, Lp * 0.85)
                    hbx, hby = sx1 - ux * hl, sy1 - uy * hl
                    verts = [
                        (sx0 + ppx * sw, sy0 + ppy * sw),
                        (hbx + ppx * sw, hby + ppy * sw),
                        (hbx + ppx * hw, hby + ppy * hw),
                        (sx1, sy1),
                        (hbx - ppx * hw, hby - ppy * hw),
                        (hbx - ppx * sw, hby - ppy * sw),
                        (sx0 - ppx * sw, sy0 - ppy * sw),
                    ]
                    return ",".join(f"[{vxp},{vyp}]" for vxp, vyp in verts)

                for j in range(n_arrows):
                    script.append(f"var qeLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"qeLayer{i}_{j}.name = 'QuiverEvolving_{i}_{j}';\n")
                    script.append(f"qeLayer{i}_{j}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                    script.append(f"qeLayer{i}_{j}.parent = PlotAnchor;\n")
                    self._emit_screen_space_mask_jsx(script, f"qeLayer{i}_{j}")
                    script.append(f"var qeContents{i}_{j} = qeLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var qeGroup{i}_{j} = qeContents{i}_{j}.addProperty('ADBE Vector Shape - Group');\n")
                    script.append(f"var qePath{i}_{j} = qeGroup{i}_{j}.property('ADBE Vector Shape');\n")
                    for k, tk in enumerate(times):
                        pts_js = _arrow_verts(bx[j], by[j], u_frames[k][j], v_frames[k][j], qrel[k][j])
                        script.append(f"var qeShp{i}_{j}_{k} = new Shape();\n")
                        script.append(f"qeShp{i}_{j}_{k}.vertices = [{pts_js}];\n")
                        script.append(f"qeShp{i}_{j}_{k}.closed = true;\n")
                        script.append(f"qePath{i}_{j}.setValueAtTime({tk}, qeShp{i}_{j}_{k});\n")
                    script.append(f"setKeyInterp(qePath{i}_{j}, {hold_js});\n")
                    script.append(f"var qeFill{i}_{j} = qeContents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    script.append(f"qeFill{i}_{j}.property('ADBE Vector Fill Opacity').setValue({int(alpha * 100)});\n")
                    script.append(f"var qeFillColor{i}_{j} = qeFill{i}_{j}.property('ADBE Vector Fill Color');\n")
                    if color_frames is not None:
                        for k, tk in enumerate(times):
                            c = color_frames[k][j]
                            script.append(f"qeFillColor{i}_{j}.setValueAtTime({tk}, [{c[0]}, {c[1]}, {c[2]}]);\n")
                        script.append(f"setKeyInterp(qeFillColor{i}_{j}, {hold_js});\n")
                    else:
                        script.append(f"qeFillColor{i}_{j}.setValue({base_color_js});\n")
                    if fade and fade > 0:
                        # Fade in with everything else (un-delayed); only the
                        # arrow animation keyframes carry the ``delay`` offset.
                        t0 = times[0] - elem.get("delay", 0.0)
                        script.append(f"var qeOp{i}_{j} = qeLayer{i}_{j}.property('Transform').property('Opacity');\n")
                        script.append(f"qeOp{i}_{j}.setValueAtTime({t0}, 0);\n")
                        script.append(f"qeOp{i}_{j}.setValueAtTime({t0 + fade}, 100);\n")
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"qeLayer{i}_{j}", f"qe{i}_{j}"))
            elif elem["type"] == "evolving_text":
                et_data = elem["data"]
                et_times = elem["frame_times"]
                et_hold_js = "true" if elem["hold"] else "false"
                et_fs = int(elem["fontsize"] * self.font_scale)
                et_font = elem["font"] or self.font_body
                et_color_js = color_to_js(elem["color"] if elem["color"] is not None else self.ui_color)
                et_fade = elem.get("fade_in", 0.0)
                et_t0 = et_times[0]

                # Vertical position from data y only (x is unused; avoid log-scale
                # errors when horizontal_alignment pins to the graph edge).
                et_ny = self._data_norm(elem["loc_y"], ymin_pad, ymax_pad, self.yscale)
                et_sy = self.height - self.height * et_ny - self.height / 2
                et_pos_y = center_y + et_sy
                graph_half_w = self.width / 2.0
                # Small inset so edge-aligned text isn't smushed against the
                # plot border; centered text needs no gap.
                et_edge_gap = 0.025 * self.width
                et_halign = elem.get("horizontal_alignment", elem.get("alignment", "left"))
                if et_halign == "center":
                    et_pos_x = center_x
                elif et_halign == "right":
                    et_pos_x = center_x + graph_half_w - et_edge_gap
                else:
                    et_pos_x = center_x - graph_half_w + et_edge_gap

                alignment_map = {
                    "left": "ParagraphJustification.LEFT_JUSTIFY",
                    "center": "ParagraphJustification.CENTER_JUSTIFY",
                    "right": "ParagraphJustification.RIGHT_JUSTIFY",
                }
                et_just = alignment_map.get(et_halign, "ParagraphJustification.LEFT_JUSTIFY")

                # Build the Source Text expression. The before/after/delimiter
                # values become single-quoted JS string literals (escaped), and
                # the whole expression is then escaped for the JSX string literal.
                def _js_sq(s):
                    return s.replace("\\", "\\\\").replace("'", "\\'")

                expr = (
                    "s = '" + _js_sq(elem["before"]) + "' + Math.round(effect(1)(1)[0]) + '"
                    + _js_sq(elem["after"]) + "';\n"
                    "s.replace(/\\B(?=(\\d{3})+(?!\\d))/g, '" + _js_sq(elem["delimiter"]) + "');"
                )
                expr_jsx = expr.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

                script.append(f"var etLayer{i} = comp.layers.addText(\"0\");\n")
                script.append(f"etLayer{i}.name = \"EvolvingText_{i}\";\n")

                # Style the text document.
                script.append(f"var etProp{i} = etLayer{i}.property('Source Text');\n")
                script.append(f"var etDoc{i} = etProp{i}.value;\n")
                script.append(f"etDoc{i}.fontSize = {et_fs};\n")
                script.append(f"etDoc{i}.font = \"{et_font}\";\n")
                script.append(f"etDoc{i}.fillColor = {et_color_js};\n")
                script.append(f"etDoc{i}.justification = {et_just};\n")
                script.append(f"etProp{i}.setValue(etDoc{i});\n")

                # Add and keyframe the Slider Control that drives the value.
                script.append(f"var etSlider{i} = etLayer{i}.property('Effects').addProperty('ADBE Slider Control');\n")
                script.append(f"etSlider{i}.name = \"Value\";\n")
                script.append(f"var etSliderVal{i} = etSlider{i}.property(1);\n")
                for tk, val in zip(et_times, et_data):
                    script.append(f"etSliderVal{i}.setValueAtTime({tk}, {val});\n")
                script.append(f"setKeyInterp(etSliderVal{i}, {et_hold_js});\n")

                # Drive the displayed text from the slider.
                script.append(f"etProp{i}.expression = \"{expr_jsx}\";\n")

                # Anchor point from alignment, measured once the expression is live.
                script.append(f"var etSR{i} = etLayer{i}.sourceRectAtTime({et_t0}, false);\n")
                script.append(f"var etAP{i} = etLayer{i}.property('Transform').property('Anchor Point');\n")
                if et_just == "ParagraphJustification.LEFT_JUSTIFY":
                    et_anchor_x = f"etSR{i}.left"
                elif et_just == "ParagraphJustification.CENTER_JUSTIFY":
                    et_anchor_x = f"etSR{i}.left + etSR{i}.width/2"
                else:
                    et_anchor_x = f"etSR{i}.left + etSR{i}.width"
                et_valign = elem.get("vertical_alignment", "center")
                if et_valign == "top":
                    et_anchor_y = f"etSR{i}.top"
                elif et_valign == "bottom":
                    et_anchor_y = f"etSR{i}.top + etSR{i}.height"
                else:
                    et_anchor_y = f"etSR{i}.top + etSR{i}.height/2"
                script.append(f"etAP{i}.setValue([{et_anchor_x}, {et_anchor_y}]);\n")
                script.append(f"etLayer{i}.property('Transform').property('Position').setValue([{et_pos_x}, {et_pos_y}]);\n")
                script.append(f"etLayer{i}.parent = PlotAnchor;\n")

                # Fade in with everything else (un-delayed); the slider holds the
                # first value until the animation starts at ``delay``.
                if et_fade and et_fade > 0:
                    et_fade_t0 = et_t0 - elem.get("delay", 0.0)
                    script.append(f"var etOp{i} = etLayer{i}.property('Transform').property('Opacity');\n")
                    script.append(f"etOp{i}.setValueAtTime({et_fade_t0}, 0);\n")
                    script.append(f"etOp{i}.setValueAtTime({et_fade_t0 + et_fade}, 100);\n")
                else:
                    script.append(f"etLayer{i}.property('Transform').property('Opacity').setValue(100);\n")

                if self.drop_shadow:
                    script.append(self._generate_drop_shadow_jsx(f"etLayer{i}", f"EvolvingText{i}"))
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
                # Ease the sequence of bar entrances so the sweep itself eases in/out.
                start_times = self._apply_meta_ease(start_times, elem)

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
                    self._emit_screen_space_mask_jsx(script, f"histLayer{i}_{j}")
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
            elif elem["type"] == "bar_stacked":
                alpha = elem.get("alpha", 0.9)
                bin_left = elem["bin_left"]
                bin_right = elem["bin_right"]
                series = elem["series"]
                colors = elem["colors"]
                n_layers = len(series)
                n_cols = len(bin_left)
                total_anim = elem.get("animate") or 0.0
                delay = elem.get("delay", 0.0)
                # Same smoothstep cumulative reveal as pie(): one shared eased
                # schedule per column; each stack segment fans open while the
                # global progress crosses its fraction of the column total.
                if total_anim and total_anim > 0:
                    n_samp = 72
                    us = [j / n_samp for j in range(n_samp + 1)]
                    gs = [u * u * (3 - 2 * u) for u in us]
                    tsamp = [delay + u * total_anim for u in us]
                for j in range(n_cols):
                    left = bin_left[j]
                    right = bin_right[j]
                    layer_heights = [series[k][j] for k in range(n_layers)]
                    total_h = sum(layer_heights)
                    if total_h <= 0:
                        continue
                    fracs = [h / total_h for h in layer_heights]
                    cum = 0.0
                    running = 0.0
                    for k in range(n_layers):
                        h = layer_heights[k]
                        cum_before = cum
                        cum += fracs[k]
                        cum_after = cum
                        base = running
                        running += h
                        x0, y0 = left, base
                        x1, y1 = right, base + h
                        sx0, sy0 = self._data_to_shape(x0, y0, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        sx1, sy1 = self._data_to_shape(x1, y1, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                        bar_width_px = abs(sx1 - sx0)
                        full_h_px = abs(sy1 - sy0)
                        position_x = center_x + (sx0 + sx1) / 2
                        position_y = center_y + sy0
                        col_js = color_to_js(colors[k])
                        script.append(f"var stkLayer{i}_{j}_{k} = comp.layers.addShape();\n")
                        script.append(f"stkLayer{i}_{j}_{k}.name = 'StackBar_{i}_{j}_{k}';\n")
                        script.append(f"stkLayer{i}_{j}_{k}.property('Transform').property('Position').setValue([{position_x}, {position_y}]);\n")
                        script.append(f"stkLayer{i}_{j}_{k}.parent = PlotAnchor;\n")
                        self._emit_screen_space_mask_jsx(script, f"stkLayer{i}_{j}_{k}")
                        script.append(f"var stkContents{i}_{j}_{k} = stkLayer{i}_{j}_{k}.property('ADBE Root Vectors Group');\n")
                        script.append(f"var stkRect{i}_{j}_{k} = stkContents{i}_{j}_{k}.addProperty('ADBE Vector Shape - Rect');\n")
                        rect_size = f"stkRect{i}_{j}_{k}.property('ADBE Vector Rect Size')"
                        rect_pos = f"stkRect{i}_{j}_{k}.property('ADBE Vector Rect Position')"
                        if total_anim and total_anim > 0:
                            width = cum_after - cum_before
                            j_enter = 0
                            for jj in range(n_samp + 1):
                                if gs[jj] <= cum_before:
                                    j_enter = jj
                                else:
                                    break
                            j_exit = n_samp
                            for jj in range(n_samp + 1):
                                if gs[jj] >= cum_after:
                                    j_exit = jj
                                    break
                            if j_enter > 0:
                                script.append(f"{rect_size}.setValueAtTime({tsamp[0]}, [{bar_width_px}, 0]);\n")
                                script.append(f"{rect_pos}.setValueAtTime({tsamp[0]}, [0, 0]);\n")
                            for jj in range(j_enter, j_exit + 1):
                                frac_in = (min(max(gs[jj] - cum_before, 0.0), width) / width) if width > 0 else 1.0
                                cur_h = full_h_px * frac_in
                                off_y = (sy1 - sy0) * frac_in / 2
                                script.append(f"{rect_size}.setValueAtTime({tsamp[jj]}, [{bar_width_px}, {cur_h}]);\n")
                                script.append(f"{rect_pos}.setValueAtTime({tsamp[jj]}, [0, {off_y}]);\n")
                        else:
                            script.append(f"{rect_size}.setValue([{bar_width_px}, {full_h_px}]);\n")
                            script.append(f"{rect_pos}.setValue([0, {(sy1 - sy0) / 2}]);\n")
                        script.append(f"var stkFill{i}_{j}_{k} = stkContents{i}_{j}_{k}.addProperty('ADBE Vector Graphic - Fill');\n")
                        script.append(f"stkFill{i}_{j}_{k}.property('ADBE Vector Fill Color').setValue({col_js});\n")
                        script.append(f"stkFill{i}_{j}_{k}.property('ADBE Vector Fill Opacity').setValue({int(alpha * 100)});\n")
                        if elem.get("drop_shadow", False):
                            script.append(self._generate_drop_shadow_jsx(f"stkLayer{i}_{j}_{k}", f"{i}_{j}_{k}"))
            elif elem["type"] == "bar_graph":
                alpha = elem.get("alpha", 0.8)
                bin_left = elem["bin_left"]
                bin_right = elem["bin_right"]
                heights = elem["heights"]
                baseline = elem.get("baseline") or [0] * len(bin_left)
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
                # Ease the sequence of bar entrances so the sweep itself eases in/out.
                start_times = self._apply_meta_ease(start_times, elem)

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

                    # Rectangle corners in data coordinates. ``base`` is the
                    # per-bar baseline (0 normally; running total when stacked).
                    base = baseline[j]
                    x0, y0 = left, base  # bottom left (baseline)
                    x1, y1 = right, base + height  # top right

                    # Geometry of this bar for a given view window. Returns the
                    # pixel width/height, the (parent-relative) baseline-center
                    # anchor, and the signed half-height offset that positions
                    # the rect above/below its baseline. Recomputed per view so
                    # the bar pans/zooms with an animated view window.
                    def _bar_geom(xmn, xmx, ymn, ymx, _x0=x0, _y0=y0, _x1=x1, _y1=y1):
                        gsx0, gsy0 = self._data_to_shape(_x0, _y0, xmn, xmx, ymn, ymx)
                        gsx1, gsy1 = self._data_to_shape(_x1, _y1, xmn, xmx, ymn, ymx)
                        return (
                            abs(gsx1 - gsx0),        # bar_width
                            abs(gsy1 - gsy0),        # bar_height
                            (gsx0 + gsx1) / 2,       # anchor_x (center)
                            gsy0,                    # anchor_y (baseline)
                            (gsy1 - gsy0) / 2,       # rect_offset_y (signed)
                        )

                    # Static geometry at the reference view (== first view
                    # keyframe when the view is animated).
                    bar_width, bar_height, anchor_x, anchor_y, rect_offset_y = _bar_geom(
                        xmin_pad, xmax_pad, ymin_pad, ymax_pad
                    )
                    position_x = center_x + anchor_x
                    position_y = center_y + anchor_y
                    script.append(f"var barLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"barLayer{i}_{j}.name = 'Bar_{i}_{j}';\n")
                    script.append(f"barLayer{i}_{j}.property('Transform').property('Position').setValue([{position_x}, {position_y}]);\n")
                    script.append(f"barLayer{i}_{j}.parent = PlotAnchor;\n")
                    self._emit_screen_space_mask_jsx(script, f"barLayer{i}_{j}")
                    script.append(f"var barContents{i}_{j} = barLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var barRect{i}_{j} = barContents{i}_{j}.addProperty('ADBE Vector Shape - Rect');\n")
                    script.append(f"barRect{i}_{j}.property('ADBE Vector Rect Size').setValue([{bar_width}, {bar_height}]);\n")
                    script.append(f"barRect{i}_{j}.property('ADBE Vector Rect Position').setValue([0, {rect_offset_y}]);\n")
                    # Animated view window: keyframe the layer position, rect
                    # size, and rect offset so the bar tracks the pan/zoom. The
                    # grow-up Scale entrance (below) is independent and composes
                    # multiplicatively, so both animations coexist.
                    if self._view_animated:
                        pos_var = f"barPos{i}_{j}"
                        size_var = f"barRectSize{i}_{j}"
                        off_var = f"barRectPos{i}_{j}"
                        script.append(f"var {pos_var} = barLayer{i}_{j}.property('Transform').property('Position');\n")
                        script.append(f"var {size_var} = barRect{i}_{j}.property('ADBE Vector Rect Size');\n")
                        script.append(f"var {off_var} = barRect{i}_{j}.property('ADBE Vector Rect Position');\n")
                        for _t in self._view_sample_times():
                            _vx0, _vx1, _vy0, _vy1 = self._view_limits_at(_t)
                            _bw, _bh, _ax, _ay, _roff = _bar_geom(_vx0, _vx1, _vy0, _vy1)
                            script.append(f"{pos_var}.setValueAtTime({_t}, [{_ax}, {_ay}]);\n")
                            script.append(f"{size_var}.setValueAtTime({_t}, [{_bw}, {_bh}]);\n")
                            script.append(f"{off_var}.setValueAtTime({_t}, [0, {_roff}]);\n")
                        _interp = 'true' if self._view_hold else 'false'
                        script.append(f"setKeyInterp({pos_var}, {_interp});\n")
                        script.append(f"setKeyInterp({size_var}, {_interp});\n")
                        script.append(f"setKeyInterp({off_var}, {_interp});\n")
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
                # Ease the sequence of bar entrances so the sweep itself eases in/out.
                start_times = self._apply_meta_ease(start_times, elem)

                # Check if gradient colors are defined
                gradient_colors = elem.get("gradient_colors")
                if not gradient_colors:
                    # Use single color for all bars
                    color_js = color_to_js(elem["color"])
                baseline = elem.get("baseline") or [0] * n_bars

                for j, (bottom, top, width) in enumerate(zip(bin_bottom, bin_top, widths)):
                    # Use gradient color for this bar if available
                    if gradient_colors and j < len(gradient_colors):
                        bar_color_js = f"[{gradient_colors[j][0]}, {gradient_colors[j][1]}, {gradient_colors[j][2]}]"
                    else:
                        bar_color_js = color_js

                    # Per-bar baseline (0 normally; running total when stacked).
                    # Ignored for pyramid bars that anchor at the y-axis.
                    base = 0 if anchor_at_y_axis else baseline[j]
                    # Rectangle corners in data coordinates (horizontal bars)
                    # Apply horizontal clipping to [xmin_pad, xmax_pad] when show_all_points is False
                    clip = not self.show_all_points
                    if width >= 0:
                        bar_start, bar_end = base, base + width
                    else:
                        bar_start, bar_end = base + width, base
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
                    self._emit_screen_space_mask_jsx(script, f"barhLayer{i}_{j}")
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

            elif elem["type"] == "barh_evolving":
                # Horizontal bars whose individual widths morph across frames.
                alpha = elem.get("alpha", 0.9)
                bin_bottom = elem["bin_bottom"]
                bin_top = elem["bin_top"]
                width_frames = elem["width_frames"]
                frame_times = elem["frame_times"]
                hold = elem.get("hold", False)
                n_bars = len(bin_bottom)
                n_frames = len(width_frames)
                total_anim = elem["animate"] if elem["animate"] else 1.0
                bar_duration = elem.get("bar_anim_times")
                animate_downward = elem.get("animate_downward", True)
                anchor_at_y_axis = elem.get("anchor_at_y_axis", True)
                delay = elem.get("delay", 0.0)
                elem_ease_speed = self.ease_speed if elem.get("ease_speed") is None else elem.get("ease_speed")
                elem_ease_influence = self.ease_influence if elem.get("ease_influence") is None else elem.get("ease_influence")
                gradient_colors = elem.get("gradient_colors")
                color_frames = elem.get("color_frames")
                if not gradient_colors and color_frames is None:
                    color_js = color_to_js(elem["color"])
                hold_js = "true" if hold else "false"

                # Per-bar grow-in duration and staggered start times so each bar
                # animates individually within [0, total_anim] (like barh).
                dur = float(bar_duration) if isinstance(bar_duration, (int, float)) else (total_anim / max(n_bars, 1))
                if n_bars > 1:
                    start_times_base = np.linspace(0, max(0.0, total_anim - dur), n_bars)
                else:
                    start_times_base = [0.0]
                centers = np.asarray(elem.get("bin_centers", []))
                if animate_downward and len(centers) == n_bars:
                    order = np.argsort(centers)[::-1]  # top to bottom
                    start_times = [0.0] * n_bars
                    for rank, j_idx in enumerate(order):
                        start_times[j_idx] = float(start_times_base[rank])
                else:
                    start_times = list(start_times_base)
                # Ease the sequence of bar entrances so the sweep itself eases in/out.
                start_times = self._apply_meta_ease(start_times, elem)

                ft0 = frame_times[0]
                for j, (bottom, top) in enumerate(zip(bin_bottom, bin_top)):
                    col = [width_frames[f][j] for f in range(n_frames)]
                    abs_col = [abs(v) for v in col]
                    max_w = max(abs_col) if abs_col else 0.0
                    if max_w <= 1e-12:
                        continue
                    ref_signed = col[int(np.argmax(abs_col))]
                    sign = 1.0 if ref_signed >= 0 else -1.0

                    center_y_data = (bottom + top) / 2
                    axis_sx, _ = self._data_to_shape(0, center_y_data, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    tip_sx, _ = self._data_to_shape(sign * max_w, center_y_data, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    _, sy_b = self._data_to_shape(0, bottom, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    _, sy_t = self._data_to_shape(0, top, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    bar_width = abs(tip_sx - axis_sx)
                    bar_height_shape = abs(sy_t - sy_b)
                    anchor_y = (sy_b + sy_t) / 2
                    position_x = center_x + axis_sx
                    position_y = center_y + anchor_y
                    rect_off = sign * bar_width / 2 if anchor_at_y_axis else bar_width / 2

                    script.append(f"var barhevLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"barhevLayer{i}_{j}.name = 'BarhEvolving_{i}_{j}';\n")
                    script.append(f"barhevLayer{i}_{j}.property('Transform').property('Position').setValue([{position_x}, {position_y}]);\n")
                    script.append(f"barhevLayer{i}_{j}.parent = PlotAnchor;\n")
                    self._emit_screen_space_mask_jsx(script, f"barhevLayer{i}_{j}")
                    script.append(f"var barhevContents{i}_{j} = barhevLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var barhevRect{i}_{j} = barhevContents{i}_{j}.addProperty('ADBE Vector Shape - Rect');\n")
                    script.append(f"barhevRect{i}_{j}.property('ADBE Vector Rect Size').setValue([{bar_width}, {bar_height_shape}]);\n")
                    script.append(f"barhevRect{i}_{j}.property('ADBE Vector Rect Position').setValue([{rect_off}, 0]);\n")
                    script.append(f"var barhevFill{i}_{j} = barhevContents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    fill_prop = f"barhevFill{i}_{j}.property('ADBE Vector Fill Color')"
                    if color_frames is not None:
                        for f, tk in enumerate(frame_times):
                            c = color_frames[f][j]
                            script.append(f"{fill_prop}.setValueAtTime({tk}, [{c[0]}, {c[1]}, {c[2]}]);\n")
                        script.append(f"setKeyInterp({fill_prop}, {hold_js});\n")
                        if elem.get("loop"):
                            script.append(f"loopOutCycle({fill_prop});\n")
                    elif gradient_colors and j < len(gradient_colors):
                        gc = gradient_colors[j]
                        script.append(f"{fill_prop}.setValue([{gc[0]}, {gc[1]}, {gc[2]}]);\n")
                    else:
                        script.append(f"{fill_prop}.setValue({color_js});\n")
                    script.append(f"barhevFill{i}_{j}.property('ADBE Vector Fill Opacity').setValue({int(alpha*100)});\n")

                    # Scale X encodes the bar's fraction of its max width: grow-in
                    # from 0 -> first frame, then morph across the remaining frames.
                    # When ``loop`` is set, skip the grow-in so loopOut only cycles
                    # the morph (otherwise the entrance would replay every lap).
                    script.append(f"var barhevScale{i}_{j} = barhevLayer{i}_{j}.property('Transform').property('Scale');\n")
                    if elem.get("loop"):
                        for f in range(n_frames):
                            frac = abs_col[f] / max_w * 100.0
                            script.append(f"barhevScale{i}_{j}.setValueAtTime({frame_times[f]}, [{frac}, 100, 100]);\n")
                        script.append(f"setKeyInterp(barhevScale{i}_{j}, {hold_js});\n")
                        script.append(f"loopOutCycle(barhevScale{i}_{j});\n")
                    else:
                        t_a = delay + start_times[j]
                        t_b = t_a + dur
                        frac0 = abs_col[0] / max_w * 100.0
                        script.append(f"barhevScale{i}_{j}.setValueAtTime({t_a}, [0, 100, 100]);\n")
                        script.append(f"barhevScale{i}_{j}.setValueAtTime({t_b}, [{frac0}, 100, 100]);\n")
                        if ft0 > t_b + 1e-6:
                            script.append(f"barhevScale{i}_{j}.setValueAtTime({ft0}, [{frac0}, 100, 100]);\n")
                        for f in range(1, n_frames):
                            frac = abs_col[f] / max_w * 100.0
                            script.append(f"barhevScale{i}_{j}.setValueAtTime({frame_times[f]}, [{frac}, 100, 100]);\n")
                        if self.easy_ease:
                            script.append(f"applyEasyEase(barhevScale{i}_{j}, {elem_ease_speed}, {elem_ease_influence});\n")
                        if hold and n_frames > 1:
                            # Snap year-to-year (keys at/after the first settle time)
                            # while keeping the eased grow-in intact.
                            script.append(
                                f"(function(p){{ for (var k=1;k<=p.numKeys;k++){{ if (p.keyTime(k) >= {ft0} - 1e-6) {{ p.setInterpolationTypeAtKey(k, KeyframeInterpolationType.HOLD, KeyframeInterpolationType.HOLD); }} }} }})(barhevScale{i}_{j});\n"
                            )
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"barhevLayer{i}_{j}", f"ev{i}_{j}"))

            elif elem["type"] == "bar_evolving":
                # Vertical bars whose individual heights morph across frames.
                alpha = elem.get("alpha", 0.9)
                bin_left = elem["bin_left"]
                bin_right = elem["bin_right"]
                height_frames = elem["height_frames"]
                frame_times = elem["frame_times"]
                hold = elem.get("hold", False)
                n_bars = len(bin_left)
                n_frames = len(height_frames)
                total_anim = elem["animate"] if elem["animate"] else 1.0
                bar_duration = elem.get("bar_anim_times")
                anchor_at_x_axis = elem.get("anchor_at_x_axis", True)
                delay = elem.get("delay", 0.0)
                elem_ease_speed = self.ease_speed if elem.get("ease_speed") is None else elem.get("ease_speed")
                elem_ease_influence = self.ease_influence if elem.get("ease_influence") is None else elem.get("ease_influence")
                color_js = color_to_js(elem["color"])
                hold_js = "true" if hold else "false"

                # Per-bar grow-in duration and staggered start times (left->right).
                dur = float(bar_duration) if isinstance(bar_duration, (int, float)) else (total_anim / max(n_bars, 1))
                if n_bars > 1:
                    start_times_base = np.linspace(0, max(0.0, total_anim - dur), n_bars)
                else:
                    start_times_base = [0.0]
                centers = np.asarray(elem.get("bin_centers", []))
                if len(centers) == n_bars:
                    order = np.argsort(centers)  # left to right
                    start_times = [0.0] * n_bars
                    for rank, j_idx in enumerate(order):
                        start_times[j_idx] = float(start_times_base[rank])
                else:
                    start_times = list(start_times_base)
                start_times = self._apply_meta_ease(start_times, elem)

                ft0 = frame_times[0]
                for j in range(n_bars):
                    left = bin_left[j]
                    right = bin_right[j]
                    col = [height_frames[f][j] for f in range(n_frames)]
                    abs_col = [abs(v) for v in col]
                    max_h = max(abs_col) if abs_col else 0.0
                    if max_h <= 1e-12:
                        continue
                    ref_signed = col[int(np.argmax(abs_col))]
                    sign = 1.0 if ref_signed >= 0 else -1.0

                    center_x_data = (left + right) / 2
                    sx_l, _ = self._data_to_shape(left, 0, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    sx_r, _ = self._data_to_shape(right, 0, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    _, axis_sy = self._data_to_shape(center_x_data, 0, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    _, tip_sy = self._data_to_shape(center_x_data, sign * max_h, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    bar_width_shape = abs(sx_r - sx_l)
                    bar_height_shape = abs(tip_sy - axis_sy)
                    anchor_x = (sx_l + sx_r) / 2
                    position_x = center_x + anchor_x
                    position_y = center_y + axis_sy
                    # Rect centered between the axis and the tip (tip is "up",
                    # i.e. negative dy in shape space when sign > 0).
                    rect_off_y = (tip_sy - axis_sy) / 2 if anchor_at_x_axis else -sign * bar_height_shape / 2

                    script.append(f"var barevLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"barevLayer{i}_{j}.name = 'BarEvolving_{i}_{j}';\n")
                    script.append(f"barevLayer{i}_{j}.property('Transform').property('Position').setValue([{position_x}, {position_y}]);\n")
                    script.append(f"barevLayer{i}_{j}.parent = PlotAnchor;\n")
                    self._emit_screen_space_mask_jsx(script, f"barevLayer{i}_{j}")
                    script.append(f"var barevContents{i}_{j} = barevLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var barevRect{i}_{j} = barevContents{i}_{j}.addProperty('ADBE Vector Shape - Rect');\n")
                    script.append(f"barevRect{i}_{j}.property('ADBE Vector Rect Size').setValue([{bar_width_shape}, {bar_height_shape}]);\n")
                    script.append(f"barevRect{i}_{j}.property('ADBE Vector Rect Position').setValue([0, {rect_off_y}]);\n")
                    script.append(f"var barevFill{i}_{j} = barevContents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    fill_prop = f"barevFill{i}_{j}.property('ADBE Vector Fill Color')"
                    script.append(f"{fill_prop}.setValue({color_js});\n")
                    script.append(f"barevFill{i}_{j}.property('ADBE Vector Fill Opacity').setValue({int(alpha*100)});\n")

                    # Scale Y encodes the bar's fraction of its max height:
                    # grow-in from 0 -> first frame, then morph across frames.
                    t_a = delay + start_times[j]
                    t_b = t_a + dur
                    frac0 = abs_col[0] / max_h * 100.0
                    script.append(f"var barevScale{i}_{j} = barevLayer{i}_{j}.property('Transform').property('Scale');\n")
                    script.append(f"barevScale{i}_{j}.setValueAtTime({t_a}, [100, 0, 100]);\n")
                    script.append(f"barevScale{i}_{j}.setValueAtTime({t_b}, [100, {frac0}, 100]);\n")
                    if ft0 > t_b + 1e-6:
                        script.append(f"barevScale{i}_{j}.setValueAtTime({ft0}, [100, {frac0}, 100]);\n")
                    for f in range(1, n_frames):
                        frac = abs_col[f] / max_h * 100.0
                        script.append(f"barevScale{i}_{j}.setValueAtTime({frame_times[f]}, [100, {frac}, 100]);\n")
                    if self.easy_ease:
                        script.append(f"applyEasyEase(barevScale{i}_{j}, {elem_ease_speed}, {elem_ease_influence});\n")
                    if hold and n_frames > 1:
                        # Snap year-to-year (keys at/after the first settle time)
                        # while keeping the eased grow-in intact.
                        script.append(
                            f"(function(p){{ for (var k=1;k<=p.numKeys;k++){{ if (p.keyTime(k) >= {ft0} - 1e-6) {{ p.setInterpolationTypeAtKey(k, KeyframeInterpolationType.HOLD, KeyframeInterpolationType.HOLD); }} }} }})(barevScale{i}_{j});\n"
                        )
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"barevLayer{i}_{j}", f"barev{i}_{j}"))

        # Generate axes AFTER plot elements for histograms and bar graphs (so they appear on top)
        # Mixed plots get axes on top to accommodate bar/histogram rendering
        if (has_bars_or_hist or not has_scatter) and not pie_only:
            self._generate_axes_jsx(script, center_x, center_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad, ANIM_DURATION, has_barh)

        # --- DYNAMIC LEGEND PLACEMENT (relative to graph area) ---
        legend_entries = []
        legend_colors = []
        legend_styles = []  # Store line styles for line_style legend
        legend_widths = []  # Store line widths for line_style legend
        _legend_seen = set()
        for elem in self.elements:
            label = elem.get('label')
            if not label or label in _legend_seen:
                continue
            _legend_seen.add(label)
            legend_entries.append(label)
            legend_colors.append(self._elem_legend_color(elem))
            # Store line style info (for line_style legends)
            legend_styles.append(elem.get('linestyle', 'solid'))
            legend_widths.append(elem.get('linewidth', 4))
        eff_xlabel, eff_ylabel, eff_title = self._effective_text()
        has_heading = bool(eff_title or self.subtitle)
        legend_loc = 'bottomright' if has_heading else 'topright'  # heading-aware default
        margin = 80
        _lfs = self.font_scale  # local alias for legend scaling
        legend_row_h  = int(30 * _lfs)
        legend_pad_y  = int(20 * _lfs)
        legend_swatch = int(24 * _lfs)
        legend_swatch_x = int(30 * _lfs)   # swatch x offset from legend left edge
        legend_label_x  = int(70 * _lfs)   # label  x offset from legend left edge
        legend_width  = int(300 * _lfs)
        legend_height = int((40 + 30 * len(legend_entries)) * _lfs)
        # Reserve a center-top region for title/subtitle so legend placement avoids it.
        title_zone_top = -self.height / 2 + 30
        title_zone_bottom = -self.height / 2 + 80 + (52 * self.font_scale if (eff_title and self.subtitle) else 0) + (45 * self.font_scale)
        title_zone_left = -self.width * 0.30
        title_zone_right = self.width * 0.30
        # Heuristic: choose legend position that maximizes minimum distance from data and text zones.
        data_px, data_py = [], []
        for elem in self.elements:
            if elem["type"] in ["line", "scatter", "line_evolving", "scatter_evolving"]:
                px, py = elem["x"], elem["y"]
            elif elem["type"] in ["histogram", "bar_graph", "bar_stacked", "bar_evolving"]:
                px = elem["bin_centers"]
                py = elem["heights"]
            elif elem["type"] in ("barh", "barh_evolving"):
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
        if self.xticks or eff_xlabel:
            text_zones.append((-self.width / 2, self.width / 2, self.height / 2 - 45, self.height / 2 + 120))
        # Left y-label/y-tick text band.
        if self.yticks or eff_ylabel:
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

        # --- Apply legend_pos override, if supplied ---
        _REGION_BOUNDS = {
            "top_left":     (0.0, 0.4, 0.0, 0.4),
            "top":          (0.3, 0.7, 0.0, 0.4),
            "top_right":    (0.6, 1.0, 0.0, 0.4),
            "left":         (0.0, 0.4, 0.3, 0.7),
            "center":       (0.3, 0.7, 0.3, 0.7),
            "right":        (0.6, 1.0, 0.3, 0.7),
            "bottom_left":  (0.0, 0.4, 0.6, 1.0),
            "bottom":       (0.3, 0.7, 0.6, 1.0),
            "bottom_right": (0.6, 1.0, 0.6, 1.0),
        }
        _pos = self.legend_pos
        _exact_placement = None  # set to (gx, gy) when an exact position is requested
        _region_anchor = None    # set to (ax, ay) when a named region is requested

        if isinstance(_pos, (tuple, list)) and len(_pos) == 2:
            # Explicit data coordinates → convert to graph-local and skip optimizer.
            gx, gy = self._data_to_shape(
                float(_pos[0]), float(_pos[1]),
                xmin_pad, xmax_pad, ymin_pad, ymax_pad,
            )
            # Clamp so the legend box stays inside the graph area.
            gx = max(x_min, min(x_max, gx))
            gy = max(y_min, min(y_max, gy))
            _exact_placement = (gx, gy)
            candidate_positions = []  # skip optimizer loop below
        elif isinstance(_pos, str):
            # Named region → filter candidates, then auto-place within the region.
            _norm = _pos.strip().lower().replace(" ", "_")
            if _norm in _REGION_BOUNDS and x_min < x_max and y_min < y_max:
                xf0, xf1, yf0, yf1 = _REGION_BOUNDS[_norm]
                xr = x_max - x_min
                yr = y_max - y_min
                x_lo = x_min + xf0 * xr
                x_hi = x_min + xf1 * xr
                y_lo = y_min + yf0 * yr
                y_hi = y_min + yf1 * yr
                filtered = [(lx, ly) for lx, ly in candidate_positions
                            if x_lo <= lx <= x_hi and y_lo <= ly <= y_hi]
                # Fall back to the midpoint of the requested region if no grid
                # candidates land there (very small graph, large legend, etc.).
                candidate_positions = filtered or [(0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi))]
                # Natural anchor for the named region: the corner/edge the name
                # implies (e.g. "top_left" -> true top-left). The optimizer is
                # biased toward this so a named region lands where the caller
                # expects instead of drifting to the least-crowded center.
                if "left" in _norm:
                    _anchor_x = x_lo
                elif "right" in _norm:
                    _anchor_x = x_hi
                else:
                    _anchor_x = 0.5 * (x_lo + x_hi)
                if "top" in _norm:
                    _anchor_y = y_lo
                elif "bottom" in _norm:
                    _anchor_y = y_hi
                else:
                    _anchor_y = 0.5 * (y_lo + y_hi)
                _region_anchor = (_anchor_x, _anchor_y)

        # Slight bias to top-right when heading exists and score ties.
        preferred_x = self.width / 2 - margin - legend_width / 2
        preferred_y = -self.height / 2 + margin + legend_height / 2
        best_score = float("-inf")
        legend_x = _exact_placement[0] if _exact_placement else preferred_x
        legend_y = _exact_placement[1] if _exact_placement else preferred_y

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

            clearance = min(min_data_dist, min_text_dist)
            if _region_anchor is not None:
                # Named region: stay near the requested corner/edge. Anchor
                # proximity dominates; clearance only nudges to break ties so
                # the legend doesn't drift to the center of the region.
                ax, ay = _region_anchor
                score = -(abs(lx - ax) + abs(ly - ay)) + 0.05 * clearance
            else:
                score = clearance
                # Small tie-breaker toward top-right for titled charts.
                if has_heading:
                    score += 0.001 * (-(abs(lx - preferred_x) + abs(ly - preferred_y)))

            if score > best_score:
                best_score = score
                legend_x, legend_y = lx, ly
        if legend_entries:
            # legendNull: a single anchor for the whole legend, parented to
            # PlotAnchor. Every swatch/line/label below is parented to this
            # null instead of PlotAnchor, so the legend can be moved or
            # animated as a unit (mirrors the cmapNull pattern).
            legend_null_cx = center_x + legend_x
            legend_null_cy = center_y + legend_y
            script.append("var legendNull = comp.layers.addNull();\n")
            script.append("legendNull.name = 'legendNull';\n")
            script.append(
                f"legendNull.property('Transform').property('Position').setValue([{legend_null_cx}, {legend_null_cy}]);\n"
            )
            script.append("legendNull.parent = PlotAnchor;\n")
            script.append(f"var legendGroup = [];\n")
            for i, (label, color) in enumerate(zip(legend_entries, legend_colors)):
                y_offset = legend_y + legend_pad_y + i * legend_row_h

                # Choose rendering mode based on legend_style
                if self.legend_style == 'line_style':
                    # Line-style legend: render actual line samples with styles
                    script.append(f"var legendLine{i} = comp.layers.addShape();\n")
                    script.append(f"legendLine{i}.property('Transform').property('Position').setValue([{center_x + legend_x - legend_width/2 + legend_swatch_x}, {center_y + y_offset}]);\n")
                    script.append(f"legendLine{i}.parent = legendNull;\n")
                    script.append(f"var legendLineContents{i} = legendLine{i}.property('ADBE Root Vectors Group');\n")
                    # Create a line path - length depends on line style
                    linestyle = legend_styles[i]
                    script.append(f"var legendLinePath{i} = legendLineContents{i}.addProperty('ADBE Vector Shape - Group');\n")
                    script.append(f"var legendLineShape{i} = legendLinePath{i}.property('ADBE Vector Shape');\n")
                    script.append(f"var shape{i} = new Shape();\n")
                    _lhalf = int(32 * _lfs)
                    # Solid line is slightly shorter to match dashed visual span (3 dashes + 2 gaps = 64 units).
                    if linestyle in ["solid", "-"]:
                        script.append(f"shape{i}.vertices = [[{-_lhalf}, 0], [{_lhalf}, 0]];\n")  # scaled for solid
                    else:
                        script.append(f"shape{i}.vertices = [[{-_lhalf}, 0], [{_lhalf}, 0]];\n")  # scaled for dashed/dotted
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

                    # Apply line style (dashes) - optimized for 2 complete dashes, scaled with font
                    if linestyle in ["dashed", "--"]:
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({int(16 * _lfs)});\n")
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({int(8 * _lfs)});\n")
                    elif linestyle in ["dotted", ":"]:
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({int(3 * _lfs)});\n")
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({int(5 * _lfs)});\n")
                    elif linestyle in ["dashdot", "-."]:
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 1').setValue({int(12 * _lfs)});\n")
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 1').setValue({int(4 * _lfs)});\n")
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Dash 2').setValue({int(3 * _lfs)});\n")
                        script.append(f"legendLineStroke{i}.property('Dashes').addProperty('ADBE Vector Stroke Gap 2').setValue({int(4 * _lfs)});\n")
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
                    script.append(f"legendSwatch{i}.property('Transform').property('Position').setValue([{center_x + legend_x - legend_width/2 + legend_swatch_x}, {center_y + y_offset}]);\n")
                    script.append(f"legendSwatch{i}.parent = legendNull;\n")
                    script.append(f"var legendSwatchContents{i} = legendSwatch{i}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var legendSwatchRect{i} = legendSwatchContents{i}.addProperty('ADBE Vector Shape - Rect');\n")
                    script.append(f"legendSwatchRect{i}.property('ADBE Vector Rect Size').setValue([{legend_swatch}, {legend_swatch}]);\n")
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
                script.append(f"legendLabelDoc{i}.font = \"{self.font_legend}\";\n")
                script.append(f"legendLabelDoc{i}.fillColor = {color_to_js(self.ui_color)};\n")
                script.append(f"legendLabelDoc{i}.justification = ParagraphJustification.LEFT_JUSTIFY;\n")
                script.append(f"legendLabelProp{i}.setValue(legendLabelDoc{i});\n")
                script.append(f"var legendLabelSR{i} = legendLabel{i}.sourceRectAtTime(0, false);\n")
                script.append(f"var legendLabelAP{i} = legendLabel{i}.property('Transform').property('Anchor Point');\n")
                script.append(f"legendLabelAP{i}.setValue([0, legendLabelSR{i}.top + legendLabelSR{i}.height/2]);\n")
                script.append(f"legendLabel{i}.property('Transform').property('Position').setValue([{center_x + legend_x - legend_width/2 + legend_label_x}, {center_y + y_offset + 2}]);\n")
                script.append(f"legendLabel{i}.parent = legendNull;\n")
                script.append(self._generate_text_slide_in_jsx(f"legendLabel{i}", f"LegendLabel{i}", ANIM_DURATION))
                # Fade-in animation for label
                script.append(f"legendLabel{i}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION}, 100);\n")
                # Apply easy ease to legend label opacity keyframes
                if self.drop_shadow:
                    script.append(self._generate_drop_shadow_jsx(f"legendLabel{i}", f"LegendLabel{i}"))



        # Title and subtitle. These are stacked just *above* the top edge of the
        # plot frame so they sit outside the graph instead of overlapping it.
        # Position is set in comp space *before* parenting to PlotAnchor (same
        # pattern as axis labels). Parenting first and then writing [center_x, y]
        # would treat center_x as a local offset and shove titles off to the side,
        # which is especially broken in multi-panel figures.
        frame_top_y = center_y - self.height / 2
        title_gap_above = int(22 * self.font_scale)   # block-to-frame gap
        title_sub_gap = int(6 * self.font_scale)      # title-to-subtitle gap
        if eff_title:
            escaped_title = eff_title.replace('"', '\\"').replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
            script.append(f"var titleLayer = comp.layers.addText(\"{escaped_title}\");\n")
            script.append("var titleProp = titleLayer.property('Source Text');\n")
            script.append("var titleDoc = titleProp.value;\n")
            script.append(f"titleDoc.fontSize = {int(48 * self.font_scale)};\n")
            script.append(f"titleDoc.font = \"{self.font_title}\";\n")
            script.append(f"titleDoc.fillColor = {color_to_js(self.label_color or self.ui_color)};\n")
            script.append("titleDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("titleProp.setValue(titleDoc);\n")
            # Anchor the title at top-center so comp-space Position pins its top edge.
            script.append("var titleSR = titleLayer.sourceRectAtTime(0, false);\n")
            script.append("var titleAP = titleLayer.property('Transform').property('Anchor Point');\n")
            script.append("titleAP.setValue([titleSR.left + titleSR.width/2, titleSR.top]);\n")
            script.append(self._generate_text_slide_in_jsx("titleLayer", "Title", ANIM_DURATION))

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
            subtitle_color = self.subtitle_color if self.subtitle_color is not None else (self.label_color or self.ui_color)
            script.append(f"var subtitleLayer = comp.layers.addText(\"{escaped_subtitle}\");\n")
            script.append("var subtitleProp = subtitleLayer.property('Source Text');\n")
            script.append("var subtitleDoc = subtitleProp.value;\n")
            script.append(f"subtitleDoc.fontSize = {int(28 * self.font_scale)};\n")
            script.append(f"subtitleDoc.font = \"{self.font_subtitle}\";\n")
            script.append(f"subtitleDoc.fillColor = {color_to_js(subtitle_color)};\n")
            script.append("subtitleDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("subtitleProp.setValue(subtitleDoc);\n")
            script.append("var subtitleSR = subtitleLayer.sourceRectAtTime(0, false);\n")
            script.append("var subtitleAP = subtitleLayer.property('Transform').property('Anchor Point');\n")
            script.append("subtitleAP.setValue([subtitleSR.left + subtitleSR.width/2, subtitleSR.top]);\n")
            script.append(self._generate_text_slide_in_jsx("subtitleLayer", "Subtitle", ANIM_DURATION))

            if self.animate_opacity:
                script.append(f"subtitleLayer.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                script.append(f"subtitleLayer.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 0.95}, 100);\n")
                if self.easy_ease:
                    script.append(f"applyEasyEase(subtitleLayer.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
            else:
                script.append(f"subtitleLayer.property('Transform').property('Opacity').setValue(100);\n")

            if self.drop_shadow:
                script.append(self._generate_drop_shadow_jsx("subtitleLayer", "Subtitle"))

        # Finalize title/subtitle stacking just above the plot frame, then parent.
        if eff_title and self.subtitle:
            script.append(
                f"subtitleLayer.property('Transform').property('Position').setValue("
                f"[{center_x}, {frame_top_y} - {title_gap_above} - subtitleSR.height]);\n"
            )
            script.append(f"subtitleLayer.parent = PlotAnchor;\n")
            script.append(
                f"titleLayer.property('Transform').property('Position').setValue("
                f"[{center_x}, {frame_top_y} - {title_gap_above} - subtitleSR.height - {title_sub_gap} - titleSR.height]);\n"
            )
            script.append(f"titleLayer.parent = PlotAnchor;\n")
        elif eff_title:
            script.append(
                f"titleLayer.property('Transform').property('Position').setValue("
                f"[{center_x}, {frame_top_y} - {title_gap_above} - titleSR.height]);\n"
            )
            script.append(f"titleLayer.parent = PlotAnchor;\n")
        elif self.subtitle:
            script.append(
                f"subtitleLayer.property('Transform').property('Position').setValue("
                f"[{center_x}, {frame_top_y} - {title_gap_above} - subtitleSR.height]);\n"
            )
            script.append(f"subtitleLayer.parent = PlotAnchor;\n")

        # Axis labels (relative to graph area)
        if eff_xlabel:
            escaped_xlabel = eff_xlabel.replace('"', '\\"').replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
            x_axis_y = self._resolve_xaxis_y(ymin_pad, ymax_pad, has_barh)
            xlabel_shape_y, xlabel_anchor_top = self._compute_xlabel_shape_y(
                xmin_pad, xmax_pad, ymin_pad, ymax_pad, x_axis_y
            )
            script.append(f"var xlabelLayer = comp.layers.addText(\"{escaped_xlabel}\");\n")
            script.append(
                f"xlabelLayer.property('Transform').property('Position').setValue([{center_x}, {center_y + xlabel_shape_y}]);\n"
            )
            script.append(f"xlabelLayer.parent = PlotAnchor;\n")
            script.append("var xlabelProp = xlabelLayer.property('Source Text');\n")
            script.append("var xlabelDoc = xlabelProp.value;\n")
            script.append(f"xlabelDoc.fontSize = {int(41 * self.font_scale)};\n")
            script.append(f"xlabelDoc.font = \"{self.font_label}\";\n")
            script.append(f"xlabelDoc.fillColor = {color_to_js(self.label_color or self.ui_color)};\n")
            script.append("xlabelDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("xlabelProp.setValue(xlabelDoc);\n")
            script.append("var xlabelSR = xlabelLayer.sourceRectAtTime(0, false);\n")
            script.append("var xlabelAP = xlabelLayer.property('Transform').property('Anchor Point');\n")
            if xlabel_anchor_top:
                script.append("xlabelAP.setValue([xlabelSR.left + xlabelSR.width/2, xlabelSR.top]);\n")
            else:
                script.append(
                    "xlabelAP.setValue([xlabelSR.left + xlabelSR.width/2, xlabelSR.top + xlabelSR.height]);\n"
                )
            script.append(self._generate_text_slide_in_jsx("xlabelLayer", "XLabel", ANIM_DURATION))

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
        if eff_ylabel:
            escaped_ylabel = eff_ylabel.replace('"', '\\"').replace('→', '->').replace('←', '<-').replace('↑', '^').replace('↓', 'v')
            y_axis_x = self._resolve_yaxis_x(xmin_pad, xmax_pad)
            ylabel_shape_x = self._compute_ylabel_shape_x(
                xmin_pad, xmax_pad, ymin_pad, ymax_pad, y_axis_x
            )
            script.append(f"var ylabelLayer = comp.layers.addText(\"{escaped_ylabel}\");\n")
            script.append(
                f"ylabelLayer.property('Transform').property('Position').setValue([{center_x + ylabel_shape_x}, {center_y}]);\n"
            )
            script.append(f"ylabelLayer.parent = PlotAnchor;\n")
            script.append("ylabelLayer.property('Transform').property('Rotation').setValue(-90);\n")
            script.append("var ylabelProp = ylabelLayer.property('Source Text');\n")
            script.append("var ylabelDoc = ylabelProp.value;\n")
            script.append(f"ylabelDoc.fontSize = {int(41 * self.font_scale)};\n")
            script.append(f"ylabelDoc.font = \"{self.font_label}\";\n")
            script.append(f"ylabelDoc.fillColor = {color_to_js(self.label_color or self.ui_color)};\n")
            script.append("ylabelDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("ylabelProp.setValue(ylabelDoc);\n")
            script.append(self._generate_text_slide_in_jsx("ylabelLayer", "YLabel", ANIM_DURATION))

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
                delay = elem.get("delay", 0)
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
                if self._view_animated:
                    script.append(f"var annotationPos{annotation_count} = annotationLayer{annotation_count}.property('Transform').property('Position');\n")
                    self._emit_view_pos_kf(
                        script, f"annotationPos{annotation_count}",
                        lambda xmn, xmx, ymn, ymx, _x=elem["x"], _y=elem["y"]: self._data_to_shape(_x, _y, xmn, xmx, ymn, ymx),
                    )

                # Set text properties
                script.append(f"var annotationProp{annotation_count} = annotationLayer{annotation_count}.property('Source Text');\n")
                script.append(f"var annotationDoc{annotation_count} = annotationProp{annotation_count}.value;\n")
                script.append(f"annotationDoc{annotation_count}.fontSize = {int(elem['fontsize'] * self.font_scale)};\n")
                script.append(f"annotationDoc{annotation_count}.font = \"{elem.get('font') or self.font_body}\";\n")
                script.append(f"annotationDoc{annotation_count}.fillColor = {color_to_js(self.ui_color)};\n")
                script.append(f"annotationDoc{annotation_count}.justification = {justification};\n")
                script.append(f"annotationProp{annotation_count}.setValue(annotationDoc{annotation_count});\n")
                # Set anchor point: horizontal from justification, vertical from vertical_alignment.
                script.append(f"var annotationSR{annotation_count} = annotationLayer{annotation_count}.sourceRectAtTime(0, false);\n")
                script.append(f"var annotationAP{annotation_count} = annotationLayer{annotation_count}.property('Transform').property('Anchor Point');\n")
                sr = f"annotationSR{annotation_count}"
                if justification == "ParagraphJustification.LEFT_JUSTIFY":
                    anchor_x = f"{sr}.left"
                elif justification == "ParagraphJustification.CENTER_JUSTIFY":
                    anchor_x = f"{sr}.left + {sr}.width/2"
                else:  # RIGHT_JUSTIFY
                    anchor_x = f"{sr}.left + {sr}.width"
                valign = elem.get("vertical_alignment", "center")
                if valign == "top":
                    anchor_y = f"{sr}.top"
                elif valign == "bottom":
                    anchor_y = f"{sr}.top + {sr}.height"
                else:  # center
                    anchor_y = f"{sr}.top + {sr}.height/2"
                script.append(f"annotationAP{annotation_count}.setValue([{anchor_x}, {anchor_y}]);\n")

                script.append(self._generate_text_slide_in_jsx(f"annotationLayer{annotation_count}", f"Annotation{annotation_count}", ANIM_DURATION, 1, delay))

                # Fade-in animation for annotation
                script.append(f"annotationLayer{annotation_count}.property('Transform').property('Opacity').setValueAtTime({delay}, 0);\n")
                script.append(f"annotationLayer{annotation_count}.property('Transform').property('Opacity').setValueAtTime({delay + ANIM_DURATION * 0.9}, 100);\n")

                # Apply easy ease to annotation opacity keyframes
                if self.easy_ease:
                    script.append(f"applyEasyEase(annotationLayer{annotation_count}.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")

                # Add drop shadow if globally enabled
                if self.drop_shadow:
                    script.append(self._generate_drop_shadow_jsx(f"annotationLayer{annotation_count}", f"Annotation{annotation_count}"))

                annotation_count += 1

        # --- COLORBAR (cmap) for scatter gradients ---
        self._generate_cmap_jsx(script, center_x, center_y, ANIM_DURATION)

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

        # --- FILM STYLE: front-heavy push-in zoom -------------------------------
        # Always keyframed on this graph's own PlotAnchor (cheap, local, no new
        # layer needed) so every panel of a figure zooms in sync even though the
        # comp-wide paper/light-leak/adjustment stack below is only emitted once.
        if self.film_style:
            script.extend(self._film_style_zoom_keyframes_jsx("PlotAnchor"))

        # --- FILM STYLE: paper/light-leak/global adjustment stack (generated
        # last, so it sits on top of everything, including the cinematic/wiggle
        # adjustment layers above). Skipped for figure panels (draw_film_style
        # False); AEFigure emits these once, comp-wide, instead. ---
        if self.film_style and draw_film_style:
            script.extend(self._film_style_edge_blur_adjustment_jsx(center_x, center_y))
            script.extend(self._film_style_vignette_adjustment_jsx(center_x, center_y))
            script.extend(self._film_style_temporal_adjustment_jsx(center_x, center_y))
            script.extend(self._film_style_light_leak_jsx(center_x, center_y))
            script.append(self._film_style_element_pass_jsx())

        body = "".join(script)
        if wrap_iife:
            body = "(function(){\n" + body + "\n})();\n"
        return "".join(preamble) + body

    def save(self, filename: str = "", folder_path=UNSET):
        """
        Write the JSX script to a file for After Effects.

        folder_path: Output directory. Defaults to
            ``aegraph_config.config.jsx_output_dir`` (a ``jsx_output`` folder in
            the aegraph package directory) when not given. The directory is
            created if it doesn't exist.
        """
        jsx = self._generate_jsx()
        _write_jsx_file(jsx, filename, folder_path)
        return self

    def render(self, ae_version=UNSET, folder_path=UNSET):
        """
        Write and run the JSX script in After Effects.
        Works on macOS using AppleScript and Windows using AfterFX.exe.

        ae_version: After Effects application name. Defaults to
            ``aegraph_config.config.ae_version`` (from ``.env.user``) when not
            given.
        folder_path: Output directory for the generated .jsx. Defaults to
            ``aegraph_config.config.jsx_output_dir`` when not given.
        """
        if folder_path is UNSET:
            folder_path = config.jsx_output_dir

        filename = f"AEGraph_{get_time()}.jsx"
        jsx = self._generate_jsx()
        abs_path = _write_jsx_file(jsx, filename, folder_path)
        _run_jsx_file(abs_path, ae_version)
        return self


class AEFigure:
    """A figure that tiles several AEGraph panels into one After Effects comp.

    Mirrors matplotlib's ``Figure`` / ``plt.subplots`` mental model. Each panel
    is a normal :class:`AEGraph` that draws into its own rectangular region of a
    single shared composition. Use :func:`subplots` for the common grid case::

        fig, axes = subplots(2, 2, comp_name="Dashboard", theme="Slate Report")
        axes[0][0].plot(x, y).set_title("A")
        axes[0][1].scatter(x, y)
        fig.suptitle("Quarterly Report").render()

    Panels are independent: each keeps its own data limits, scales, ticks,
    legend, and colorbar. Spacing is controlled by ``wspace`` / ``hspace``
    (fraction of a grid cell reserved around each panel, which also leaves room
    for that panel's axis labels and title).
    """

    def __init__(self, comp_name="AEFigure", comp_width=None, comp_height=None,
                 fps=60, theme=None, margin=0.06, wspace=0.22, hspace=0.30,
                 bg_color=UNSET, **panel_defaults):
        # Apply the figure theme into config up front (same as AEGraph) so the
        # shared comp background and any UNSET panel args inherit it.
        if theme is not None:
            from aegraph_config import apply_theme as _apply_theme
            _apply_theme(theme)
        self.comp_name = comp_name
        self.comp_width = int(comp_width) if comp_width else config.comp_width
        self.comp_height = int(comp_height) if comp_height else config.comp_height
        self.fps = fps if fps is not None else 60
        self.theme = theme
        self.margin = float(margin)
        self.wspace = float(wspace)
        self.hspace = float(hspace)
        # One shared background for the whole figure. UNSET means "inherit the
        # panels' resolved background color". Pass "none" to omit it.
        self.bg_color = bg_color
        self.panel_defaults = panel_defaults
        self._suptitle = None
        self._suptitle_kwargs = {}
        self._panels = []  # list of dicts: {graph, ns, spec, _origin}
        self._grid = None  # (nrows, ncols) when created via subplots()
        self._pending_film_style_overrides = {}  # set via film_style_parameters()

    # --- layout geometry ---------------------------------------------------

    def _grid_geometry(self, nrows, ncols):
        """Return (left, top, cell_w, cell_h) in comp pixels for the grid."""
        m = self.margin
        top_extra = 0.08 if self._suptitle else 0.0
        left = self.comp_width * m
        right = self.comp_width * (1 - m)
        top = self.comp_height * (m + top_extra)
        bottom = self.comp_height * (1 - m)
        cell_w = (right - left) / ncols
        cell_h = (bottom - top) / nrows
        return left, top, cell_w, cell_h

    def _cell_rect(self, r, c, rowspan, colspan, nrows, ncols):
        """Return (center_x, center_y, panel_w, panel_h) for a (spanned) cell."""
        left, top, cell_w, cell_h = self._grid_geometry(nrows, ncols)
        box_w = colspan * cell_w
        box_h = rowspan * cell_h
        cx = left + c * cell_w + box_w / 2.0
        cy = top + r * cell_h + box_h / 2.0
        panel_w = box_w * (1 - self.wspace)
        panel_h = box_h * (1 - self.hspace)
        return cx, cy, panel_w, panel_h

    # --- panel construction ------------------------------------------------

    def _make_panel(self, pw, ph, spec, **kwargs):
        opts = dict(self.panel_defaults)
        opts.update(kwargs)
        if self.theme is not None and not opts.get("theme"):
            opts["theme"] = self.theme
        g = AEGraph(comp_name=self.comp_name,
                    comp_width=self.comp_width, comp_height=self.comp_height,
                    width=max(1, int(round(pw))), height=max(1, int(round(ph))),
                    fps=self.fps, **opts)
        if self._pending_film_style_overrides:
            g.film_style_parameters(**self._pending_film_style_overrides)
        ns = f"_p{len(self._panels)}"
        self._panels.append({"graph": g, "ns": ns, "spec": spec, "_origin": None})
        return g

    def film_style_parameters(self, **kwargs):
        """Override film-style knobs for every panel in this figure (existing
        panels are updated immediately; panels added afterward via
        ``add_subplot``/``subplots`` inherit them too). See
        ``AEGraph.film_style_parameters`` for the accepted keys. Returns
        ``self`` for chaining."""
        self._pending_film_style_overrides.update(kwargs)
        for item in self._panels:
            item["graph"].film_style_parameters(**kwargs)
        return self

    def add_subplot(self, row, col, rowspan=1, colspan=1,
                    nrows=None, ncols=None, **kwargs):
        """Add a panel occupying ``rowspan x colspan`` cells at ``(row, col)``.

        ``nrows`` / ``ncols`` default to the grid established by
        :meth:`subplots`; pass them explicitly to build an ad-hoc grid.
        """
        if nrows is None or ncols is None:
            if self._grid is None:
                raise ValueError(
                    "Provide nrows/ncols, or build the figure with subplots() first."
                )
            nrows, ncols = self._grid
        else:
            self._grid = (nrows, ncols)
        _, _, pw, ph = self._cell_rect(row, col, rowspan, colspan, nrows, ncols)
        spec = {"kind": "grid", "r": row, "c": col, "rowspan": rowspan,
                "colspan": colspan, "nrows": nrows, "ncols": ncols}
        return self._make_panel(pw, ph, spec, **kwargs)

    def subplots(self, nrows, ncols):
        """Create a full ``nrows x ncols`` grid; return a 2D list of panels."""
        self._grid = (nrows, ncols)
        axes = []
        for r in range(nrows):
            row_axes = [self.add_subplot(r, c, nrows=nrows, ncols=ncols)
                        for c in range(ncols)]
            axes.append(row_axes)
        return axes

    def inset(self, parent, x, y, w, h, **kwargs):
        """Add a picture-in-picture panel inside ``parent``.

        ``x, y, w, h`` are fractions (0..1) of the parent panel's plotting box,
        measured from its top-left corner.
        """
        spec = {"kind": "inset", "parent": parent,
                "x": float(x), "y": float(y), "w": float(w), "h": float(h)}
        # Initial size estimate; resolved exactly at render time.
        return self._make_panel(parent.width * w, parent.height * h, spec, **kwargs)

    def suptitle(self, text, fontsize=None, color=None):
        """Set a figure-level title centered above the panels."""
        self._suptitle = text
        self._suptitle_kwargs = {"fontsize": fontsize, "color": color}
        return self

    def facet(self, data, by, x=None, y=None, kind="line", ncols=None,
              sharex=True, sharey=True, titles=True, **plot_kwargs):
        """Small-multiples: one panel per group of ``data[by]``.

        Args:
            data: A pandas DataFrame.
            by: Column name to split the data into facets.
            x, y: Column names to plot in each facet.
            kind: ``"line"``, ``"scatter"``, ``"bar"``, or ``"area"``.
            ncols: Columns in the facet grid (defaults to ~sqrt of group count).
            sharex / sharey: Give every facet the same x / y limits so the small
                multiples are directly comparable.
            titles: Set each panel's title to its group value.

        Returns the list of created panels.
        """
        if pd is None or not isinstance(data, pd.DataFrame):
            raise ValueError("facet() requires a pandas DataFrame")
        if x is None or y is None:
            raise ValueError("facet() requires x and y column names")
        groups = list(pd.unique(data[by]))
        n = len(groups)
        if n == 0:
            return []
        if ncols is None:
            ncols = int(math.ceil(math.sqrt(n)))
        nrows = int(math.ceil(n / ncols))
        self._grid = (nrows, ncols)

        # Shared limits computed once across the whole dataset.
        xlim = (float(data[x].min()), float(data[x].max())) if sharex else None
        ylim = (float(data[y].min()), float(data[y].max())) if sharey else None

        panels = []
        for idx, g in enumerate(groups):
            r, c = divmod(idx, ncols)
            panel = self.add_subplot(r, c, nrows=nrows, ncols=ncols)
            sub = data[data[by] == g]
            gx = sub[x].to_numpy()
            gy = sub[y].to_numpy()
            if kind == "scatter":
                panel.scatter(gx, gy, **plot_kwargs)
            elif kind == "bar":
                panel.bar_graph(gx, gy, **plot_kwargs)
            elif kind == "area":
                panel.area(gx, gy, **plot_kwargs)
            else:
                panel.plot(gx, gy, **plot_kwargs)
            if titles:
                panel.set_title(str(g))
            if xlim is not None:
                panel.set_xlim(*xlim)
            if ylim is not None:
                panel.set_ylim(*ylim)
            panels.append(panel)
        return panels

    # --- rendering ---------------------------------------------------------

    def _resolve_layout(self):
        resolved = {}
        for item in self._panels:
            spec = item["spec"]
            if spec["kind"] == "grid":
                cx, cy, pw, ph = self._cell_rect(
                    spec["r"], spec["c"], spec["rowspan"], spec["colspan"],
                    spec["nrows"], spec["ncols"])
                item["_origin"] = (cx, cy)
                item["graph"].width = max(1, int(round(pw)))
                item["graph"].height = max(1, int(round(ph)))
                resolved[id(item["graph"])] = (cx, cy, pw, ph)
        for item in self._panels:
            spec = item["spec"]
            if spec["kind"] == "inset":
                pcx, pcy, ppw, pph = resolved.get(
                    id(spec["parent"]),
                    (self.comp_width / 2, self.comp_height / 2,
                     spec["parent"].width, spec["parent"].height))
                x, y, w, h = spec["x"], spec["y"], spec["w"], spec["h"]
                ix = pcx - ppw / 2 + x * ppw + (w * ppw) / 2
                iy = pcy - pph / 2 + y * pph + (h * pph) / 2
                item["_origin"] = (ix, iy)
                item["graph"].width = max(1, int(round(w * ppw)))
                item["graph"].height = max(1, int(round(h * pph)))
                resolved[id(item["graph"])] = (ix, iy, w * ppw, h * pph)

    def _suptitle_jsx(self):
        if not self._suptitle:
            return ""
        text = str(self._suptitle).replace("\\", "\\\\").replace('"', '\\"')
        fs = self._suptitle_kwargs.get("fontsize") or 64
        color = self._suptitle_kwargs.get("color")
        if color is None and self._panels:
            color = getattr(self._panels[0]["graph"], "ui_color", None)
        if color is None:
            color = [0.9, 0.9, 0.9]
        color_js = color_to_js(color)
        # Font: explicit suptitle kwarg wins; else inherit the first panel's
        # title font; else fall back to the config default.
        font = self._suptitle_kwargs.get("font")
        if font is None and self._panels:
            font = getattr(self._panels[0]["graph"], "font_title", None)
        if font is None:
            font = config.font_title
        x = self.comp_width / 2
        y = self.comp_height * (self.margin + 0.03)
        s = [
            f'var figTitle = comp.layers.addText("{text}");\n',
            "figTitle.name = 'FigureTitle';\n",
            f"figTitle.property('Transform').property('Position').setValue([{x}, {y}]);\n",
            "var figTitleProp = figTitle.property('Source Text');\n",
            "var figTitleDoc = figTitleProp.value;\n",
            f"figTitleDoc.fontSize = {int(fs)};\n",
            f'figTitleDoc.font = "{font}";\n',
            f"figTitleDoc.fillColor = {color_js};\n",
            "figTitleDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n",
            "figTitleProp.setValue(figTitleDoc);\n",
        ]
        return "".join(s)

    def _resolve_bg_color(self):
        """Resolve the single figure background color.

        Honors an explicit ``bg_color`` on the figure. Otherwise uses the
        figure's ``theme`` (when set), then the first panel's background, then
        ``config.bg_color``. Returns None when no background should be drawn.
        """
        bg = self.bg_color
        if bg is not UNSET and bg is not None:
            if isinstance(bg, str) and bg.lower() == "none":
                return None
            return bg
        if self.theme is not None:
            from aegraph_config import get_theme
            bg = get_theme(self.theme)["bg"]
        elif self._panels:
            bg = getattr(self._panels[0]["graph"], "bg_color", None)
        else:
            bg = config.bg_color
        if bg is None:
            return None
        if isinstance(bg, str) and bg.lower() == "none":
            return None
        return bg

    def _figure_bg_jsx(self):
        """Emit one comp-wide background rectangle for the whole figure."""
        bg = self._resolve_bg_color()
        if bg is None:
            return ""
        bg_js = color_to_js(bg)
        # Inherit stroke styling from the first panel when available.
        stroke_color = None
        stroke_width = 0
        if self._panels:
            g0 = self._panels[0]["graph"]
            stroke_color = getattr(g0, "bg_stroke_color", None)
            stroke_width = getattr(g0, "bg_stroke_width", 0) or 0
        cx = self.comp_width / 2
        cy = self.comp_height / 2
        s = [
            # Standalone AEGraph used to set this; subplot panels skip per-panel
            # GraphBG and rely on one shared FigureBG instead, so the comp's own
            # background must track the active theme too (especially when the comp
            # is reused across re-renders).
            f"comp.bgColor = {bg_js};\n",
            "try { comp.layer('FigureBG').remove(); } catch (e) {}\n",
            "var figBG = comp.layers.addShape();\n",
            "figBG.name = 'FigureBG';\n",
            "var figBGc = figBG.property('ADBE Root Vectors Group');\n",
            "var figBGr = figBGc.addProperty('ADBE Vector Shape - Rect');\n",
            f"figBGr.property('ADBE Vector Rect Size').setValue([{self.comp_width}, {self.comp_height}]);\n",
            "figBGr.property('ADBE Vector Rect Position').setValue([0, 0]);\n",
            "var figBGf = figBGc.addProperty('ADBE Vector Graphic - Fill');\n",
            f"figBGf.property('ADBE Vector Fill Color').setValue({bg_js});\n",
        ]
        if stroke_color is not None and stroke_width:
            s.append("var figBGs = figBGc.addProperty('ADBE Vector Graphic - Stroke');\n")
            s.append(f"figBGs.property('ADBE Vector Stroke Color').setValue({color_to_js(stroke_color)});\n")
            s.append(f"figBGs.property('ADBE Vector Stroke Width').setValue({stroke_width});\n")
        s.append(f"figBG.property('Transform').property('Position').setValue([{cx}, {cy}]);\n")
        return "".join(s)

    def _figure_distress_jsx(self):
        """Emit one comp-wide distress texture for the whole figure.

        Reuses the first panel's distress setting (and AEGraph's texture
        emitter), positions it at the comp center, and stacks it just above
        the shared FigureBG -- so the grunge appears once instead of once per
        panel. Returns "" when no panel requests a texture.
        """
        if not self._panels:
            return ""
        g0 = self._panels[0]["graph"]
        if not getattr(g0, "distress_texture", None):
            return ""
        move_line = (
            "    var __figBG = null;\n"
            "    try { __figBG = comp.layer('FigureBG'); } catch (e) { __figBG = null; }\n"
            "    if (__figBG) { distressLayer.moveBefore(__figBG); }\n"
        )
        cx = self.comp_width / 2
        cy = self.comp_height / 2
        return "".join(g0._distress_block_jsx(cx, cy, move_line, parent_var=None))

    def _figure_film_style_jsx(self):
        """Emit the comp-wide pieces of the film style once for the whole
        figure: paper background and light-leak overlay (stacked the same
        way the shared distress texture is), plus the three global
        adjustment layers. None of these are parented to a single panel, so
        each gets its own push-in zoom keyframes directly (``apply_zoom``)
        instead of inheriting a panel's PlotAnchor zoom. Each panel still
        zooms its own content independently (see ``_generate_jsx``), using
        the same config values so everything settles in sync. Returns ""
        when no panel enables ``film_style``.
        """
        if not self._panels:
            return ""
        g0 = self._panels[0]["graph"]
        if not getattr(g0, "film_style", False):
            return ""
        cx = self.comp_width / 2
        cy = self.comp_height / 2
        paper_move_line = (
            "    var __figBG = null;\n"
            "    try { __figBG = comp.layer('FigureBG'); } catch (e) { __figBG = null; }\n"
            "    if (__figBG) { paperLayer.moveBefore(__figBG); }\n"
        )
        s = []
        s.extend(g0._film_style_paper_jsx(cx, cy, paper_move_line, parent_var=None, apply_zoom=True))
        s.extend(g0._film_style_edge_blur_adjustment_jsx(cx, cy))
        s.extend(g0._film_style_vignette_adjustment_jsx(cx, cy))
        s.extend(g0._film_style_temporal_adjustment_jsx(cx, cy))
        s.extend(g0._film_style_light_leak_jsx(cx, cy, parent_var=None, apply_zoom=True))
        return "".join(s)

    def _build_jsx(self):
        self._resolve_layout()
        uses_easy = any(getattr(it["graph"], "easy_ease", False) for it in self._panels)
        uses_film_style = any(getattr(it["graph"], "film_style", False) for it in self._panels)
        parts = []
        if uses_easy:
            parts.append(_JSX_EASY_EASE_FN)
        parts.append(_JSX_HELPER_FUNCTIONS)
        if uses_film_style:
            parts.append(_JSX_FILM_STYLE_FN)
        parts.append(_jsx_comp_header(self.comp_name, self.comp_width,
                                      self.comp_height, self.fps))
        # One shared background painted first so it sits behind every panel.
        parts.append(self._figure_bg_jsx())
        # One shared distress/grunge texture for the whole figure (above the
        # background), instead of one per panel.
        parts.append(self._figure_distress_jsx())
        # One shared paper texture, light leak, and adjustment stack for the
        # whole figure, instead of one per panel.
        parts.append(self._figure_film_style_jsx())
        parts.append(self._suptitle_jsx())
        for item in self._panels:
            parts.append(item["graph"]._generate_jsx(
                include_preamble=False, origin=item["_origin"],
                wrap_iife=True, ns=item["ns"], draw_bg=False,
                draw_distress=False, draw_film_style=False))
        # Per-element Roughen Edges + Gaussian Blur + Multiply pass, run once
        # over the whole comp after every panel (and the shared film-style
        # layers) have been drawn.
        if uses_film_style:
            parts.append(self._panels[0]["graph"]._film_style_element_pass_jsx())
        return "".join(parts)

    def save(self, filename: str = "", folder_path=UNSET):
        """Write the combined figure JSX to disk."""
        jsx = self._build_jsx()
        _write_jsx_file(jsx, filename, folder_path)
        return self

    def render(self, ae_version=UNSET, folder_path=UNSET):
        """Write and run the combined figure JSX in After Effects."""
        if folder_path is UNSET:
            folder_path = config.jsx_output_dir
        filename = f"AEGraph_{get_time()}.jsx"
        jsx = self._build_jsx()
        abs_path = _write_jsx_file(jsx, filename, folder_path)
        _run_jsx_file(abs_path, ae_version)
        return self


def subplots(nrows=1, ncols=1, comp_name="AEFigure", comp_width=None,
             comp_height=None, fps=60, theme=None, wspace=0.22, hspace=0.30,
             margin=0.06, suptitle=None, bg_color=UNSET, **panel_defaults):
    """Create an :class:`AEFigure` and a grid of panels (matplotlib-style).

    Returns ``(fig, axes)``. ``axes`` is a single panel when ``nrows == ncols
    == 1``, a 1D list when one dimension is 1, otherwise a 2D nested list.
    Extra keyword arguments are forwarded to every panel's :class:`AEGraph`.
    """
    fig = AEFigure(comp_name=comp_name, comp_width=comp_width,
                   comp_height=comp_height, fps=fps, theme=theme,
                   wspace=wspace, hspace=hspace, margin=margin,
                   bg_color=bg_color, **panel_defaults)
    if suptitle is not None:
        fig.suptitle(suptitle)
    axes = fig.subplots(nrows, ncols)
    if nrows == 1 and ncols == 1:
        return fig, axes[0][0]
    if nrows == 1:
        return fig, axes[0]
    if ncols == 1:
        return fig, [row[0] for row in axes]
    return fig, axes
