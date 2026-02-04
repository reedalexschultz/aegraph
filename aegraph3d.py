import tempfile
import subprocess
import os
import numpy as np

COLOR_NAMES = {
    "red": [1, 0, 0],
    "green": [0, 1, 0],
    "blue": [0, 0, 1],
    "white": [1, 1, 1],
    "black": [0, 0, 0],
    "cyan": [0, 1, 1],
    "magenta": [1, 0, 1],
    "yellow": [1, 1, 0],
    "gray": [0.5, 0.5, 0.5],
}

def color_to_js(color):
    return COLOR_NAMES[color.lower()] if isinstance(color, str) else list(color)

class AEGraph3D:
    def __init__(
        self,
        width=800,
        height=800,
        depth=800,
        comp_width=1920,
        comp_height=1080,
        comp_name="AEGraph3D",
        fps=24,
        duration=8,
        camera_distance=2500,
        easy_ease=True,
        ease_speed=0,
        ease_influence=33,
    ):
        self.width = width
        self.height = height
        self.depth = depth

        self.comp_width = comp_width
        self.comp_height = comp_height
        self.comp_name = comp_name
        self.fps = fps
        self.duration = duration
        self.camera_distance = camera_distance

        self.easy_ease = easy_ease
        self.ease_speed = ease_speed
        self.ease_influence = ease_influence

        self.elements = []

        self.xlim = None
        self.ylim = None
        self.zlim = None

    def scatter3d(self, x, y, z, color="red", radius=12,
              animate=1.0, delay=0.0, z_time_expr=False):
        self.elements.append({
            "z_time_expr": z_time_expr,
            "x": list(x),
            "y": list(y),
            "z": list(z),
            "color": color,
            "radius": radius,
            "animate": animate,
            "delay": delay
        })
        return self



    def _data_to_3d(self, x, y, z):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        xmin, xmax = self.xlim if self.xlim else (float(x.min()), float(x.max()))
        ymin, ymax = self.ylim if self.ylim else (float(y.min()), float(y.max()))
        zmin, zmax = self.zlim if self.zlim else (float(z.min()), float(z.max()))

        def m(v, a, b, s):
            return 0 if a == b else s * (v - a) / (b - a) - s / 2

        X = [m(v, xmin, xmax, self.width) for v in x.flatten()]
        Y = [-m(v, ymin, ymax, self.height) for v in y.flatten()]
        Z = [m(v, zmin, zmax, self.depth) for v in z.flatten()]

        return X, Y, Z


    def _generate_jsx(self):
        out = []
        add = out.append

        add("app.beginUndoGroup('AEGraph3D');\n")

        if self.easy_ease:
            add("""
function applyEasyEase(p,s,i){
    if(!p || p.numKeys < 2) return;
    var d = p.value.length || 1;
    var ei=[], eo=[];
    for(var k=0;k<d;k++){
        ei.push(new KeyframeEase(s,i));
        eo.push(new KeyframeEase(s,i));
    }
    for(var j=1;j<=p.numKeys;j++){
        p.setTemporalEaseAtKey(j, ei, eo);
    }
}
""")

        add(f"""
var comp = app.project.items.addComp(
    "{self.comp_name}",
    {self.comp_width},
    {self.comp_height},
    1,
    {self.duration},
    {self.fps}
);
comp.openInViewer();
""")

        add(f"""
var cam = comp.layers.addCamera("Camera", [{self.comp_width/2}, {self.comp_height/2}]);
cam.property("Transform").property("Position").setValue([
    {self.comp_width/2},
    {self.comp_height/2},
    -{self.camera_distance}
]);
""")

        add("""
var graph = comp.layers.addNull();
graph.name = "GraphOrigin";
graph.threeDLayer = true;
graph.property("Transform").property("Position").setValue([
    comp.width/2,
    comp.height/2,
    0
]);
""")

        W, H, D = self.width, self.height, self.depth

        add(f"""
function addAxis(name, verts, color, rx, ry){{
    var l = comp.layers.addShape();
    l.name = name;
    l.threeDLayer = true;
    l.parent = graph;

    l.property("Transform").property("Position").setValue([0,0,0]);
    l.property("Transform").property("X Rotation").setValue(rx);
    l.property("Transform").property("Y Rotation").setValue(ry);

    var g = l.property("ADBE Root Vectors Group");
    var s = g.addProperty("ADBE Vector Shape - Group");
    var p = s.property("ADBE Vector Shape");

    var sh = new Shape();
    sh.vertices = verts;
    sh.closed = false;
    p.setValue(sh);

    var st = g.addProperty("ADBE Vector Graphic - Stroke");
    st.property("ADBE Vector Stroke Color").setValue(color);
    st.property("ADBE Vector Stroke Width").setValue(4);
}}

// X axis
addAxis("X Axis", [[-{W/2},0],[{W/2},0]], [1,0,0], 0, 0);

// Y axis
addAxis("Y Axis", [[0,-{H/2}],[0,{H/2}]], [0,1,0], 0, 0);

// Z axis (VISIBLE)
addAxis("Z Axis", [[0,-{D/2}],[0,{D/2}]], [0,0,1], 90, 0);
""")

        for ei, e in enumerate(self.elements):
            X, Y, Z = self._data_to_3d(e["x"], e["y"], e["z"])
            col = color_to_js(e["color"])
            n = len(X)
            dt = e["animate"] / max(n, 1)

            for i, (x, y, z) in enumerate(zip(X, Y, Z)):
                t0 = e["delay"] + i * dt
                t1 = t0 + dt

                add(f"""
var p{ei}_{i} = comp.layers.addShape();
p{ei}_{i}.name = "Point_{ei}_{i}";
p{ei}_{i}.threeDLayer = true;
p{ei}_{i}.parent = graph;
// p{ei}_{i}.setAutoOrient(AutoOrientType.CAMERA_ORIENT);

var pos = p{ei}_{i}.property("Transform").property("Position");

// set initial X,Y and placeholder Z
pos.setValue([{x}, {y}, 0]);

// animate Z as a function of time (expression)
pos.expression =
"x = {x};\\n" +
"y = {y};\\n" +
"z = Math.sin((x/60) + time) * Math.cos((y/60) + time) * 20;\\n" +
"[x, y, z]";

var g = p{ei}_{i}.property("ADBE Root Vectors Group");
var el = g.addProperty("ADBE Vector Shape - Ellipse");
el.property("ADBE Vector Ellipse Size").setValue([{e["radius"]*2},{e["radius"]*2}]);

var f = g.addProperty("ADBE Vector Graphic - Fill");
f.property("ADBE Vector Fill Color").setValue([{col[0]},{col[1]},{col[2]}]);

var s = p{ei}_{i}.property("Transform").property("Scale");
s.setValueAtTime({t0}, [0,0,0]);
s.setValueAtTime({t1}, [100,100,100]);
""")
                if self.easy_ease:
                    add(f"applyEasyEase(s,{self.ease_speed},{self.ease_influence});\n")

        add("app.endUndoGroup();\n")
        return "".join(out)

    def render(self, ae_version="Adobe After Effects 2025"):
        jsx = self._generate_jsx()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsx") as f:
            f.write(jsx.encode("utf-8"))
            path = f.name

        subprocess.run([
            "osascript",
            "-e",
            f'''
tell application "{ae_version}"
    activate
    DoScriptFile "{path}"
end tell
'''
        ])

        os.unlink(path)
