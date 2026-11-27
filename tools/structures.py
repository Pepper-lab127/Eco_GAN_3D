"""3D models and dimensioned massing drawings of the two structures on the
1:500 site models: St. Andrews Lock and Dam (the Caméré curtain bridge-dam
carrying PTH 44, model A) and the Red River Floodway outlet control
structure (model B).

RECONSTRUCTIONS, not surveys. Published figures (Parks Canada / Historic
Places, Associated Engineering, PSPC, Manitoba floodway reports) give the
dam ~270 m overall and 240.1 m between abutments, 5 piers making 6 bays,
steel trusses ~40 m long, 15 m piers, a 6 m concrete sill and 89 curtains;
the lock 13.7 m wide and ~62 m long; the floodway outlet an uncontrolled
concrete crest and stilling basin with a 14 ft (4.3 m) drop. Positions,
lengths, crest and wall-top levels come from the 1 m lidar and Overture Maps
(the dam's weir line is 239 m long; the outlet crest 88 m at 223.3 m).
Everything else (truss depth and panelling, deck and sill levels, pier
shape, basin length, baffle blocks, wall tops) is an ESTIMATE, flagged in
the drawings.

Outputs (outputs/structures/):
  - <name>_detail_m.obj        detailed model, real metres, local axes
  - <name>_detail_1to500.obj   the same, in the site model's millimetres and
                               position (drops into lockport_1to500.obj)
  - <name>_massing_m.obj / _1to500.obj   the simplified wood massing
  - <name>_massing_drawings.svg/.pdf/.jpg  dimensioned drawings + parts list
  - <name>_detail.png / _massing.png       previews

    python -m tools.structures
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw

from drawing.svg import Svg

OUT = "outputs/structures"
INK = "#111111"
BASE_Z = 216.5            # elevation of the site models' base board top (3 mm card, 1.5 m layers from 218 m)


# --------------------------------------------------------------------------- mesh
class Mesh:
    def __init__(self):
        self.v, self.objs = [], []          # objs: (name, [faces])

    def _start(self, name):
        if not self.objs or self.objs[-1][0] != name:
            self.objs.append((name, []))
        return self.objs[-1][1]

    def _add(self, name, pts, faces):
        f = self._start(name)
        base = len(self.v)
        self.v += [tuple(map(float, p)) for p in pts]
        f += [[base + i + 1 for i in fc] for fc in faces]

    def box(self, name, x0, x1, y0, y1, z0, z1):
        p = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        self._add(name, p, [[3, 2, 1, 0], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])

    def prism_y(self, name, prof_xz, y0, y1):
        """Extrude a profile polygon in the x-z plane along y (closed, convex or simple)."""
        n = len(prof_xz)
        p = [(x, y0, z) for x, z in prof_xz] + [(x, y1, z) for x, z in prof_xz]
        faces = [list(range(n))[::-1], list(range(n, 2 * n))]
        faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        self._add(name, p, faces)

    def prism_x(self, name, prof_yz, x0, x1):
        """Extrude a profile in the y-z plane along x."""
        n = len(prof_yz)
        p = [(x0, y, z) for y, z in prof_yz] + [(x1, y, z) for y, z in prof_yz]
        faces = [list(range(n)), list(range(n, 2 * n))[::-1]]
        faces += [[(i + 1) % n, i, n + i, n + (i + 1) % n] for i in range(n)]
        self._add(name, p, faces)

    def prism_z(self, name, poly_xy, z0, z1):
        n = len(poly_xy)
        p = [(x, y, z0) for x, y in poly_xy] + [(x, y, z1) for x, y in poly_xy]
        faces = [list(range(n))[::-1], list(range(n, 2 * n))]
        faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        self._add(name, p, faces)

    def beam(self, name, a, b, w, h=None):
        """Square-ish member from point a to point b, w wide (h deep)."""
        h = h or w
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            return
        d /= L
        up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([0, 1.0, 0])
        s1 = np.cross(d, up)
        s1 /= np.linalg.norm(s1)
        s2 = np.cross(s1, d)
        c = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        p = [a + s1 * u + s2 * v for u, v in c] + [b + s1 * u + s2 * v for u, v in c]
        self._add(name, p, [[3, 2, 1, 0], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])

    def write(self, path, xf=None, header=""):
        V = np.array(self.v)
        if xf is not None:
            V = xf(V)
        with open(path, "w") as f:
            f.write(header)
            for x, y, z in V:
                f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
            for name, faces in self.objs:
                f.write(f"o {name}\n")
                for fc in faces:
                    f.write("f " + " ".join(map(str, fc)) + "\n")


def placer(origin_rc, angle_deg, win, scale=500):
    """Local metres (x along the axis, y 90 deg CCW of it, z elevation) to the site model's OBJ mm:
    x east, y north, z up from the base board (BASE_Z). origin_rc: DEM (row, col) of local (0, 0)."""
    r0, r1, c0, c1 = win
    mm = 1000.0 / scale
    a = math.radians(angle_deg)
    ox = (origin_rc[1] - c0) * mm
    oy = -(origin_rc[0] - r0) * mm

    def xf(V):
        e = V[:, 0] * math.cos(a) - V[:, 1] * math.sin(a)
        n = V[:, 0] * math.sin(a) + V[:, 1] * math.cos(a)
        return np.c_[ox + e * mm, oy + n * mm, (V[:, 2] - BASE_Z) * mm]
    return xf


# --------------------------------------------------------------------------- St. Andrews
DAM = dict(
    bay=36.1, pier_t=3.5, pier_len=14.0, nose=2.5, n_bays=6,
    sill_top=214.5, sill_base=208.5, sill_w=8.0, pier_base=207.0, pier_top=222.0,
    chord_lo=222.0, chord_hi=226.0, truss_y=4.0, panel=4.5,
    deck_lo=226.0, deck_hi=227.0, deck_w=12.0, walk_w=3.0,
    lock_w=13.7, lock_wall=4.0, lock_len=62.0, lock_top=224.3, lock_floor=212.0, lock_gap=6.0,
    abut_len=10.0, abut_w=16.0, abut_lo=214.0,
    curtain_dx=2.4, water_up=219.7, water_dn=217.7,
)


def dam_layout(d=DAM):
    """x positions along the axis (0 = east face of the lock's east wall)."""
    piers, x = [], 0.0
    for k in range(d["n_bays"] - 1):
        x += d["bay"]
        piers.append((x, x + d["pier_t"]))
        x += d["pier_t"]
    east = x + d["bay"]
    lock_e = (-d["lock_wall"], 0.0)
    lock_w = (-d["lock_wall"] * 2 - d["lock_w"], -d["lock_wall"] - d["lock_w"])
    west_ab = (lock_w[0] - d["abut_len"], lock_w[0])
    east_ab = (east, east + d["abut_len"])
    return dict(piers=piers, east=east, lock_e=lock_e, lock_w=lock_w, west_ab=west_ab, east_ab=east_ab)


def build_dam_detail(d=DAM):
    m = Mesh()
    L = dam_layout(d)
    # Fixed dam (sill) under the bays.
    m.box("sill_fixed_dam", 0, L["east"], -d["sill_w"] / 2, d["sill_w"] / 2, d["sill_base"], d["sill_top"])
    # Piers with pointed upstream (south, -y) noses.
    for i, (a, b) in enumerate(L["piers"]):
        hl = d["pier_len"] / 2
        poly = [(a, -hl), (a, hl), (b, hl), (b, -hl), ((a + b) / 2, -hl - d["nose"])]
        m.prism_z(f"pier_{i + 1}", [(a, hl), (b, hl), (b, -hl), ((a + b) / 2, -hl - d["nose"]), (a, -hl)][::-1],
                  d["pier_base"], d["pier_top"])
        m.box(f"pier_{i + 1}_cap", a - 0.3, b + 0.3, -hl, hl, d["pier_top"] - 0.6, d["pier_top"])
    # Abutments.
    for nm, (a, b) in (("west_abutment", L["west_ab"]), ("east_abutment", L["east_ab"])):
        m.box(nm, a, b, -d["abut_w"] / 2, d["abut_w"] / 2, d["abut_lo"], d["deck_lo"])
    # Lock: walls run upstream (south, -y) from the dam line, floor, mitre gates at both ends.
    y_dn, y_up = d["lock_gap"], -(d["lock_len"] + d["lock_gap"])
    for nm, (a, b) in (("lock_east_wall", L["lock_e"]), ("lock_west_wall", L["lock_w"])):
        m.box(nm, a, b, y_up - 6, y_dn + 10, d["lock_floor"] - 1.5, d["lock_top"])
    m.box("lock_floor", L["lock_w"][1], L["lock_e"][0], y_up - 6, y_dn + 10, d["lock_floor"] - 1.5, d["lock_floor"])
    for gy in (y_dn, y_up):
        xa, xb = L["lock_w"][1], L["lock_e"][0]
        mid = (xa + xb) / 2
        for sgn, xs in ((1, xa), (-1, xb)):
            m.beam(f"lock_gate_{'dn' if gy == y_dn else 'up'}", (xs, gy, d["lock_floor"] + (d["lock_top"] - d["lock_floor"]) / 2),
                   (mid, gy - 2.6, d["lock_floor"] + (d["lock_top"] - d["lock_floor"]) / 2), 0.8,
                   d["lock_top"] - d["lock_floor"] - 0.3)
    # Trusses: supports at lock west wall, lock east wall (x=0), pier centres, east abutment.
    sup = [L["lock_w"][1] - d["lock_wall"] / 2, 0.0] + [(a + b) / 2 for a, b in L["piers"]] + [L["east"]]
    for s, (xa, xb) in enumerate(zip(sup[:-1], sup[1:])):
        n = max(2, int(round((xb - xa) / d["panel"])))
        xs = np.linspace(xa, xb, n + 1)
        for ty in (-d["truss_y"], d["truss_y"]):
            nm = f"truss_span_{s + 1}"
            m.beam(nm, (xa, ty, d["chord_lo"]), (xb, ty, d["chord_lo"]), 0.6, 0.6)
            m.beam(nm, (xa, ty, d["chord_hi"]), (xb, ty, d["chord_hi"]), 0.6, 0.7)
            for i, x in enumerate(xs):
                m.beam(nm, (x, ty, d["chord_lo"]), (x, ty, d["chord_hi"]), 0.35)
                if i < n:
                    # Pratt diagonals: slope down toward mid-span.
                    if x + (xs[1] - xs[0]) / 2 < (xa + xb) / 2:
                        m.beam(nm, (x, ty, d["chord_hi"]), (xs[i + 1], ty, d["chord_lo"]), 0.3)
                    else:
                        m.beam(nm, (x, ty, d["chord_lo"]), (xs[i + 1], ty, d["chord_hi"]), 0.3)
        for x in xs:
            m.beam(f"cross_frames_{s + 1}", (x, -d["truss_y"], d["chord_lo"]), (x, d["truss_y"], d["chord_lo"]), 0.35)
            m.beam(f"cross_frames_{s + 1}", (x, -d["truss_y"], d["chord_hi"]), (x, d["truss_y"], d["chord_hi"]), 0.35)
        for i in range(n):
            m.beam(f"cross_frames_{s + 1}", (xs[i], -d["truss_y"], d["chord_lo"]), (xs[i + 1], d["truss_y"], d["chord_lo"]), 0.2)
    # Deck, sidewalk (north, downstream side), railings.
    x0, x1 = L["west_ab"][0], L["east_ab"][1]
    m.box("deck", x0, x1, -d["deck_w"] / 2, d["deck_w"] / 2, d["deck_lo"], d["deck_hi"])
    m.box("sidewalk", x0, x1, d["deck_w"] / 2 - d["walk_w"], d["deck_w"] / 2, d["deck_hi"], d["deck_hi"] + 0.25)
    for ry in (-d["deck_w"] / 2 + 0.15, d["deck_w"] / 2 - 0.15, d["deck_w"] / 2 - d["walk_w"]):
        top = d["deck_hi"] + (1.3 if ry > 0 else 1.0)
        m.beam("railings", (x0, ry, top), (x1, ry, top), 0.15)
        for x in np.arange(x0, x1 + 0.1, 3.0):
            m.beam("railings", (x, ry, d["deck_hi"]), (x, ry, top), 0.12)
    # Curtain frames: hang from the bottom chord on the upstream side, lean onto the sill; curtains down.
    for b in range(d["n_bays"]):
        xa = 0.0 if b == 0 else L["piers"][b - 1][1]
        xb = L["piers"][b][0] if b < len(L["piers"]) else L["east"]
        xs = np.arange(xa + d["curtain_dx"] / 2, xb, d["curtain_dx"])
        for x in xs:
            m.beam("curtain_frames", (x, -d["truss_y"], d["chord_lo"]), (x, -d["sill_w"] / 2, d["sill_top"]), 0.25)
        # Curtain (Douglas fir laths) as a thin inclined plate between the frames, to above the upstream water.
        top_z = d["water_up"] + 0.6
        tf = (top_z - d["sill_top"]) / (d["chord_lo"] - d["sill_top"])
        ytop = -d["sill_w"] / 2 + (d["sill_w"] / 2 - d["truss_y"]) * tf
        for xa_, xb_ in zip(xs[:-1], xs[1:]):
            m.prism_x("curtains", [(-d["sill_w"] / 2, d["sill_top"]), (ytop, top_z), (ytop + 0.12, top_z),
                                   (-d["sill_w"] / 2 + 0.12, d["sill_top"])], xa_ + 0.15, xb_ - 0.15)
    # Water (separate objects: delete for printing).
    m.box("water_upstream", x0 - 20, x1 + 20, -95, -d["sill_w"] / 2, d["water_up"] - 0.05, d["water_up"])
    m.box("water_downstream", x0 - 20, x1 + 20, d["sill_w"] / 2, 60, d["water_dn"] - 0.05, d["water_dn"])
    return m, L


def build_dam_massing(d=DAM):
    """Simplified wood massing: deck slab, one solid truss beam, piers, curtain plates, lock walls, abutments.
    Bottoms of the river pieces sit on the base board (BASE_Z)."""
    m = Mesh()
    L = dam_layout(d)
    x0, x1 = L["west_ab"][0], L["east_ab"][1]
    parts = []

    def add(name, qty, x0_, x1_, y0, y1, z0, z1, note):
        m.box(name, x0_, x1_, y0, y1, z0, z1)
        parts.append((name, qty, x1_ - x0_, y1 - y0, z1 - z0, z0, note))

    add("D1 DECK", 1, x0, x1, -d["deck_w"] / 2, d["deck_w"] / 2, d["deck_lo"], d["deck_hi"], "road + sidewalk, one slab")
    add("D2 TRUSS BEAM", 1, L["lock_w"][1] - d["lock_wall"] / 2, L["east"], -d["truss_y"] - 0.5, d["truss_y"] + 0.5,
        d["chord_lo"], d["chord_hi"], "solid block for the 7 truss spans; score panel lines if you like")
    for i, (a, b) in enumerate(L["piers"]):
        add(f"D3 PIER {i + 1}", 1, a, b, -d["pier_len"] / 2, d["pier_len"] / 2, BASE_Z, d["pier_top"],
            "cut the upstream (south) end to a point if you like")
    for b in range(d["n_bays"]):
        xa = 0.0 if b == 0 else L["piers"][b - 1][1]
        xb = L["piers"][b][0] if b < len(L["piers"]) else L["east"]
        add(f"D4 CURTAIN {b + 1}", 1, xa, xb, -d["truss_y"] - 0.5, -d["truss_y"], BASE_Z, d["chord_lo"],
            "thin plate under the upstream face of the truss: the curtains, down")
    y_dn, y_up = d["lock_gap"], -(d["lock_len"] + d["lock_gap"])
    for nm, (a, b) in (("D5 LOCK WALL E", L["lock_e"]), ("D5 LOCK WALL W", L["lock_w"])):
        add(nm, 1, a, b, y_up - 6, y_dn + 10, BASE_Z, d["lock_top"], "runs upstream (south) from the dam line")
    for nm, (a, b) in (("D6 ABUTMENT W", L["west_ab"]), ("D6 ABUTMENT E", L["east_ab"])):
        add(nm, 1, a, b, -d["abut_w"] / 2, d["abut_w"] / 2, BASE_Z, d["deck_lo"], "trim the bottom to the card at the bank")
    return m, parts, L


# --------------------------------------------------------------------------- Floodway outlet
OUTLET = dict(length=88.0, crest=223.3, crest_w=3.5, approach_top=220.8, approach_len=20.0, upstream_face_lo=217.5,
              roll_len=8.0, basin_floor=216.0, basin_len=26.0, baffle_y=9.0, baffle=(1.2, 1.4, 1.2), baffle_dx=2.6,
              end_sill_h=1.2, end_sill_w=2.0, wall_t=1.5, wall_top=225.5, wall_top_dn=219.5, slab_t=1.0,
              water_dn=218.1, wing=10.0)


def build_outlet_detail(o=OUTLET):
    m = Mesh()
    Lx = o["length"]
    y_roll0 = o["crest_w"]
    y_basin0 = y_roll0 + o["roll_len"]
    y_basin1 = y_basin0 + o["basin_len"]
    # Approach slab (upstream, -y).
    m.box("approach_slab", 0, Lx, -o["approach_len"], 0, o["approach_top"] - o["slab_t"], o["approach_top"])
    # Crest and rollway (ogee-like curve) as a y-z profile extruded along x.
    prof = [(0.0, o["upstream_face_lo"]), (0.0, o["crest"])]
    for t in np.linspace(0, 1, 9):
        y = y_roll0 + t * o["roll_len"]
        z = o["crest"] - (o["crest"] - o["basin_floor"]) * (3 * t ** 2 - 2 * t ** 3)
        prof.append((y, z))
    prof += [(y_basin0, o["basin_floor"] - o["slab_t"]), (0.0, o["basin_floor"] - o["slab_t"])]
    prof = [prof[0], prof[1], (y_roll0, o["crest"])] + prof[3:]
    m.prism_x("crest_and_rollway", prof, 0, Lx)
    m.box("stilling_basin_floor", 0, Lx, y_basin0, y_basin1, o["basin_floor"] - o["slab_t"], o["basin_floor"])
    bw, bl, bh = o["baffle"]
    for k, x in enumerate(np.arange(bw, Lx - bw, o["baffle_dx"])):
        yb = y_basin0 + o["baffle_y"] + (0 if k % 2 == 0 else 0)
        m.box("baffle_blocks", x, x + bw, yb, yb + bl, o["basin_floor"], o["basin_floor"] + bh)
    for k, x in enumerate(np.arange(0, Lx, 2 * bw)):
        m.box("end_sill_dentates", x, min(x + bw, Lx), y_basin1 - o["end_sill_w"], y_basin1, o["basin_floor"],
              o["basin_floor"] + o["end_sill_h"])
    # Training walls each side: full height to the end of the rollway, then sloping down to the end of the basin; wing walls.
    for nm, xa, xb in (("training_wall_sw", -o["wall_t"], 0), ("training_wall_ne", Lx, Lx + o["wall_t"])):
        prof = [(-o["approach_len"], o["basin_floor"] - o["slab_t"]), (-o["approach_len"], o["wall_top"]),
                (y_basin0, o["wall_top"]), (y_basin1, o["wall_top_dn"]), (y_basin1, o["basin_floor"] - o["slab_t"])]
        m.prism_x(nm, prof, xa, xb)
    for nm, x, sgn in (("wing_wall_sw", -o["wall_t"], -1), ("wing_wall_ne", Lx + o["wall_t"], 1)):
        a = (x, y_basin1)
        b = (x + sgn * o["wing"] * 0.7, y_basin1 + o["wing"] * 0.7)
        m.beam(nm, (a[0], a[1], (o["wall_top_dn"] + o["basin_floor"]) / 2), (b[0], b[1], (o["wall_top_dn"] + o["basin_floor"]) / 2),
               o["wall_t"], o["wall_top_dn"] - o["basin_floor"] + 1.0)
    m.box("water_downstream", -15, Lx + 15, y_basin0 - 2, y_basin1 + 25, o["water_dn"] - 0.05, o["water_dn"])
    return m


def build_outlet_massing(o=OUTLET):
    m = Mesh()
    parts = []
    Lx = o["length"]
    y_basin0 = o["crest_w"] + o["roll_len"]
    y_basin1 = y_basin0 + o["basin_len"]
    prof = [(0.0, BASE_Z), (0.0, o["crest"]), (o["crest_w"], o["crest"]), (y_basin0, BASE_Z + 0.4), (y_basin0, BASE_Z)]
    m.prism_x("F1 CREST + ROLLWAY", prof, 0, Lx)
    parts.append(("F1 CREST + ROLLWAY", 1, Lx, y_basin0, o["crest"] - BASE_Z, BASE_Z,
                  "a prism: vertical upstream face, flat crest 3.5 m, sloping rollway (see section)"))
    m.box("F2 APPROACH SLAB", 0, Lx, -o["approach_len"], 0, BASE_Z, o["approach_top"])
    parts.append(("F2 APPROACH SLAB", 1, Lx, o["approach_len"], o["approach_top"] - BASE_Z, BASE_Z,
                  "upstream apron; trim to the card where the channel bed rises"))
    for nm, xa in (("F3 TRAINING WALL SW", -o["wall_t"]), ("F3 TRAINING WALL NE", Lx)):
        prof = [(-o["approach_len"], BASE_Z), (-o["approach_len"], o["wall_top"]), (y_basin0, o["wall_top"]),
                (y_basin1, o["wall_top_dn"]), (y_basin1, BASE_Z)]
        m.prism_x(nm, prof, xa, xa + o["wall_t"])
        parts.append((nm, 1, o["wall_t"], o["approach_len"] + y_basin1, o["wall_top"] - BASE_Z, BASE_Z,
                      "a profile: level top, then sloping down to the end of the basin"))
    m.box("F4 END SILL", 0, Lx, y_basin1 - o["end_sill_w"], y_basin1, BASE_Z, BASE_Z + 0.8)
    parts.append(("F4 END SILL", 1, Lx, o["end_sill_w"], 0.8, BASE_Z, "optional: the basin is underwater; a low strip"))
    return m, parts


# --------------------------------------------------------------------------- preview render
def render(mesh, png, view=(-35, 25), size=1800, skip=("water",), title="", xrange=None):
    V = np.array(mesh.v)
    az, el = map(math.radians, view)
    R = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    P = V @ R.T
    Xp = P[:, 0]
    Yp = -(P[:, 1] * math.sin(el) + P[:, 2] * math.cos(el))
    depth = P[:, 1] * math.cos(el) - P[:, 2] * math.sin(el)
    def inside(f):
        return xrange is None or any(xrange[0] <= V[i - 1, 0] <= xrange[1] for i in f)
    faces = [(nm, f) for nm, fs in mesh.objs for f in fs if not any(s in nm for s in skip) and "water" not in nm
             and inside(f)]
    water = [(nm, f) for nm, fs in mesh.objs for f in fs if "water" in nm and not any(s in nm for s in skip)]
    if xrange is not None:
        water = []
    used = np.unique([i - 1 for _, f in faces for i in f])
    sx = (size - 80) / max(np.ptp(Xp[used]), np.ptp(Yp[used]) * 1.0)
    ox, oy = 40 - Xp[used].min() * sx, 60 - Yp[used].min() * sx
    W = int(np.ptp(Xp[used]) * sx + 80)
    H = int(np.ptp(Yp[used]) * sx + 100)
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    light = np.array([-0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    allf = [(nm, f, True) for nm, f in water] + [(nm, f, False) for nm, f in faces]
    order = sorted(allf, key=lambda t: (not t[2], -np.mean(depth[np.array(t[1]) - 1])))
    for nm, f, is_w in order:
        idx = np.array(f) - 1
        pts = V[idx]
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        shade = 0.55 + 0.45 * abs(float(n @ light))
        c = (205, 215, 222) if is_w else tuple(int(v * shade) for v in ((175, 175, 175) if "deck" not in nm else (215, 215, 215)))
        d.polygon([(Xp[i] * sx + ox, Yp[i] * sx + oy) for i in idx], fill=c, outline=(60, 60, 60) if not is_w else None)
    if title:
        d.text((20, 15), title, fill=(20, 20, 20))
    im.save(png)


# --------------------------------------------------------------------------- drawings
def dim_h(svg, x1, x2, y, text, ext_from=None, size=2.0):
    svg.line(x1, y, x2, y, stroke=INK, stroke_width=0.15)
    for x in (x1, x2):
        svg.line(x - 0.9, y + 0.9, x + 0.9, y - 0.9, stroke=INK, stroke_width=0.25)
        if ext_from is not None:
            svg.line(x, ext_from, x, y + (1.2 if y > ext_from else -1.2), stroke=INK, stroke_width=0.08)
    svg.text((x1 + x2) / 2, y - 1.0, text, size=size, fill=INK, anchor="middle")


def dim_v(svg, x, y1, y2, text, ext_from=None, size=2.0):
    svg.line(x, y1, x, y2, stroke=INK, stroke_width=0.15)
    for y in (y1, y2):
        svg.line(x - 0.9, y + 0.9, x + 0.9, y - 0.9, stroke=INK, stroke_width=0.25)
        if ext_from is not None:
            svg.line(ext_from, y, x + (1.2 if x > ext_from else -1.2), y, stroke=INK, stroke_width=0.08)
    svg.text(x + 1.2, (y1 + y2) / 2 + 0.7, text, size=size, fill=INK, rotate=None)


def rect(svg, x, y, w, h, fill="#ffffff", sw=0.25, **kw):
    svg.rect(x, y, w, h, fill=fill, stroke=INK, stroke_width=sw, **kw)


def fmt(m_):
    return f"{m_:.1f} m / {m_ * 2:.1f} mm"


def table(svg, x, y, rows, head, widths, size=2.0):
    lh = size * 1.7
    cx = [x]
    for w in widths[:-1]:
        cx.append(cx[-1] + w)
    for c, h in zip(cx, head):
        svg.text(c, y, h, size=size * 0.85, fill=INK, spacing=0.15, opacity=0.8)
    svg.line(x, y + lh * 0.4, x + sum(widths), y + lh * 0.4, stroke=INK, stroke_width=0.2)
    yy = y + lh * 1.2
    for r in rows:
        for c, v in zip(cx, r):
            svg.text(c, yy, str(v), size=size, fill=INK)
        yy += lh
    svg.line(x, yy - lh * 0.6, x + sum(widths), yy - lh * 0.6, stroke=INK, stroke_width=0.2)
    return yy


def dam_drawings(parts, L, d=DAM, path_base=f"{OUT}/st_andrews_dam_massing_drawings"):
    Wp, Hp = 914.4, 609.6
    svg = Svg(Wp, Hp)
    svg.layer("DRAWING")
    s = 1000.0 / 500                         # plan and elevation at 1:500: model size, usable as templates
    m_ = 1000.0 / 500
    x0, x1 = L["west_ab"][0], L["east_ab"][1]
    ox = 60 - x0 * s
    svg.text(30, 26, "ST. ANDREWS LOCK AND DAM (PTH 44 BRIDGE) · SIMPLIFIED MASSING FOR A 1:500 WOOD MODEL", size=4.5,
             fill=INK, spacing=0.3)
    svg.text(30, 33, "PLAN AND ELEVATION AT 1:500 = MODEL SIZE (PRINT AT 100% TO USE AS TEMPLATES); SECTION AT 1:100. DIMENSIONS: REAL METRES / MODEL MILLIMETRES AT 1:500. "
             "RECONSTRUCTION FROM PUBLISHED FIGURES AND LIDAR: ITEMS MARKED * ARE ESTIMATES.", size=2.3, fill=INK)
    # ---- Elevation (looking north, from upstream), heights from the base board.
    zb = 95 + (d["deck_hi"] + 1.5 - BASE_Z) * s
    Y = lambda z: zb - (z - BASE_Z) * s
    X = lambda x: ox + x * s
    svg.text(30, 48, "ELEVATION · LOOKING NORTH (DOWNSTREAM) FROM UPSTREAM · 1:500", size=2.8, fill=INK, spacing=0.3)
    svg.line(X(x0 - 8), Y(BASE_Z), X(x1 + 8), Y(BASE_Z), stroke=INK, stroke_width=0.4)
    svg.text(X(x1 + 9), Y(BASE_Z) + 0.8, f"BASE BOARD TOP = {BASE_Z} M", size=1.8, fill=INK)
    svg.line(X(x0 - 8), Y(d["water_up"]), X(x1 + 8), Y(d["water_up"]), stroke=INK, stroke_width=0.12,
             stroke_dasharray="2 1")
    svg.text(X(x1 + 9), Y(d["water_up"]) + 0.8, f"WATER ABOVE DAM {d['water_up']} M (LIDAR)", size=1.8, fill=INK)
    for p in parts:
        nm, q, lx, ly, lz, z0, note = p
    rect(svg, X(x0), Y(d["deck_hi"]), (x1 - x0) * s, (d["deck_hi"] - d["deck_lo"]) * s, fill="#dddddd")
    tx0 = L["lock_w"][1] - d["lock_wall"] / 2
    rect(svg, X(tx0), Y(d["chord_hi"]), (L["east"] - tx0) * s, (d["chord_hi"] - d["chord_lo"]) * s, fill="#ffffff")
    for a, b in L["piers"]:
        rect(svg, X(a), Y(d["pier_top"]), (b - a) * s, (d["pier_top"] - BASE_Z) * s, fill="#bbbbbb")
    for b in range(d["n_bays"]):
        xa = 0.0 if b == 0 else L["piers"][b - 1][1]
        xb = L["piers"][b][0] if b < len(L["piers"]) else L["east"]
        rect(svg, X(xa), Y(d["chord_lo"]), (xb - xa) * s, (d["chord_lo"] - BASE_Z) * s, fill="#f2f2f2", sw=0.12)
        for xc in np.arange(xa + d["curtain_dx"] / 2, xb, d["curtain_dx"]):
            svg.line(X(xc), Y(d["chord_lo"]), X(xc), Y(BASE_Z), stroke=INK, stroke_width=0.06, opacity=0.6)
        dim_h(svg, X(xa), X(xb), Y(BASE_Z) + 8, fmt(xb - xa) + "*", ext_from=Y(BASE_Z), size=1.4)
    for a, b in (L["lock_w"], L["lock_e"]):
        rect(svg, X(a), Y(d["lock_top"]), (b - a) * s, (d["lock_top"] - BASE_Z) * s, fill="#999999")
    for a, b in (L["west_ab"], L["east_ab"]):
        rect(svg, X(a), Y(d["deck_lo"]), (b - a) * s, (d["deck_lo"] - BASE_Z) * s, fill="#cccccc")
    dim_h(svg, X(x0), X(x1), Y(d["deck_hi"]) - 6, "OVERALL " + fmt(x1 - x0), ext_from=Y(d["deck_hi"]))
    dim_h(svg, X(L["lock_w"][0]), X(L["lock_e"][1]), Y(d["deck_hi"]) - 12,
          f"LOCK {fmt(L['lock_e'][1] - L['lock_w'][0])} (CHAMBER {d['lock_w']} M)", ext_from=Y(d["lock_top"]))
    a, b = L["piers"][0]
    dim_h(svg, X(a), X(b), Y(BASE_Z) + 14, f"PIER {fmt(b - a)}*", ext_from=Y(BASE_Z), size=1.6)
    xr = X(x1) + 10
    for z0_, z1_, lab in ((BASE_Z, d["chord_lo"], "TO TRUSS*"), (d["chord_lo"], d["chord_hi"], "TRUSS*"),
                          (d["chord_hi"], d["deck_hi"], "DECK"), (BASE_Z, d["lock_top"], "LOCK WALL")):
        xr += 0 if lab != "LOCK WALL" else 0
    dim_v(svg, X(x1) + 6, Y(d["chord_lo"]), Y(BASE_Z), f"{fmt(d['chord_lo'] - BASE_Z)}*", ext_from=X(x1))
    dim_v(svg, X(x1) + 6, Y(d["chord_hi"]), Y(d["chord_lo"]), f"{fmt(d['chord_hi'] - d['chord_lo'])}*", ext_from=X(x1))
    dim_v(svg, X(x1) + 6, Y(d["deck_hi"]), Y(d["chord_hi"]), f"{fmt(d['deck_hi'] - d['chord_hi'])}", ext_from=X(x1))
    dim_v(svg, X(x0) - 22, Y(d["lock_top"]), Y(BASE_Z), f"{fmt(d['lock_top'] - BASE_Z)}", ext_from=X(L["lock_w"][0]))
    svg.text(X(x0) - 4, Y(d["deck_hi"]) + 0.8, f"DECK TOP {d['deck_hi']} M*", size=1.7, fill=INK, anchor="end")
    # ---- Plan.
    py0 = zb + 45
    PY = lambda y: py0 + (d["lock_len"] + d["lock_gap"] + 6 - y) * s * 0.0 + (12 - y) * s
    svg.text(30, py0 + (12 - (DAM["lock_gap"] + 10)) * s - 5, "PLAN · NORTH UP · DAM AXIS ROTATED ABOUT 4° (EAST END NORTH) ON THE SITE · 1:500", size=2.8,
             fill=INK, spacing=0.3)
    y_dn, y_up = d["lock_gap"], -(d["lock_len"] + d["lock_gap"])
    for a, b in (L["lock_w"], L["lock_e"]):
        rect(svg, X(a), PY(y_dn + 10), (b - a) * s, (y_dn + 10 - (y_up - 6)) * s, fill="#999999")
    for a, b in L["piers"]:
        rect(svg, X(a), PY(d["pier_len"] / 2), (b - a) * s, d["pier_len"] * s, fill="#bbbbbb")
    for a, b in (L["west_ab"], L["east_ab"]):
        rect(svg, X(a), PY(d["abut_w"] / 2), (b - a) * s, d["abut_w"] * s, fill="#cccccc")
    rect(svg, X(x0), PY(d["deck_w"] / 2), (x1 - x0) * s, d["deck_w"] * s, fill="none", sw=0.35)
    svg.line(X(x0), PY(d["deck_w"] / 2 - d["walk_w"]), X(x1), PY(d["deck_w"] / 2 - d["walk_w"]), stroke=INK,
             stroke_width=0.12, stroke_dasharray="1.5 0.8")
    dim_v(svg, X(x1) + 6, PY(d["deck_w"] / 2), PY(-d["deck_w"] / 2), "DECK " + fmt(d["deck_w"]) + "*", ext_from=X(x1))
    dim_v(svg, X(L["piers"][2][0]) - 4, PY(d["pier_len"] / 2), PY(-d["pier_len"] / 2), "PIER " + fmt(d["pier_len"]) + "*",
          ext_from=X(L["piers"][2][0]))
    dim_v(svg, X(L["lock_w"][0]) - 22, PY(y_dn + 10), PY(y_up - 6), "LOCK WALLS " + fmt(y_dn + 10 - (y_up - 6)),
          ext_from=X(L["lock_w"][0]))
    svg.text(X(L["lock_w"][1]) + 1, PY(y_up - 2), "LOCK", size=1.8, fill=INK)
    svg.text(X(x0) - 3, PY(-y_up + 99) if False else PY(y_up - 10), "UPSTREAM (SOUTH) ↓", size=1.8, fill=INK)
    cx, cy = X(x1) + 40, PY(0)
    svg.circle(cx, cy, 5, fill="none", stroke=INK, stroke_width=0.3)
    svg.line(cx, cy + 4, cx, cy - 9, stroke=INK, stroke_width=0.4)
    svg.text(cx, cy - 10.5, "N", size=2.4, fill=INK, anchor="middle")
    # ---- Cross section through a bay (looking east), enlarged.
    sy0 = PY(y_up - 6) + 22
    s = 1000.0 / 100
    sx0 = 60
    svg.text(30, sy0 - 6, "SECTION THROUGH A BAY · LOOKING EAST · 1:100", size=2.8, fill=INK, spacing=0.3)
    zb2 = sy0 + (d["deck_hi"] + 2 - BASE_Z) * s
    SY = lambda z: zb2 - (z - BASE_Z) * s
    SX = lambda y: sx0 + 10 + (y + 10) * s
    svg.line(SX(-12), SY(BASE_Z), SX(12), SY(BASE_Z), stroke=INK, stroke_width=0.4)
    rect(svg, SX(-d["deck_w"] / 2), SY(d["deck_hi"]), d["deck_w"] * s, (d["deck_hi"] - d["deck_lo"]) * s, fill="#dddddd")
    rect(svg, SX(-d["truss_y"] - 0.5), SY(d["chord_hi"]), (2 * d["truss_y"] + 1) * s, (d["chord_hi"] - d["chord_lo"]) * s)
    rect(svg, SX(-d["truss_y"] - 0.5), SY(d["chord_lo"]), 0.5 * s, (d["chord_lo"] - BASE_Z) * s, fill="#f2f2f2")
    svg.text(SX(-d["truss_y"] - 0.5) - 1, SY((d["chord_lo"] + BASE_Z) / 2), "CURTAINS D4", size=1.6, fill=INK, anchor="end")
    dim_h(svg, SX(-d["deck_w"] / 2), SX(d["deck_w"] / 2), SY(d["deck_hi"]) - 5, "DECK " + fmt(d["deck_w"]) + "*",
          ext_from=SY(d["deck_hi"]), size=1.6)
    dim_h(svg, SX(-d["truss_y"] - 0.5), SX(d["truss_y"] + 0.5), SY(d["chord_lo"]) + 4,
          "TRUSS " + fmt(2 * d["truss_y"] + 1) + "*", ext_from=SY(d["chord_lo"]), size=1.6)
    xd = SX(d["deck_w"] / 2) + 8
    dim_v(svg, xd, SY(d["deck_hi"]), SY(d["deck_lo"]), "DECK " + fmt(d["deck_hi"] - d["deck_lo"]) + "*", ext_from=SX(d["deck_w"] / 2), size=1.8)
    dim_v(svg, xd, SY(d["chord_hi"]), SY(d["chord_lo"]), "TRUSS D2 " + fmt(d["chord_hi"] - d["chord_lo"]) + "*", ext_from=SX(d["truss_y"] + 0.5), size=1.8)
    dim_v(svg, xd, SY(d["chord_lo"]), SY(BASE_Z), "PIERS / CURTAINS " + fmt(d["chord_lo"] - BASE_Z) + "*", ext_from=SX(-d["truss_y"]), size=1.8)
    svg.text(SX(-12), SY(d["water_up"]) - 0.6, "UPSTREAM ←", size=1.6, fill=INK)
    svg.text(SX(12), SY(d["water_up"]) - 0.6, "→ DOWNSTREAM", size=1.6, fill=INK, anchor="end")
    # ---- Parts list.
    rows = []
    for nm, q, lx, ly, lz, z0, note in parts:
        rows.append([nm, q, f"{lx:.1f} × {ly:.1f} × {lz:.1f}", f"{lx * m_:.1f} × {ly * m_:.1f} × {lz * m_:.1f}",
                     f"{(z0 - BASE_Z) * m_:.0f}", note])
    svg.text(sx0 + 260, sy0 - 6, "PARTS · WOOD MASSING · LENGTH × WIDTH × HEIGHT", size=2.8, fill=INK, spacing=0.3)
    table(svg, sx0 + 260, sy0, rows, ["PART", "QTY", "REAL (M)", "MODEL 1:500 (MM)", "BOTTOM ABOVE BASE (MM)", "NOTE"],
          [40, 10, 40, 42, 34, 120], size=1.9)
    notes = ["ASSEMBLY: glue piers, lock walls and abutments on the base board; curtain plates between piers on the upstream",
             "(south) face; the truss beam on the piers and lock walls; the deck on top. Abutments sit on the bank card: trim to fit.",
             "PUBLISHED: ~270 m overall, 240.1 m between abutments, 5 piers / 6 bays, ~40 m trusses, 15 m piers, 6 m sill, 89 curtains,",
             "lock 13.7 m wide, ~62 m long. LIDAR: dam axis 239 m, lock wall tops 224.3 m, water 219.7 / 217.7 m. * = estimate."]
    for i, t in enumerate(notes):
        svg.text(30, Hp - 26 + i * 4.2, t, size=2.0, fill=INK)
    svg.save(path_base + ".svg")


def outlet_drawings(parts, o=OUTLET, path_base=f"{OUT}/floodway_outlet_massing_drawings"):
    Wp, Hp = 914.4, 609.6
    svg = Svg(Wp, Hp)
    svg.layer("DRAWING")
    s = 1000.0 / 250
    m_ = 2.0
    Lx = o["length"]
    yb0 = o["crest_w"] + o["roll_len"]
    yb1 = yb0 + o["basin_len"]
    svg.text(30, 26, "RED RIVER FLOODWAY OUTLET CONTROL STRUCTURE · SIMPLIFIED MASSING FOR A 1:500 WOOD MODEL", size=4.5,
             fill=INK, spacing=0.3)
    svg.text(30, 33, "DRAWN AT 1:250 (TWICE MODEL SIZE). DIMENSIONS: REAL METRES / MODEL MILLIMETRES AT 1:500. "
             "RECONSTRUCTION FROM PUBLISHED FIGURES AND LIDAR: ITEMS MARKED * ARE ESTIMATES.", size=2.3, fill=INK)
    # Plan (crest along x).
    ox, oy = 70, 70
    X = lambda x: ox + x * s
    Y = lambda y: oy + (y + o["approach_len"]) * s
    svg.text(30, oy - 10, "PLAN · LOCAL AXES (CREST RUNS SW–NE AT 45° ON THE SITE; FLOW TO THE NW) · 1:250", size=2.8,
             fill=INK, spacing=0.3)
    rect(svg, X(0), Y(-o["approach_len"]), Lx * s, o["approach_len"] * s, fill="#eeeeee")
    rect(svg, X(0), Y(0), Lx * s, yb0 * s, fill="#bbbbbb")
    svg.line(X(0), Y(o["crest_w"]), X(Lx), Y(o["crest_w"]), stroke=INK, stroke_width=0.15)
    rect(svg, X(0), Y(yb1 - o["end_sill_w"]), Lx * s, o["end_sill_w"] * s, fill="#999999")
    for xa in (-o["wall_t"], Lx):
        rect(svg, X(xa), Y(-o["approach_len"]), o["wall_t"] * s, (o["approach_len"] + yb1) * s, fill="#888888")
    dim_h(svg, X(0), X(Lx), Y(-o["approach_len"]) - 6, "CREST LENGTH " + fmt(Lx) + " (LIDAR)", ext_from=Y(-o["approach_len"]))
    dim_v(svg, X(Lx) + 10, Y(-o["approach_len"]), Y(0), "APPROACH " + fmt(o["approach_len"]) + "*", ext_from=X(Lx))
    dim_v(svg, X(Lx) + 10, Y(0), Y(yb0), "CREST + ROLLWAY " + fmt(yb0) + "*", ext_from=X(Lx))
    dim_v(svg, X(Lx) + 10, Y(yb0), Y(yb1), "BASIN " + fmt(o["basin_len"]) + "*", ext_from=X(Lx))
    svg.text(X(Lx / 2), Y(-o["approach_len"] / 2), "F2 APPROACH SLAB · FLOW ↓", size=2.0, fill=INK, anchor="middle")
    svg.text(X(Lx / 2), Y(yb0 / 2) + 0.7, "F1 CREST + ROLLWAY", size=2.0, fill=INK, anchor="middle")
    svg.text(X(Lx / 2), Y((yb0 + yb1) / 2), "STILLING BASIN (UNDERWATER: BASE BOARD)", size=2.0, fill=INK, anchor="middle")
    svg.text(X(-o["wall_t"]) - 1.5, Y(yb1 / 2), "F3", size=2.0, fill=INK, anchor="end")
    svg.text(X(Lx + o["wall_t"]) + 1.5, Y(yb1 / 2), "F3", size=2.0, fill=INK)
    # Section along the flow.
    sx, sy = 480, 70
    zb = sy + (o["wall_top"] + 2 - BASE_Z) * s
    SX = lambda y: sx + (y + o["approach_len"]) * s
    SY = lambda z: zb - (z - BASE_Z) * s
    svg.text(sx, sy - 10, "SECTION ALONG THE FLOW · 1:250", size=2.8, fill=INK, spacing=0.3)
    svg.line(SX(-o["approach_len"] - 5), SY(BASE_Z), SX(yb1 + 5), SY(BASE_Z), stroke=INK, stroke_width=0.4)
    svg.text(SX(yb1 + 6), SY(BASE_Z) + 0.8, f"BASE BOARD {BASE_Z} M", size=1.7, fill=INK)
    wall = [(-o["approach_len"], BASE_Z), (-o["approach_len"], o["wall_top"]), (yb0, o["wall_top"]),
            (yb1, o["wall_top_dn"]), (yb1, BASE_Z)]
    svg.path("M" + "L".join(f"{SX(y):.2f},{SY(z):.2f}" for y, z in wall) + "Z", fill="#eeeeee", stroke=INK,
             stroke_width=0.15, stroke_dasharray="1 0.6")
    crest = [(0.0, BASE_Z), (0.0, o["crest"]), (o["crest_w"], o["crest"]), (yb0, BASE_Z + 0.4), (yb0, BASE_Z)]
    svg.path("M" + "L".join(f"{SX(y):.2f},{SY(z):.2f}" for y, z in crest) + "Z", fill="#bbbbbb", stroke=INK,
             stroke_width=0.3)
    rect(svg, SX(-o["approach_len"]), SY(o["approach_top"]), o["approach_len"] * s, (o["approach_top"] - BASE_Z) * s,
         fill="#dddddd")
    rect(svg, SX(yb1 - o["end_sill_w"]), SY(BASE_Z + 0.8), o["end_sill_w"] * s, 0.8 * s, fill="#999999")
    svg.line(SX(yb0), SY(o["water_dn"]), SX(yb1 + 5), SY(o["water_dn"]), stroke=INK, stroke_width=0.12,
             stroke_dasharray="2 1")
    svg.text(SX(yb1 + 6), SY(o["water_dn"]) + 0.8, f"RED RIVER {o['water_dn']} M (LIDAR)", size=1.7, fill=INK)
    dim_v(svg, SX(0) - 8, SY(o["crest"]), SY(BASE_Z), f"CREST {o['crest']} M: {fmt(o['crest'] - BASE_Z)}",
          ext_from=SX(0))
    dim_v(svg, SX(-o["approach_len"]) - 8, SY(o["wall_top"]), SY(BASE_Z), f"WALL {fmt(o['wall_top'] - BASE_Z)}*",
          ext_from=SX(-o["approach_len"]))
    dim_v(svg, SX(-o["approach_len"]) + 4, SY(o["approach_top"]), SY(BASE_Z), f"{fmt(o['approach_top'] - BASE_Z)}*",
          ext_from=SX(-o["approach_len"]))
    dim_h(svg, SX(0), SX(o["crest_w"]), SY(o["crest"]) - 5, "CREST " + fmt(o["crest_w"]) + "*", ext_from=SY(o["crest"]),
          size=1.6)
    dim_h(svg, SX(o["crest_w"]), SX(yb0), SY(BASE_Z) + 6, "ROLLWAY " + fmt(o["roll_len"]) + "*", ext_from=SY(BASE_Z),
          size=1.6)
    dim_v(svg, SX(yb1) + 4, SY(o["wall_top_dn"]), SY(BASE_Z), f"{fmt(o['wall_top_dn'] - BASE_Z)}*", ext_from=SX(yb1))
    svg.text(SX(-o["approach_len"] / 2), SY(o["approach_top"]) - 1, "FROM THE FLOODWAY →", size=1.7, fill=INK,
             anchor="middle")
    # Parts.
    rows = [[nm, q, f"{lx:.1f} × {ly:.1f} × {lz:.1f}", f"{lx * m_:.1f} × {ly * m_:.1f} × {lz * m_:.1f}",
             f"{(z0 - BASE_Z) * m_:.0f}", note] for nm, q, lx, ly, lz, z0, note in parts]
    ty = 330
    svg.text(30, ty - 6, "PARTS · WOOD MASSING · LENGTH (ALONG CREST) × DEPTH (ALONG FLOW) × HEIGHT", size=2.8, fill=INK,
             spacing=0.3)
    table(svg, 30, ty, rows, ["PART", "QTY", "REAL (M)", "MODEL 1:500 (MM)", "BOTTOM ABOVE BASE (MM)", "NOTE"],
          [48, 10, 40, 42, 34, 150], size=1.9)
    # 1:500 full-size templates of the profiles.
    tx, tyy = 30, 400
    svg.text(tx, tyy - 6, "FULL-SIZE TEMPLATES AT 1:500 (PRINT AT 100%): CUT THESE PROFILES", size=2.8, fill=INK, spacing=0.3)
    T = 2.0
    zt = tyy + 20
    for k, (pts, lab) in enumerate(((crest, "F1 PROFILE (EXTRUDE 176 MM ALONG THE CREST)"),
                                    (wall, "F3 PROFILE ×2 (3 MM THICK)"))):
        y0 = min(p[0] for p in pts)
        x_off = tx + k * 140
        svg.path("M" + "L".join(f"{x_off + (y - y0) * T:.2f},{zt - (z - BASE_Z) * T:.2f}" for y, z in pts) + "Z",
                 fill="none", stroke=INK, stroke_width=0.25)
        svg.text(x_off, zt + 6, lab, size=1.8, fill=INK)
    notes = ["PUBLISHED: completed 1968; concrete on bedrock, uncontrolled crest; combined rollway and stilling basin; flow drops",
             "about 14 ft (4.3 m) into the Red; design capacity 1,700 m³/s. LIDAR: crest 88 m long at 223.3 m; embankments to about",
             "232 m at each end; Red River 218.1 m. Basin, baffle blocks and walls are underwater or unmeasured: * = estimate.",
             "ASSEMBLY: the crest runs SW–NE at 45° on model B; set F1 on the base board with the vertical face to the SE (upstream)."]
    for i, t in enumerate(notes):
        svg.text(30, Hp - 30 + i * 4.2, t, size=2.0, fill=INK)
    svg.save(path_base + ".svg")


def main():
    os.makedirs(OUT, exist_ok=True)
    hdr = "# {}\n# Reconstruction from published figures + 1 m lidar; see tools/structures.py. Units: {}\n"
    # St. Andrews: local x along the dam axis from the lock's east wall (DEM row 2139, col 1325), angle about 4.07 deg.
    ang_dam = math.degrees(math.atan2(2139 - 2122, 1564 - 1325))
    place_A = placer((2139, 1325), ang_dam, (2030, 2250, 1245, 1645))
    dm, L = build_dam_detail()
    dm.write(f"{OUT}/st_andrews_dam_detail_m.obj", header=hdr.format("St. Andrews Lock and Dam, detailed", "metres, local axes, z = elevation"))
    dm.write(f"{OUT}/st_andrews_dam_detail_1to500.obj", xf=place_A,
             header=hdr.format("St. Andrews Lock and Dam, detailed", "mm, site model A coordinates (z 0 = base board 216.5 m)"))
    mm_, parts_d, _ = build_dam_massing()
    mm_.write(f"{OUT}/st_andrews_dam_massing_m.obj", header=hdr.format("St. Andrews massing", "metres"))
    mm_.write(f"{OUT}/st_andrews_dam_massing_1to500.obj", xf=place_A, header=hdr.format("St. Andrews massing", "mm, model A"))
    render(dm, f"{OUT}/st_andrews_dam_detail.png", view=(-160, 20), skip=("water_upstream",),
           title="ST. ANDREWS LOCK AND DAM · DETAIL (RECONSTRUCTION) · FROM DOWNSTREAM")
    render(dm, f"{OUT}/st_andrews_dam_detail_bay.png", view=(-35, 18), xrange=(-35, 42),
           title="ST. ANDREWS · LOCK, WEST BAY AND PIER 1 FROM UPSTREAM: CURTAINS DOWN, DECK TRUSS, RAILINGS")
    render(mm_, f"{OUT}/st_andrews_dam_massing.png", view=(-35, 22), title="ST. ANDREWS · WOOD MASSING (FROM UPSTREAM)")
    dam_drawings(parts_d, L)
    # Floodway outlet: local x along the crest from its SW end (row 1488, col 1801), at 45 deg; +y downstream (NW).
    place_B = placer((1488, 1801), 45.0, (1300, 1520, 1490, 1890))
    om = build_outlet_detail()
    om.write(f"{OUT}/floodway_outlet_detail_m.obj", header=hdr.format("Floodway outlet control, detailed", "metres"))
    om.write(f"{OUT}/floodway_outlet_detail_1to500.obj", xf=place_B, header=hdr.format("Floodway outlet, detailed", "mm, model B"))
    omm, parts_o = build_outlet_massing()
    omm.write(f"{OUT}/floodway_outlet_massing_m.obj", header=hdr.format("Floodway outlet massing", "metres"))
    omm.write(f"{OUT}/floodway_outlet_massing_1to500.obj", xf=place_B, header=hdr.format("Floodway outlet massing", "mm, model B"))
    render(om, f"{OUT}/floodway_outlet_detail.png", view=(-150, 28), title="FLOODWAY OUTLET CONTROL · DETAIL (RECONSTRUCTION)")
    render(omm, f"{OUT}/floodway_outlet_massing.png", view=(-150, 28), title="FLOODWAY OUTLET · WOOD MASSING")
    outlet_drawings(parts_o)
    print("dam: overall", L["west_ab"][0], L["east_ab"][1], "piers", L["piers"])
    for p in parts_d + parts_o:
        print(p[0], f"{p[2] * 2:.1f} x {p[3] * 2:.1f} x {p[4] * 2:.1f} mm")


if __name__ == "__main__":
    main()
