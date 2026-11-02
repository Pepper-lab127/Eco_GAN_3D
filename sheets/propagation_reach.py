"""Sheet 09b - How far, how fast: the riparian plants of the Lockport list
on one distance scale.

Each row is one plant, drawn at left, with a ground line running out to the
right on a logarithmic scale of distance from the parent, 0.1 m to 3 km.
Every way it moves is drawn where it lands:

  - under the ground (rhizomes, stolons, root suckers): a running line with
    shoots, out to the distance reached in a year (or the clone's reach)
  - layering: stems arching over and rooting
  - ants: short tracks along the ground with the seed at the end
  - falling: seed or acorns on the ground under the parent
  - animals (birds; squirrels and mice caching acorns): flight arcs to where
    the seed is dropped or buried
  - wind: a plume of seed in the air thinning with distance, settling to
    the ground

A bracket under each ground line spans the range. Ranges from a published
figure for the species are drawn in full ink; estimates (most distances)
are drawn lighter, and their labels say EST. Data: drawing/propagation.py.

    python -m sheets.propagation_reach
"""

import argparse
import math
import os

import numpy as np

from drawing import flora, trees
from drawing.propagation import KIND, PLANTS
from drawing.svg import PAPER, Svg

INK = "#111111"


def build(args):
    rng = np.random.default_rng(args.seed)
    Wp, Hp = PAPER[args.paper]
    svg = Svg(Wp, Hp)
    annot = svg.text if args.annotate else (lambda *a_, **k_: None)
    L = dict(grid="DISTANCE GRID, DOTTED · 0.05 mm", ground="GROUND LINES · 0.15 mm",
             under="UNDER THE GROUND · RHIZOMES, STOLONS, SUCKERS · 0.15 mm", arch="LAYERED STEMS · 0.15 mm",
             flight="ANIMALS · FLIGHT AND CARRY ARCS · 0.05 mm", wind="WIND · SEED IN THE AIR",
             ground_seed="SEED ON THE GROUND", ants="ANTS · TRACKS", bracket="RANGE BRACKETS · 0.08 mm",
             wood="PLANTS · WOOD", twigs="PLANTS · TWIGS", leaves="PLANTS · LEAVES",
             stems="PLANTS · STEMS", text="TEXT")
    for v in L.values():
        svg.layer(v)

    x_a, x_b = args.axis
    lo_e, hi_e = math.log10(args.dmin), math.log10(args.dmax)

    def X(d):
        return x_a + (math.log10(max(d, args.dmin)) - lo_e) / (hi_e - lo_e) * (x_b - x_a)

    n = len(PLANTS)
    top = args.top
    row_h = (args.bottom - top) / n
    # Distance grid: decades heavier, with labels top and bottom.
    svg.layer(L["grid"])
    for e in range(int(math.floor(lo_e)), int(math.ceil(hi_e)) + 1):
        for k in range(1, 10):
            d = k * 10.0 ** e
            if not args.dmin <= d <= args.dmax:
                continue
            x = X(d)
            svg.line(x, top, x, args.bottom, stroke=INK, stroke_width=0.08 if k == 1 else 0.04,
                     stroke_dasharray="0.2 0.8" if k == 1 else "0.1 1.6", opacity=0.8 if k == 1 else 0.5)
            if k == 1:
                lab = f"{d:g} M" if d < 1000 else f"{d / 1000:g} KM"
                svg.layer(L["text"])
                for yy in (top - 4, args.bottom + 6):
                    svg.text(x, yy, lab, size=1.8, fill=INK, anchor="middle", spacing=0.15)
                svg.layer(L["grid"])

    for i, p in enumerate(PLANTS):
        y0 = top + (i + 1) * row_h - args.ground_from_bottom
        # Name.
        svg.layer(L["text"])
        svg.text(args.name_x, y0 - 6.5, p["name"], size=2.1, fill=INK, spacing=0.2)
        svg.text(args.name_x, y0 - 3.2, p["latin"].upper(), size=1.5, fill=INK, spacing=0.15, opacity=0.75)
        # The plant, scaled to the row.
        kind, form, h_m = p["draw"]
        h_mm = row_h * (0.78 if kind == "tree" else 0.6)
        if kind == "tree":
            trees.draw(svg, dict(wood=L["wood"], twigs=L["twigs"], leaves=L["leaves"]), form, args.plant_x, y0,
                       h_mm, h_mm * 0.75, rng)
        else:
            flora.plant(svg, dict(stems=L["stems"], leaves=L["leaves"]), form, args.plant_x, y0, h_mm, rng)
        svg.layer(L["ground"])
        svg.line(args.plant_x - 12, y0, x_b, y0, stroke=INK, stroke_width=0.15)

        labels = []
        for j, (how, lo, hi, per, est) in enumerate(p["reach"]):
            op = 0.55 if est else 1.0
            style = KIND[how]
            xs = lambda m: [X(d) for d in m]
            if style == "under":
                k = 120
                ds = np.logspace(math.log10(max(lo, args.dmin)), math.log10(hi), k)
                xx = np.linspace(X(args.dmin), X(hi), k)
                yy = y0 + 2.2 + 0.5 * np.sin(np.linspace(0, 26, k)) + 0.8 * j
                svg.layer(L["under"])
                d_ = "M" + "L".join(f"{a:.2f},{b:.2f}" for a, b in zip(xx, yy))
                svg.path(d_, fill="none", stroke=INK, stroke_width=0.15, opacity=op)
                # Shoots rising from it, out to the reach.
                for x in np.linspace(X(args.dmin) + 4, X(hi), 9):
                    svg.line(x, y0 + 1.6, x, y0 - 2.4, stroke=INK, stroke_width=0.12, opacity=op)
            elif style == "arch":
                svg.layer(L["arch"])
                for d in np.linspace(lo, hi, 4):
                    xe = X(d)
                    xs0 = X(args.dmin)
                    hh = min((xe - xs0) * 0.35, row_h * 0.3)
                    svg.path(f"M{xs0:.2f},{y0:.2f}C{xs0 + (xe - xs0) * 0.15:.2f},{y0 - hh * 1.3:.2f} "
                             f"{xe - (xe - xs0) * 0.2:.2f},{y0 - hh * 1.2:.2f} {xe:.2f},{y0:.2f}",
                             fill="none", stroke=INK, stroke_width=0.15, opacity=op)
                    svg.circle(xe, y0, 0.5, fill="#ffffff", stroke=INK, stroke_width=0.12, opacity=op)
                    for a in (-0.5, 0, 0.5):
                        svg.line(xe, y0, xe + math.sin(a) * 2.2, y0 + math.cos(a) * 2.2, stroke=INK,
                                 stroke_width=0.06, opacity=op)
            elif style == "flight":
                svg.layer(L["flight"])
                land = 10 ** rng.uniform(math.log10(max(lo, args.dmin)), math.log10(hi), args.flights)
                xs0 = X(args.dmin) + 2
                for d in land:
                    xe = X(d)
                    hh = min(row_h * 0.55, 3 + (xe - xs0) * 0.18)
                    svg.path(f"M{xs0:.2f},{y0 - 3:.2f}Q{(xs0 + xe) / 2:.2f},{y0 - hh:.2f} {xe:.2f},{y0:.2f}",
                             fill="none", stroke=INK, stroke_width=0.05, opacity=0.5 * op)
                svg.layer(L["ground_seed"])
                svg.path("".join(f"M{X(d) + rng.normal(0, 0.4):.2f},{y0 - 0.3:.2f}h0" for d in land), fill="none",
                         stroke=INK, stroke_width=0.45, stroke_linecap="round", opacity=op)
            elif style == "wind":
                # A plume: more seed near the parent, thinning far out; higher in the air mid-flight.
                m = args.wind_seeds
                u = rng.uniform(0, 1, m) ** 1.8
                ld = math.log10(max(lo, args.dmin)) + u * (math.log10(hi) - math.log10(max(lo, args.dmin)))
                xx = np.array([X(10 ** v) for v in ld])
                frac = (xx - X(args.dmin)) / max(X(hi) - X(args.dmin), 1)
                air = rng.random(m) < 0.6
                yy = np.where(air, y0 - (row_h * 0.6) * np.sin(np.pi * np.clip(frac + rng.normal(0, 0.08, m), 0, 1))
                              * rng.uniform(0.2, 1.0, m), y0 - 0.3)
                svg.layer(L["wind"])
                svg.path("".join(f"M{a:.2f},{b:.2f}h0" for a, b, s in zip(xx, yy, air) if s), fill="none", stroke=INK,
                         stroke_width=0.3, stroke_linecap="round", opacity=0.85 * op)
                svg.layer(L["ground_seed"])
                svg.path("".join(f"M{a:.2f},{b:.2f}h0" for a, b, s in zip(xx, yy, air) if not s), fill="none",
                         stroke=INK, stroke_width=0.35, stroke_linecap="round", opacity=op)
            elif style == "drop":
                m = 40
                ds = 10 ** rng.uniform(math.log10(max(lo, args.dmin)), math.log10(max(hi, args.dmin * 1.01)), m)
                svg.layer(L["ground_seed"])
                svg.path("".join(f"M{X(d):.2f},{y0 - 0.35:.2f}h0" for d in ds), fill="none", stroke=INK,
                         stroke_width=0.55 if p["key"] == "oak" else 0.3, stroke_linecap="round", opacity=op)
            elif style == "ants":
                svg.layer(L["ants"])
                for _ in range(10):
                    d = 10 ** rng.uniform(math.log10(lo), math.log10(hi))
                    xs0, xe = X(args.dmin) + 1, X(d)
                    pts = np.linspace(xs0, xe, 14)
                    yy = y0 - 0.5 - np.abs(np.sin(np.linspace(0, rng.uniform(4, 9), 14))) * 1.2
                    svg.path("M" + "L".join(f"{a:.2f},{b:.2f}" for a, b in zip(pts, yy)), fill="none", stroke=INK,
                             stroke_width=0.06, stroke_dasharray="0.25 0.35", opacity=op)
                    svg.circle(xe, y0 - 0.5, 0.35, fill=INK, stroke="none", opacity=op)
            # Bracket under the ground line.
            yb = y0 + 5.5 + 3.4 * j
            svg.layer(L["bracket"])
            a_, b_ = X(lo), X(hi)
            svg.line(a_, yb, b_, yb, stroke=INK, stroke_width=0.08, opacity=op)
            for xq in (a_, b_):
                svg.line(xq, yb - 0.7, xq, yb + 0.7, stroke=INK, stroke_width=0.08, opacity=op)
            rng_txt = (f"{lo:g}–{hi:g} M" if hi < 1000 else f"{lo:g} M – {hi / 1000:g} KM")
            labels.append((b_, yb, f"{how.upper()} · {rng_txt} {per.upper()}{' · EST.' if est else ''}", op))
        svg.layer(L["text"])
        for (xq, yb, t, op), (how, lo, hi, per, est) in zip(labels, p["reach"]):
            if xq + len(t) * 1.05 > Wp - 12:              # no room at the right: label at the bracket's left end
                svg.text(X(lo) - 1.6, yb + 0.6, t, size=1.5, fill=INK, anchor="end", spacing=0.12, opacity=max(op, 0.75))
            else:
                svg.text(xq + 1.6, yb + 0.6, t, size=1.5, fill=INK, spacing=0.12, opacity=max(op, 0.75))
        annot(args.name_x, y0 + 4, p["notes"], size=1.4, fill=INK, spacing=0.1)

    annot(20, 20, "HOW FAR, HOW FAST: HOW THE RIVERBANK PLANTS SPREAD", size=5.2, fill=INK, spacing=0.4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmin", type=float, default=0.1, help="m at the left of the scale (the parent)")
    ap.add_argument("--dmax", type=float, default=3000.0)
    ap.add_argument("--axis", type=float, nargs=2, default=[190.0, 840.0], help="mm, left and right of the scale")
    ap.add_argument("--top", type=float, default=60.0)
    ap.add_argument("--bottom", type=float, default=560.0)
    ap.add_argument("--ground-from-bottom", type=float, default=22.0, help="mm from the row's bottom to its ground")
    ap.add_argument("--name-x", type=float, default=40.0)
    ap.add_argument("--plant-x", type=float, default=135.0)
    ap.add_argument("--flights", type=int, default=24)
    ap.add_argument("--wind-seeds", type=int, default=260)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--out", default="outputs/sheets/09b_propagation_reach.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
