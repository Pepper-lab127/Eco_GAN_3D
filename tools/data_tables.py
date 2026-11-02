"""Typeset some of the project's source data as tables (SVG, for PNG export):
one column per thing recorded, left-justified, ink on transparent.

  - wells: the two water-well logs near the dam (Groundwater Information
    Network), depth to the top of each layer
  - river: the 1997 crest weeks from the two Lockport gauges (Water Survey
    of Canada 05OJ010 flow below the dam, 05OJ021 level above it), each with
    its own data flags as published (A partial day, B ice or backwater,
    E estimated)
  - plants: the Lockport plant list (data/lockport/ecology/lockport_flora.csv)

    python -m tools.data_tables [--out outputs/tables]
"""

import argparse
import csv
import datetime as dt
import os

from drawing import hydrometric
from drawing.geology import LAYERS, LOCKPORT_WELLS, NAMES
from drawing.svg import Svg

INK = "#111111"
FT = 0.3048


def wells_table():
    head = ["WELL", "LAYER", "TOP, M BELOW GROUND", "TOP, FT", "THICKNESS, M"]
    rows = []
    for name, w in LOCKPORT_WELLS.items():
        tops = w["tops"]
        for i, layer in enumerate(LAYERS):
            below = tops[LAYERS[i + 1]] if i + 1 < len(LAYERS) else w["bottom"]
            thick = below - tops[layer]
            rows.append([f"{name.upper()} BANK", NAMES[layer],
                         f"{tops[layer]:.2f}", f"{tops[layer] / FT:.0f}",
                         f"{thick:.2f}" + (" (ABSENT)" if thick == 0 else "")])
        rows.append([f"{name.upper()} BANK", "BOTTOM OF HOLE", f"{w['bottom']:.2f}", f"{w['bottom'] / FT:.0f}", ""])
    return ("WATER-WELL LOGS NEAR ST. ANDREWS LOCK AND DAM · GROUNDWATER INFORMATION NETWORK", head, rows)


def river_table(start, end):
    dq, q, sq = hydrometric.load_wsc_daily("data/lockport/daily_05OJ010.csv", "discharge")
    dh, h, sh = hydrometric.load_wsc_daily("data/lockport/daily.csv", "level")
    flow = {d: (v, s) for d, v, s in zip(dq, q, sq)}
    level = {d: (v, s) for d, v, s in zip(dh, h, sh)}
    head = ["DATE", "FLOW BELOW THE DAM, M³/S", "FLAG", "LEVEL ABOVE THE DAM, M", "FLAG"]
    rows = []
    d = start
    while d <= end:
        fv, fs = flow.get(d, (None, ""))
        lv, ls = level.get(d, (None, ""))
        rows.append([d.strftime("%b %d %Y").upper(), f"{fv:,.0f}" if fv is not None else "—", fs,
                     f"{lv:.3f}" if lv is not None else "—", ls])
        d += dt.timedelta(days=1)
    return (f"RED RIVER AT LOCKPORT, DAILY, {start:%b %d} – {end:%b %d %Y} · WATER SURVEY OF CANADA 05OJ010 AND 05OJ021"
            .upper(),
            head, rows)


def plants_table(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = list(csv.reader(f))
    head = [c.upper() for c in r[0]]
    rows = [[row[0], row[1].upper(), row[2].upper(), row[3].upper(), row[4].upper()] for row in r[1:]]
    return ("PLANTS IN AND AROUND LOCKPORT · PROJECT PLANT LIST", head, rows)


def draw(svg, x, y, table, size=2.2, gap=6.0, lead=1.75, italic_col=None):
    """Draw a table with its top-left at (x, y); returns its (width, height) in mm."""
    title, head, rows = table
    cw = size * 0.6 + 0.12                                # mm per character, with letter-spacing
    widths = [max(len(str(c)) for c in col) * cw for col in zip(head, *rows)]
    xs = [x]
    for wdt in widths[:-1]:
        xs.append(xs[-1] + wdt + gap)
    total = xs[-1] + widths[-1] - x
    line = size * lead
    svg.text(x, y, title, size=size * 0.9, fill=INK, spacing=0.25)
    yy = y + line * 1.6
    for cx, h in zip(xs, head):
        svg.text(cx, yy, h, size=size * 0.8, fill=INK, spacing=0.2, opacity=0.8)
    svg.line(x, yy + line * 0.45, x + total, yy + line * 0.45, stroke=INK, stroke_width=0.15)
    yy += line * 1.35
    prev = None
    for row in rows:
        if prev is not None and row[0] != prev and table[0].startswith("WATER-WELL"):
            svg.line(x, yy - line * 0.7, x + total, yy - line * 0.7, stroke=INK, stroke_width=0.06, opacity=0.6)
            yy += line * 0.3
        for k, (cx, c) in enumerate(zip(xs, row)):
            kw = dict(font_style="italic") if k == italic_col else {}
            svg.text(cx, yy, str(c), size=size, fill=INK, spacing=0.12, **kw)
        prev = row[0]
        yy += line
    svg.line(x, yy - line * 0.55, x + total, yy - line * 0.55, stroke=INK, stroke_width=0.15)
    return max(total, len(title) * (size * 0.9 * 0.62 + 0.25)), yy - y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1997-04-20")
    ap.add_argument("--end", default="1997-05-20")
    ap.add_argument("--flora", default="data/lockport/ecology/lockport_flora.csv")
    ap.add_argument("--out", default="outputs/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tables = dict(
        wells=(wells_table(), None),
        river=(river_table(dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)), None),
        plants=(plants_table(a.flora), 0),
    )
    sizes = {}
    for name, (t, it) in tables.items():
        probe = Svg(2000, 2000)
        w, h = draw(probe, 0, 0, t, italic_col=it)
        sizes[name] = (w, h)
        svg = Svg(w + 20, h + 20, background=None)
        draw(svg, 10, 14, t, italic_col=it)
        svg.save(os.path.join(a.out, f"data_{name}.svg"))
    # All three together, stacked, left edges aligned.
    W = max(w for w, h in sizes.values()) + 20
    H = sum(h for w, h in sizes.values()) + 20 + 22 * (len(sizes) - 1)
    svg = Svg(W, H, background=None)
    y = 14
    for name, (t, it) in tables.items():
        draw(svg, 10, y, t, italic_col=it)
        y += sizes[name][1] + 22
    svg.save(os.path.join(a.out, "data_all.svg"))
    for name, (w, h) in sizes.items():
        print(f"{name}: {w:.0f} x {h:.0f} mm")
    print(f"all: {W:.0f} x {H:.0f} mm")


if __name__ == "__main__":
    main()
