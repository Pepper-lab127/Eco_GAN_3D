"""Sheet 03 - Hydrograph surface.

Every water year of the Red River record drawn as one line, stacked from
oldest (back) to newest (front) so the spring floods rise over the years
behind them. The horizontal axis is the river's own calendar: ice cover is
compressed, break-up and flood crest are stretched. The ground turns from
black to white at break-up.

    python -m sheets.hydrograph_surface                      # placeholder data
    python -m sheets.hydrograph_surface --csv data/lockport/daily.csv
"""

import argparse
import os

import numpy as np

from drawing import hydrometric, seasons
from drawing.svg import PAPER, Svg, flag, points_to_path


INK_DARK = "#0d0d0d"
INK_LIGHT = "#f4f4f2"


def build(args):
    if args.csv:
        if args.param is None:
            # Use discharge when the file has it, otherwise water level.
            try:
                hydrometric.load_wsc_daily(args.csv, "discharge")
                args.param = "discharge"
            except ValueError:
                args.param = "level"
        dates, values, syms = hydrometric.load_wsc_daily(args.csv, args.param)
        station = hydrometric.station_id(args.csv)
        source = (f"WATER SURVEY OF CANADA, STATION {station}" if station else os.path.basename(args.csv)) \
            + f", {dates[0].isoformat()} TO {dates[-1].isoformat()}"
        placeholder = False
    else:
        args.param = args.param or "discharge"
        if args.param != "discharge":
            raise SystemExit("Placeholder data is discharge only; pass --csv for water level.")
        dates, values, syms = hydrometric.placeholder_daily()
        source = "PLACEHOLDER — SYNTHETIC SERIES, NOT OBSERVED DATA"
        placeholder = True

    years, rows = hydrometric.water_year_matrix(dates, values, min_coverage=args.min_coverage)
    if not years:
        raise SystemExit(f"No water year has {args.min_coverage:.0%} of days recorded; lower --min-coverage.")
    unrecorded = np.isnan(rows).all(axis=0)
    breakup, freezeup = hydrometric.median_ice_dates(dates, syms)
    phases = seasons.phases_with_ice_dates(breakup, freezeup)
    units = hydrometric.UNITS[args.param]

    W, H = PAPER[args.paper]
    m = 14.0                                  # sheet margin
    px0, px1 = m + 46.0, W - m - 62.0         # plot x extent
    top, bottom = m + 70.0, H - m - 46.0      # stacked baselines
    amp = args.amplitude                      # mm for the record maximum

    axis = seasons.PhaseAxis(phases, px0, px1)
    flip_x = axis.phase_x("BREAK-UP")
    xs = axis.x(np.arange(365))

    n = len(years)
    if args.param == "level":
        floor = np.nanmin(rows)
        rows = rows - floor
    vmax = np.nanmax(rows)
    scale = amp / vmax

    # Push the stack down until no crest rises above the title block.
    ceiling = m + 50.0
    row_max = np.nan_to_num(np.nanmax(rows, axis=1)) * scale
    for _ in range(100):
        step = (bottom - top) / max(n - 1, 1)
        top = max(m + 70.0, ceiling + np.max(row_max - np.arange(n) * step))
    if bottom - top < 0.4 * (bottom - m - 70.0):
        print("note: crests are tall for this record; try a smaller --amplitude")
    baselines = top + np.arange(n) * step

    decimals = 2 if args.param == "level" else 0
    peaks = sorted(range(n), key=lambda i: -np.nanmax(rows[i]))[: args.flags]

    svg = Svg(W, H, background="#ffffff")
    frame_y0, frame_y1 = m, H - m

    # Layers by line weight, declared bottom to top so they stack in this order.
    L = dict(ground="GROUND (BLACK / WHITE)", phases="PHASE HAIRLINES · 0.06 mm",
             # One layer: each year's dotted gaps must stay under the years in front of it.
             years=f"YEAR LINES AND NO-RECORD DOTS · {args.line:g} mm",
             axis="AXIS AND SCALE · 0.15 mm", ticks="AXIS TICKS · 0.06–0.12 mm",
             flags="FLAGS (LEADERS AND LABELS) AND YEAR TICKS · 0.08 mm", frame="FRAME · 0.15 mm", text="TEXT")
    for name in L.values():
        svg.layer(name)

    def draw(ink, ground, clip):
        def lay(key):
            # Each layer gets its own copy of this half's clip.
            if lay.open:
                svg.close_group()
            svg.layer(L[key])
            svg.open_group(clip_path=clip)
            lay.open = True
        lay.open = False

        lay("ground")
        svg.rect(m, frame_y0, W - 2 * m, frame_y1 - frame_y0, fill=ground)

        # Phase hairlines through the full height, like the kelp life-cycle sheet.
        for name, xs0, _, _ in axis.boundaries():
            lay("phases")
            svg.line(xs0, frame_y0 + 52, xs0, bottom + 10, stroke=ink, stroke_width=0.06, opacity=0.55)

        # Stacked years, back to front; the fill hides lines behind each crest.
        lay("years")
        for i in range(n):
            ys = baselines[i] - rows[i] * scale
            d = points_to_path(xs, ys, close_to=baselines[i] + 0.01)
            svg.path(d, fill=ground, stroke=ink, stroke_width=args.line,
                     stroke_linejoin="round")
            # Days without a record run as a dotted line at the year's base.
            gap = np.where(np.isnan(rows[i]), baselines[i], np.nan)
            svg.path(points_to_path(xs, gap), fill="none", stroke=ink, stroke_width=args.line,
                     stroke_dasharray="0.15 0.9", stroke_linecap="round", opacity=0.8)

        # Year labels every decade.
        for i, y in enumerate(years):
            if y % 10 == 0 or i in (0, n - 1):
                lay("text")
                svg.text(px0 - 4, baselines[i] + 0.6, str(y), size=1.8, fill=ink, anchor="end")
                lay("flags")
                svg.line(px0 - 2.5, baselines[i], px0 - 0.5, baselines[i], stroke=ink, stroke_width=0.08)

        # Phase axis: weekly ticks reveal how the calendar is stretched.
        ay = bottom + 10
        lay("axis")
        svg.line(px0, ay, px1, ay, stroke=ink, stroke_width=0.15)
        lay("ticks")
        for k in range(0, 365, 7):
            svg.line(axis.x(k), ay, axis.x(k), ay + 1.2, stroke=ink, stroke_width=0.06)
        for name, xs0, xs1, d0 in axis.boundaries():
            lay("ticks")
            svg.line(xs0, ay, xs0, ay + 4, stroke=ink, stroke_width=0.12)
            lay("text")
            svg.text(xs0 + 1.0, ay + 4.2, name, size=1.9, fill=ink, spacing=0.25)
            svg.text(xs0 + 1.0, ay + 7.0, seasons.day_label(d0), size=1.6, fill=ink, opacity=0.7)
        lay("text")
        svg.text(px1, ay + 13, "1 TICK = 7 DAYS", size=1.6, fill=ink, anchor="end", opacity=0.7)

        # Flags on the largest crests (leader to the right margin).
        for rank, i in enumerate(peaks):
            k = int(np.nanargmax(rows[i]))
            x, y = xs[k], baselines[i] - rows[i][k] * scale
            peak_val = rows[i][k] + (floor if args.param == "level" else 0)
            label = f"{years[i]} · {peak_val:,.{decimals}f} {units} · {seasons.day_label(k)}"
            lay("flags")
            svg.circle(x, y, 0.45, fill=ink)
            flag(svg, x, y, label, dx=px1 - x + 6, color=ink, size=1.8)
            lay("text")

        # Vertical scale for the line height.
        unit_step = _nice(vmax / 3)
        sx, sy = px1 + 8, bottom
        h = unit_step * scale
        lay("axis")
        svg.line(sx, sy, sx, sy - h, stroke=ink, stroke_width=0.15)
        svg.line(sx - 1, sy, sx + 1, sy, stroke=ink, stroke_width=0.15)
        svg.line(sx - 1, sy - h, sx + 1, sy - h, stroke=ink, stroke_width=0.15)
        lay("text")
        svg.text(sx + 2, sy - h / 2, f"{unit_step:,.{decimals}f} {units}", size=1.8, fill=ink)

        # Title block.
        lay("text")
        tx, ty = m + 10, frame_y0 + 14
        svg.text(tx, ty, "RED RIVER AT LOCKPORT", size=5.2, fill=ink, spacing=0.4)
        svg.text(tx, ty + 7, "HYDROGRAPH SURFACE — "
                 + ("DAILY DISCHARGE" if args.param == "discharge" else "DAILY WATER LEVEL")
                 + f", WATER YEARS {years[0]}–{years[-1]}", size=2.4, fill=ink, spacing=0.3)
        svg.text(tx, ty + 12, "Each line is one water year (Nov 1 – Oct 31), oldest at the back. "
                 "Time is measured in river phases, not months: ice cover is compressed,", size=1.9, fill=ink)
        svg.text(tx, ty + 15, "break-up and flood crest are stretched. The ground turns at break-up.",
                 size=1.9, fill=ink)
        svg.text(tx, ty + 20, f"SOURCE: {source}", size=1.7, fill=ink,
                 weight="bold" if placeholder else None, opacity=1 if placeholder else 0.7)
        notes = []
        if args.param == "level":
            notes.append(f"Lines rise from the lowest recorded level, {floor:.2f} m above station datum.")
        if breakup:
            notes.append(f"BREAK-UP placed from median end of ice flags: {seasons.day_label(axis.day_edges[2])}.")
        else:
            notes.append("Phase dates are typical for the Red River, not derived from this record.")
        if unrecorded.any() or np.isnan(rows).mean() > 0.05:
            notes.append("Dotted: no record (the gauge reports open-water season only, or data is missing).")
        for j, note in enumerate(notes):
            svg.text(tx, ty + 23 + 3 * j, note, size=1.7, fill=ink, opacity=0.7)

        svg.close_group()

    draw(INK_LIGHT, INK_DARK, svg.clip_rect(0, 0, flip_x, H))
    draw(INK_DARK, "#ffffff", svg.clip_rect(flip_x, 0, W - flip_x, H))
    svg.layer(L["frame"])
    svg.rect(m, frame_y0, W - 2 * m, frame_y1 - frame_y0, fill="none", stroke=INK_DARK, stroke_width=0.15)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}  ({n} water years, {'placeholder' if placeholder else 'observed'})")


def _nice(v):
    p = 10 ** np.floor(np.log10(v))
    for f in (1, 2, 2.5, 5, 10):
        if f * p >= v:
            return f * p
    return 10 * p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="Water Survey of Canada daily CSV (omit for placeholder data)")
    ap.add_argument("--param", choices=["discharge", "level"],
                    help="default: discharge if the CSV has it, else level")
    ap.add_argument("--min-coverage", type=float, default=0.4,
                    help="fraction of days a water year needs to be drawn")
    ap.add_argument("--paper", choices=list(PAPER), default="A2")
    ap.add_argument("--amplitude", type=float, default=95.0, help="mm height of the record maximum")
    ap.add_argument("--line", type=float, default=0.12, help="line weight in mm")
    ap.add_argument("--flags", type=int, default=5, help="number of largest crests to label")
    ap.add_argument("--out", default="outputs/sheets/03_hydrograph_surface.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
