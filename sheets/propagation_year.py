"""Sheet 09c - The propagation year: when each riparian plant flowers, sets
seed, sends it away, waits through the cold and sprouts, set against the
river's and the air's year at Lockport.

A wheel, January at the top, running clockwise:

  - centre: mean daily air temperature at Winnipeg airport, 2002-2008
    (ECCC 5023222); the dashed circle is 0 °C, and days below it are hatched
  - next: the river's median daily flow below the dam, 1962-2008 (gauge
    05OJ010), one radial tick per day; the dotted arc inside it marks the
    days the river is ice-covered in most years (the gauge's B flags)
  - outer rings, one per plant: flowering (fine line with open flowers),
    seed or fruit ripening (heavy line), seed leaving the plant (dots),
    dormant seed waiting out the cold (dashed), germination (sprouts), and
    vegetative spread (wavy line)

The river and air are measured. The plants' calendar is typical for southern
Manitoba from the species literature, rounded to the half month: indicative,
not observed at Lockport. Data: drawing/propagation.py.

    python -m sheets.propagation_year
"""

import argparse
import datetime as dt
import glob
import math
import os

import numpy as np
import pandas as pd

from drawing import hydrometric
from drawing.propagation import PLANTS
from drawing.svg import PAPER, Svg

INK = "#111111"
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def build(args):
    rng = np.random.default_rng(args.seed)
    Wp, Hp = PAPER[args.paper]
    svg = Svg(Wp, Hp)
    annot = svg.text if args.annotate else (lambda *a_, **k_: None)
    L = dict(spokes="MONTHS · SPOKES, DOTTED · 0.04 mm", temp="AIR · MEAN DAILY TEMPERATURE 2002–08 · 0.15 mm",
             frost="AIR · BELOW 0 °C, HATCH · 0.04 mm", flow="RIVER · MEDIAN DAILY FLOW 1962–2008, TICKS · 0.1 mm",
             ice="RIVER · ICE-COVERED IN MOST YEARS, DOTTED", base="PLANTS · RING BASELINES · 0.03 mm",
             flower="PLANTS · FLOWERING · 0.12 mm", ripen="PLANTS · RIPENING · 0.4 mm",
             disperse="PLANTS · SEED LEAVING", cold="PLANTS · WAITING OUT THE COLD, DASHED · 0.15 mm",
             germ="PLANTS · GERMINATION · 0.08 mm", spread="PLANTS · VEGETATIVE SPREAD, WAVY · 0.12 mm",
             text="TEXT")
    for v in L.values():
        svg.layer(v)
    cx, cy = Wp / 2, Hp / 2

    def P(t, r):
        """Month t (0-12, fractional) and radius r (mm) to page x, y."""
        a = t / 12 * 2 * math.pi
        return cx + r * math.sin(a), cy - r * math.cos(a)

    def arc_pts(t0, t1, r_fn, step=0.02):
        if t1 < t0:
            t1 += 12
        ts = np.arange(t0, t1 + 1e-9, step)
        return [P(t % 12, r_fn(t)) for t in ts]

    def path(pts):
        return "M" + "L".join(f"{x:.2f},{y:.2f}" for x, y in pts)

    def ranges(t0, t1):
        return [(t0, t1)] if t0 <= t1 else [(t0, 12.0), (0.0, t1)]

    def month_of(doy):
        d = dt.date(2001, 1, 1) + dt.timedelta(days=int(doy) - 1)
        return d.month - 1 + (d.day - 1) / 30.5

    # ------------------------------------------------------------ air temperature
    frames = [pd.read_csv(f, encoding="utf-8-sig") for f in sorted(glob.glob(args.climate))]
    cl = pd.concat(frames)
    cl["date"] = pd.to_datetime(cl["Date/Time"])
    cl = cl.dropna(subset=["Mean Temp (°C)"])
    cl["doy"] = cl["date"].dt.dayofyear.clip(upper=365)
    temp = cl.groupby("doy")["Mean Temp (°C)"].mean().reindex(range(1, 366)).interpolate().bfill().ffill()
    temp = pd.Series(np.convolve(np.r_[temp.values[-7:], temp.values, temp.values[:7]], np.ones(15) / 15, "valid"),
                     index=range(1, 366))
    r0, kT = args.temp_r0, args.temp_k
    svg.layer(L["temp"])
    pts = [P(month_of(d), r0 + kT * temp[d]) for d in range(1, 366)]
    svg.path(path(pts) + "Z", fill="none", stroke=INK, stroke_width=0.15)
    zero = [P(t, r0) for t in np.arange(0, 12.001, 0.02)]
    svg.path(path(zero) + "Z", fill="none", stroke=INK, stroke_width=0.08, stroke_dasharray="0.6 0.6", opacity=0.8)
    svg.layer(L["frost"])
    d_ = []
    for d in range(1, 366, 2):
        if temp[d] < 0:
            a, b = P(month_of(d), r0), P(month_of(d), r0 + kT * temp[d])
            d_.append(f"M{a[0]:.2f},{a[1]:.2f}L{b[0]:.2f},{b[1]:.2f}")
    svg.path("".join(d_), fill="none", stroke=INK, stroke_width=0.04, opacity=0.7)
    svg.layer(L["text"])
    for T in (-20, 0, 20):
        x, y = P(0, r0 + kT * T)
        svg.text(x + 0.8, y + 0.6, f"{T:+d} °C" if T else "0 °C", size=1.4, fill=INK, opacity=0.8)

    # ------------------------------------------------------------ river
    dates, q, syms = hydrometric.load_wsc_daily(args.flow_csv, "discharge")
    df = pd.DataFrame(dict(date=pd.to_datetime(dates), q=q, ice=[s == "B" for s in syms]))
    df["doy"] = df["date"].dt.dayofyear.clip(upper=365)
    med = df.groupby("doy")["q"].median().reindex(range(1, 366)).interpolate()
    ice = df.groupby("doy")["ice"].mean().reindex(range(1, 366)).fillna(0)
    rq0, kq = args.flow_r0, args.flow_len / med.max()
    svg.layer(L["flow"])
    d_ = []
    for d in range(1, 366):
        a, b = P(month_of(d), rq0), P(month_of(d), rq0 + kq * med[d])
        d_.append(f"M{a[0]:.2f},{a[1]:.2f}L{b[0]:.2f},{b[1]:.2f}")
    svg.path("".join(d_), fill="none", stroke=INK, stroke_width=0.1)
    svg.layer(L["ice"])
    d_ = []
    for d in range(1, 366):
        if ice[d] > 0.5:
            x, y = P(month_of(d), rq0 - 3)
            d_.append(f"M{x:.2f},{y:.2f}h0")
    svg.path("".join(d_), fill="none", stroke=INK, stroke_width=0.5, stroke_linecap="round")
    peak = int(med.idxmax())
    svg.layer(L["text"])
    x, y = P(month_of(peak), rq0 + kq * med[peak] + 2.5)
    svg.text(x, y, f"{med[peak]:,.0f} m³/s", size=1.5, fill=INK, anchor="middle", opacity=0.85)
    print(f"median flow peak {med[peak]:.0f} m3/s on day {peak}; ice-covered in most years: {int((ice > 0.5).sum())} days")

    # ------------------------------------------------------------ plants
    n = len(PLANTS)
    r_in, r_out = args.plants_r
    w = (r_out - r_in) / n
    for i, p in enumerate(PLANTS):
        rc = r_in + (i + 0.5) * w
        svg.layer(L["base"])
        svg.path(path([P(t, rc) for t in np.arange(0, 12.001, 0.02)]) + "Z", fill="none", stroke=INK,
                 stroke_width=0.03, opacity=0.5)
        wind = any(m[0] == "wind" for m in p["reach"])
        for stage, t0, t1 in p["year"]:
            for a, b in ranges(t0, t1):
                if stage == "flower":
                    svg.layer(L["flower"])
                    svg.path(path(arc_pts(a, b, lambda t: rc - w * 0.3)), fill="none", stroke=INK, stroke_width=0.12)
                    for t in np.arange(a, b, 0.12):
                        x, y = P(t, rc - w * 0.3)
                        svg.circle(x, y, 0.45, fill="#ffffff", stroke=INK, stroke_width=0.08)
                elif stage == "ripen":
                    svg.layer(L["ripen"])
                    svg.path(path(arc_pts(a, b, lambda t: rc - w * 0.08)), fill="none", stroke=INK, stroke_width=0.4,
                             stroke_linecap="round")
                elif stage == "disperse":
                    svg.layer(L["disperse"])
                    m = int((b - a) * args.dots_per_month)
                    ts = rng.uniform(a, b, m)
                    # Wind-borne seed drifts outward across the ring; carried seed stays near it.
                    rr = rc + (rng.uniform(-0.05, 0.45, m) if wind else rng.normal(0.12, 0.08, m)) * w
                    svg.path("".join(f"M{P(t, r)[0]:.2f},{P(t, r)[1]:.2f}h0" for t, r in zip(ts, rr)), fill="none",
                             stroke=INK, stroke_width=0.32, stroke_linecap="round", opacity=0.85)
                elif stage == "cold":
                    svg.layer(L["cold"])
                    svg.path(path(arc_pts(a, b, lambda t: rc + w * 0.18)), fill="none", stroke=INK, stroke_width=0.15,
                             stroke_dasharray="1.0 0.8")
                    if p["key"] == "trillium":                     # two winters: a second dashed line
                        svg.path(path(arc_pts(a, b, lambda t: rc + w * 0.26)), fill="none", stroke=INK,
                                 stroke_width=0.15, stroke_dasharray="1.0 0.8")
                elif stage == "germinate":
                    svg.layer(L["germ"])
                    for t in np.arange(a + 0.05, b, 0.15):
                        x0, y0 = P(t, rc + w * 0.32)
                        x1, y1 = P(t, rc + w * 0.47)
                        svg.line(x0, y0, x1, y1, stroke=INK, stroke_width=0.08)
                        xl, yl = P(t - 0.03, rc + w * 0.44)
                        xr, yr = P(t + 0.03, rc + w * 0.44)
                        svg.path(f"M{xl:.2f},{yl:.2f}L{x1:.2f},{y1:.2f}L{xr:.2f},{yr:.2f}", fill="none", stroke=INK,
                                 stroke_width=0.08)
                elif stage == "spread":
                    svg.layer(L["spread"])
                    svg.path(path(arc_pts(a, b, lambda t: rc + w * (0.04 + 0.07 * math.sin(t * 40)), step=0.005)),
                             fill="none", stroke=INK, stroke_width=0.12)
        svg.layer(L["text"])
        x, y = P(0, rc)
        svg.text(x - 2.0, y + 0.6, p["name"], size=1.7, fill=INK, anchor="end", spacing=0.15)

    # ------------------------------------------------------------ months
    svg.layer(L["spokes"])
    for k in range(12):
        a, b = P(k, args.temp_r0 + kT * -28), P(k, r_out + 2)
        svg.line(a[0], a[1], b[0], b[1], stroke=INK, stroke_width=0.04, stroke_dasharray="0.2 0.8", opacity=0.7)
    svg.layer(L["text"])
    for k, mname in enumerate(MONTHS):
        x, y = P(k + 0.5, r_out + 7)
        svg.text(x, y + 0.8, mname, size=2.0, fill=INK, anchor="middle", spacing=0.25)

    annot(20, 20, "THE PROPAGATION YEAR", size=5.2, fill=INK, spacing=0.4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    svg.save(args.out)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow-csv", default="data/lockport/daily_05OJ010.csv")
    ap.add_argument("--climate", default="data/lockport/climate/en_climate_daily_MB_5023222_*.csv")
    ap.add_argument("--temp-r0", type=float, default=70.0, help="mm: radius of 0 °C")
    ap.add_argument("--temp-k", type=float, default=1.3, help="mm per °C")
    ap.add_argument("--flow-r0", type=float, default=112.0, help="mm: radius of zero flow")
    ap.add_argument("--flow-len", type=float, default=38.0, help="mm: tick length at the median peak")
    ap.add_argument("--plants-r", type=float, nargs=2, default=[158.0, 272.0], help="mm: inner and outer plant rings")
    ap.add_argument("--dots-per-month", type=float, default=90)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--paper", choices=list(PAPER), default="ARCHD")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--out", default="outputs/sheets/09c_propagation_year.svg")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
