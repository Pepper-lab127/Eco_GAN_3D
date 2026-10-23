"""Environment and Climate Change Canada daily climate files, and the ice
estimate built on them."""

import csv
import datetime as dt
import glob

import numpy as np


def load_eccc_daily(pattern):
    """Read ECCC daily climate CSVs (one per year) matching a glob pattern.

    Returns {date: {"tmean", "tmin", "tmax", "snow_ground", "gust_dir", "gust_spd"}}
    with None for blanks; also the station name.
    """
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    days, station = {}, None
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                d = dt.date.fromisoformat(row["Date/Time"])
                station = row.get("Station Name") or station
                days[d] = dict(
                    tmean=num(row.get("Mean Temp (°C)")),
                    tmin=num(row.get("Min Temp (°C)")),
                    tmax=num(row.get("Max Temp (°C)")),
                    snow_ground=num(row.get("Snow on Grnd (cm)")),
                    gust_dir=num(row.get("Dir of Max Gust (10s deg)")),
                    gust_spd=num(row.get("Spd of Max Gust (km/h)")),
                    snow=num(row.get("Total Snow (cm)")),
                    precip=num(row.get("Total Precip (mm)")),
                )
    return days, station


def freezing_degree_days(days, start, end):
    """Accumulated freezing degree-days from start to end: frost adds, thaw
    subtracts, never below zero. Missing days are filled from neighbours.
    Returns {date: AFDD in degC days}."""
    dates = [start + dt.timedelta(k) for k in range((end - start).days + 1)]
    known = [(d.toordinal(), days[d]["tmean"]) for d in dates if d in days and days[d]["tmean"] is not None]
    xs, ts = np.array(known).T
    afdd, out = 0.0, {}
    for d in dates:
        t = float(np.interp(d.toordinal(), xs, ts))
        afdd = max(0.0, afdd - t)
        out[d] = afdd
    return out


def stefan_ice(afdd, alpha=0.015):
    """Ice thickness in m from accumulated freezing degree-days (Stefan's law,
    h = alpha * sqrt(AFDD)). alpha ~0.014-0.017 m/(degC day)^0.5 for rivers
    under snow (Michel 1971)."""
    return alpha * np.sqrt(afdd)


def load_eccc_hourly_wind(pattern):
    """Read ECCC hourly climate CSVs: [(datetime, direction deg, speed km/h)]
    for hours with both values (direction 0 = calm is skipped); also the station."""
    out, station = [], None
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                station = row.get("Station Name") or station
                d, s = row.get("Wind Dir (10s deg)"), row.get("Wind Spd (km/h)")
                if not d or not s or float(d) == 0:
                    continue
                t = dt.datetime.fromisoformat(row["Date/Time (LST)"])
                out.append((t, float(d) * 10, float(s)))
    out.sort()
    return out, station
