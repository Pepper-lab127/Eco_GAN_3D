"""Daily river records: Water Survey of Canada CSV loader and a placeholder
generator for building sheets before real data is downloaded."""

import csv
import datetime as dt

import numpy as np


PARAMS = {"discharge": "1", "level": "2"}
UNITS = {"discharge": "m³/s", "level": "m"}


def load_wsc_daily(path, param="discharge"):
    """Read a daily CSV exported from wateroffice.ec.gc.ca.

    Accepts the standard long format (ID, PARAM, Date, Value, SYM) with any
    preamble lines, or a plain two-column Date,Value file.

    Returns (dates, values, symbols) as lists; symbol "B" marks ice/backwater.
    """
    code = PARAMS[param]
    dates, values, syms = [], [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    header_i = None
    for i, row in enumerate(rows):
        lowered = [c.strip().lower() for c in row]
        if "date" in lowered and "value" in lowered:
            header_i = i
            break
    if header_i is None:
        raise ValueError(f"No header with 'Date' and 'Value' columns in {path}")

    cols = [c.strip().lower() for c in rows[header_i]]
    i_date, i_val = cols.index("date"), cols.index("value")
    i_param = cols.index("param") if "param" in cols else None
    i_sym = cols.index("sym") if "sym" in cols else None

    for row in rows[header_i + 1:]:
        if len(row) <= max(i_date, i_val):
            continue
        if i_param is not None and row[i_param].strip() not in (code, param):
            continue
        try:
            d = dt.date.fromisoformat(row[i_date].strip().replace("/", "-"))
        except ValueError:
            continue
        v = row[i_val].strip()
        dates.append(d)
        values.append(float(v) if v else float("nan"))
        syms.append(row[i_sym].strip() if i_sym is not None and len(row) > i_sym else "")

    if not dates:
        raise ValueError(f"No {param} rows found in {path}")
    return dates, values, syms


def placeholder_daily(first_year=1962, last_year=2025, seed=7):
    """Synthetic Red River-like discharge: low ice-covered winters, a spring
    break-up / snowmelt flood of very variable size, summer storm pulses.

    Used only until real data is supplied; sheets label it as placeholder.
    """
    rng = np.random.default_rng(seed)
    dates, values, syms = [], [], []
    for year in range(first_year, last_year + 1):
        start = dt.date(year, 1, 1)
        n = (dt.date(year + 1, 1, 1) - start).days
        doy = np.arange(n)

        base = 90 + 60 * rng.random()
        peak = float(np.exp(rng.normal(np.log(1400), 0.55)))
        peak_day = int(rng.normal(112, 9))  # late April
        rise = 9 + 6 * rng.random()
        fall = 22 + 25 * rng.random()
        t = doy - peak_day
        flood = np.where(t < 0, np.exp(-(t / rise) ** 2), np.exp(-t / fall))
        q = base + peak * flood

        # Occasional second crest (ice jam release / late snow).
        if rng.random() < 0.35:
            d2 = peak_day + int(rng.integers(12, 30))
            q += 0.35 * peak * np.exp(-((doy - d2) / 7.0) ** 2)

        # Summer storms.
        for _ in range(rng.integers(1, 5)):
            ds = int(rng.integers(150, 290))
            q += rng.uniform(80, 500) * np.exp(-np.clip(doy - ds, 0, None) / 10.0) * (doy >= ds)

        q *= np.exp(rng.normal(0, 0.03, n))
        ice = (doy < peak_day - 18) | (doy > 318)

        for k in range(n):
            dates.append(start + dt.timedelta(days=int(k)))
            values.append(float(q[k]))
            syms.append("B" if ice[k] else "")
    return dates, values, syms


def water_year_matrix(dates, values, start_month=11, start_day=1):
    """Arrange a daily series into rows of 365 days, one per water year.

    Water year Y runs from start_month/start_day of Y-1 to the day before in Y.
    Feb 29 is dropped so every row lines up by calendar day.
    Returns (years, array[n_years, 365]) with NaN for missing days.
    """
    by_date = dict(zip(dates, values))
    first = min(dates).year + (1 if (min(dates).month, min(dates).day) > (start_month, start_day) else 0)
    last = max(dates).year + (1 if (max(dates).month, max(dates).day) >= (start_month, start_day) else 0)
    years = list(range(first, last + 1))
    out = np.full((len(years), 365), np.nan)
    for r, y in enumerate(years):
        d = dt.date(y - 1, start_month, start_day)
        k = 0
        while k < 365:
            if not (d.month == 2 and d.day == 29):
                v = by_date.get(d)
                if v is not None:
                    out[r, k] = v
                k += 1
            d += dt.timedelta(days=1)
    # Keep only years that are mostly complete.
    keep = np.mean(np.isnan(out), axis=1) < 0.3
    return [y for y, ok in zip(years, keep) if ok], out[keep]


def median_ice_dates(dates, syms):
    """Median spring break-up (last 'B' flag Jan-Jun) and autumn freeze-up
    (first 'B' flag Oct-Dec) as days of year, or None if no ice flags."""
    spring, autumn = {}, {}
    for d, s in zip(dates, syms):
        if s != "B":
            continue
        doy = d.timetuple().tm_yday
        if d.month <= 6:
            spring[d.year] = max(spring.get(d.year, 0), doy)
        elif d.month >= 10:
            autumn[d.year] = min(autumn.get(d.year, 999), doy)
    breakup = int(np.median(list(spring.values()))) if spring else None
    freezeup = int(np.median(list(autumn.values()))) if autumn else None
    return breakup, freezeup
