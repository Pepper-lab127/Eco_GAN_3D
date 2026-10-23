# Lockport site data

Drop downloaded source data here. The sheets in `sheets/` read from this folder.

## River gauge record (sheet 03, later sheets 02 and 05)

1. Go to https://wateroffice.ec.gc.ca and open *Historical Hydrometric Data*.
2. Search for **Red River** stations in Manitoba and pick the one nearest
   Lockport (the station at the lock/dam or at Selkirk).
3. Download **daily** data as CSV, full period of record. Discharge and water
   level can be in one file; the loader picks the `PARAM` you ask for
   (1 = discharge, 2 = level).
4. Save as `data/lockport/daily.csv`, then:

```
python -m sheets.hydrograph_surface --csv data/lockport/daily.csv
python -m sheets.hydrograph_surface --csv data/lockport/daily.csv --param level
```

The `SYM` column's `B` flag (ice / backwater conditions) is used to place
the start of the break-up phase on the time axis.

### Gauges on file

- `daily.csv`: station **05OJ021**, daily water level 1996-2025, open-water
  season only. Matches Manitoba's "Red River Upstream of Dam at Lockport".
- `daily_05OJ024.csv`: station **05OJ024**, daily water level Aug 2022 - Oct
  2025. Also on the pool above the dam: on 669 shared days it reads within
  about 1 cm of 05OJ021 (median -0.012 m). Useful as a cross-check or to
  extend the record, not for the river below the dam (which the lidar shows
  about 2 m lower, ~217.6 m).
- `daily_05OJ010.csv`: station **05OJ010**, below the dam. Daily water
  level 2002-2008 (year-round, ~217-218 m normally, matching the lidar water
  below the dam) and daily discharge 1962-2008 (year-round, with `B` ice
  flags; peak 4,320 m³/s on 1997-05-04). Sheet 02 floods the reach below the
  dam from its levels; sheet 03 can be drawn from its flow:

  ```
  python -m sheets.hydrograph_surface --csv data/lockport/daily_05OJ010.csv
  ```
- `daily_05OJ014.csv`: station **05OJ014**, Red River Floodway near Lockport.
  Daily flow only, 1969-1976 (peak 1,130 m³/s), so it cannot place the
  floodway's water in 1997. Kept for a possible 1970s flood sheet.

## Elevation (sheets 01, 02, 04, 05)

`dem/lockport_dtm_1m.tif` (bare ground) and `dem/lockport_dsm_1m.tif` (with
trees and buildings) are 1 m lidar from NRCan's HRDEM, cut to a 3.5 x 4 km
window around the lock and floodway outlet (EPSG:3979, heights in metres,
CGVD2013, rounded to the centimetre).

They merge two surveys: **MB-Red_River_2020** (river corridor) with gaps filled
from **MB-Cooks_Creek** (2014). Water surfaces are flat at the level on the day
of each flight, so the riverbed is not captured, and there is a visible step
in the river where the two surveys meet.

To rebuild them, or cut a different window, without downloading whole tiles:

```
pip install rasterio
python tools/fetch_hrdem.py                  # Lockport defaults
python tools/fetch_hrdem.py --keep-projects  # also each survey on its own
```
