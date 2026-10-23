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
