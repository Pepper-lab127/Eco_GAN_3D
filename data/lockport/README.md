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

High Resolution DEM (HRDEM, 1-2 m lidar) from NRCan, via open.canada.ca or
the Geo.ca viewer. Area: roughly 50.06-50.11 N, 96.98-96.90 W. Download the
**DTM** GeoTIFF tile(s) and save them as `data/lockport/dem/*.tif`.
