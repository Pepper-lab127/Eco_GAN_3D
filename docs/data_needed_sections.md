# Data for the two sections (sheets 04a and 04b)

Section line: DEM row 1840, about 300 m below St. Andrews Lock and Dam,
west to east across the Red River at the Lockport fishing grounds. The
winter section spans the valley; the riverbank section is a tall close-up
of the wooded east bank on the same line.

Already on file: 1 m lidar ground and canopy (DSM minus DTM) along the line,
and year-round water levels below the dam (05OJ010, 2002-2008).

Please download the following and upload them here (or add them to
`data/lockport/`). Each file can be uploaded as-is.

## 1. Daily air temperature, winters 2002-2008 (for ice thickness)

Ice thickens with accumulated cold (freezing degree-days), a standard
estimate, so daily temperatures give the ice on each winter day.

Environment and Climate Change Canada, Winnipeg Richardson Int'l A, daily
data, one CSV per year. Open each link in your browser; it downloads a CSV:

- https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=3698&Year=2002&Month=1&Day=1&timeframe=2&submit=Download+Data
- …same link with `Year=2003`, `2004`, `2005`, `2006`, `2007`, `2008`

If a file comes back empty, the station ID has changed: on
climate.weather.gc.ca go to Historical Data, search "Winnipeg Richardson",
choose **Daily**, and download each year from 2002 to 2008 as CSV.

## 2. Hourly wind, one recent year (for the sky and wind)

Same station, hourly data (wind speed and direction), one CSV per month.
Twelve links, January to December 2024:

- https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=51097&Year=2024&Month=1&Day=1&timeframe=1&submit=Download+Data
- …same link with `Month=2` through `Month=12`

(Same fallback: Historical Data → "Winnipeg Richardson" → **Hourly**.)

## 3. Well logs near Lockport (for lake clay and the limestone aquifer)

The depth to clay, till and limestone comes from drillers' logs. Either:

- **Groundwater Information Network** (NRCan): gin.gw-info.net → map viewer
  → zoom to Lockport → click wells on both banks near the dam → open each
  well's lithology log and export or screenshot it; or
- **Manitoba Groundwater well search** (GWDrill), same idea.

Three to six wells within about 2 km of the dam is plenty, ideally a couple
on each bank. Screenshots of the logs are fine: I need each well's location
and the depths at which clay, till/gravel and limestone begin, and the water
level if listed.

## 4. River depth at the section (water column)

The lidar stops at the water surface. Any of these gives the bed:

- added to the Manitoba request (`docs/data_request_1997_floodway.md`, item 5):
  surveyed river cross-sections near the dam from the province's hydraulic
  models;
- a Canadian Hydrographic Service chart covering Lockport, if one exists for
  this reach;
- failing both, the sheet will draw an estimated bed, clearly marked, using
  the known geology: the dam was built on the limestone ledges of the
  St. Andrews Rapids.

## 5. Optional: fish

Counts or timing of the spring fish run at the dam (walleye, channel catfish,
etc.), e.g. from fishway monitoring. Without them the fish layer is drawn as
a labelled diagram, not data.

## Drawn as diagrams, labelled as such

Tree species (cottonwood, elm, ash are typical of this bank; lidar gives
heights, not species), root spread and depth, and the fish run (unless item 5
arrives).
