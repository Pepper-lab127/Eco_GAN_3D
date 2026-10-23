# Data still needed — Lockport drawings

What would most raise the drawings toward the reference sheets, by sheet.
Upload files here or add them to `data/lockport/`. Most important first.

## 1. Manitoba request (one email, many sheets)

Send `docs/data_request_1997_floodway.md` (fill in name and institution) to
Manitoba Transportation and Infrastructure, Hydrologic Forecast Centre. It asks for:

- daily floodway flow and levels, April–June 1997 → sheets 01, 05
- daily flow at James Avenue, 1997 → floodway cross-check
- the Klohn-Crippen review of the 1997 floodway operation
- **surveyed river cross-sections (bathymetry)** near the dam → replaces the
  chart-read bed in sheets 01, 04a, 04b, 05
- fish passage / spring fish-run counts at the dam → winter section fish layer

## 2. Downloads you can do yourself

| Data | Where | For |
|---|---|---|
| **Hourly water level and flow**, Red River below the dam (05OJ010) and above it (05OJ021), any years available | wateroffice.ec.gc.ca → Historical / Real-time data → station → *hourly* or *unit values* | daily rhythms of the dam, ice break-up hour by hour (sheets 02, 03, 05) |
| **More well logs**, 4–6 more within 2 km of the dam, both banks | gin.gw-info.net (Groundwater Information Network) map viewer, or Manitoba GWDrill | truer geology between the two logs (04a, 04b); water levels in the aquifer |
| **Groundwater levels**, an observation well in the carbonate aquifer near Lockport/Selkirk | Manitoba groundwater monitoring network (observation well hydrographs) | the aquifer as a moving water body, not just a texture (04a, 04b) |
| **Daily air temperature and snow on ground, 1962–2025** | climate.weather.gc.ca, Winnipeg Richardson, Daily, one CSV per year (you have 2002–2008) | ice for every winter of the flow record; snow load on the ice |
| **Ice break-up and freeze-up dates** at Selkirk/Lockport | Manitoba ice reports; Water Survey `B` flags (already in 05OJ010 to 2008) | the season axis and ice sheets |
| **Satellite images** of the dam on chosen days: open water, ice, the 1997 and 2011 floods | Sentinel-2 (2015+) and Landsat (1984+) via EarthExplorer or Copernicus Browser; download true-colour and the SWIR bands | real flood extents and ice cover to check and texture the plans |
| **Historic air photos** of the dam and rapids, 1920s–1990s | National Air Photo Library (NAPL), Natural Resources Canada | the St. Andrews Rapids before and after the dam; bank change |
| **Older lidar or a second survey** of the banks | `python tools/fetch_hrdem.py --keep-projects` (already possible here) | bank erosion 2014 → 2020 (cut-terrain block drawing) |
| **Suspended sediment / turbidity**, Red River at Selkirk or Lockport | Water Survey of Canada sediment data, or Manitoba water quality (Red River at Selkirk) | the colour and load of the water in plans and sections |
| **Water temperature**, Red River near Lockport | Manitoba water quality monitoring; ECCC long-term monitoring | water column layering in the winter section |
| **Fish species and timing** at Lockport (walleye, channel catfish, sturgeon) | Manitoba Fisheries; Fish Futures / angler surveys; DFO | fish run in the winter section |
| **Vegetation species** on the banks | Manitoba Conservation Data Centre; iNaturalist observations near Lockport (exportable CSV) | naming and drawing the riparian wood (04b) |
| **Full Navionics/CHS depth chart** upstream and downstream (screenshots of each panel, north up, same zoom) | Navionics web app; or Canadian Hydrographic Service if a chart exists | bed for the whole plan window, not just near the dam |

## 3. Already on file

Lidar ground and surface (1 m), gauges 05OJ010/014/015/021/024 and 05OC029,
daily climate 2002–2008, hourly wind 2024, two well logs, chart soundings
near the dam.
