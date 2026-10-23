# Data request: Red River Floodway, spring 1997

**To:** Manitoba Transportation and Infrastructure — Hydrologic Forecast Centre /
Water Management and Structures (Red River Floodway operations)

Contact routes: the Hydrologic Forecast Centre contact listed on
manitoba.ca/floodinfo, or the Red River Floodway page
(gov.mb.ca/mti/wms/rrf). The province's geospatial contact
(manitobamaps@gov.mb.ca) can redirect if needed.

---

**Subject:** Request for 1997 Red River Floodway daily operation data (academic research)

Hello,

I am [NAME], [ROLE / PROGRAM] at [INSTITUTION], working on an architectural
research project on natural processes at St. Andrews Lock and Dam, Lockport.
The project turns open hydrometric and lidar data into scaled drawings of the
river at Lockport, including the 1997 flood.

For 1997 I have Water Survey of Canada daily data for the Red River above the
dam (05OJ021), near Lockport (05OJ010) and at James Avenue (05OJ015, level
only). The floodway gauge near Lockport (05OJ014) ends in 1976, so I cannot
place the floodway's water in 1997. I would be grateful for any of the
following, in any format (CSV, spreadsheet or scanned tables):

1. **Daily mean discharge in the Red River Floodway channel**, 1 April to 30 June
   1997 (measured or as computed by the province).
2. **Daily water levels in the floodway channel**, especially near the outlet
   structure at Lockport, for the same period.
3. **Daily discharge of the Red River at James Avenue** (or through Winnipeg) for
   the same period, if computed.
4. If they can be shared, the **independent review of the 1997 floodway
   operation** (Klohn-Crippen Consultants, for the Manitoba Water Commission),
   or a reference for obtaining it.
5. **Surveyed river cross-sections (bathymetry)** of the Red River near St.
   Andrews Lock and Dam, for example those used in the province's hydraulic
   models of the lower Red River.
6. If available, **fish passage or spring fish-run monitoring** at the dam
   (counts or timing by species).

The data will be used for non-commercial academic work and credited to the
Province of Manitoba in every drawing that uses it. If a formal data request
form or licence applies, please let me know and I will complete it.

Thank you for your time,

[NAME]
[INSTITUTION]
[EMAIL] · [PHONE]

---

## When it arrives

- Daily floodway **flow** (CSV with a date column and a value column, or WSC
  format): save as `data/lockport/floodway_1997.csv` and run

  ```
  python -m sheets.flood_particles --floodway-flow data/lockport/floodway_1997.csv
  ```

  The floodway's share of each day's dots then goes along its bed and the
  timeline shows it as heavy ticks, as on the 1974 sheet.
- Floodway **levels** near the outlet would let the floodway be flooded to its
  real edge instead of its bed; that needs a small addition to the sheet.
- A James Avenue **flow** series would allow a cross-check by water balance
  (see `data/lockport/README.md`).
