# Photo textures (optional)

The rendered sheets (04a scene, 04c) take surface grain from photographs placed
here, if present; otherwise they use procedural noise. Only the grain is used
(the photo is turned grey and high-passed), so colour and layer placement still
come from the data. Any missing file is skipped.

Suggested sources: Texture Labs (texturelabs.org, free), Poly Haven or
ambientCG (CC0). Save as JPG, at least 2000 px wide, with these names:

| file | what to look for | used in |
|---|---|---|
| `snow.jpg` | wind-packed snow surface, side light | 04c snowpack, 04a snow and bank faces |
| `water.jpg` | murky water, underwater haze or smoke | 04c silt clouds |
| `clay.jpg` | wet clay or mud, fine cracks | 04c riverbed, 04a lake clay |
| `till.jpg` | gravelly soil with stones | 04a till |
| `shale.jpg` | layered shale or slate | 04a shale |
| `limestone.jpg` | limestone or dolomite rock face | 04a carbonate aquifer |
