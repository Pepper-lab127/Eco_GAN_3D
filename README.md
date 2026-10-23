# Eco_GAN_3D
generative adversarial network for ecological architecture

# 3D Ecological Architecture GAN

A TensorFlow 3D GAN for generating volumetric architectural/ecological forms.

The system generates 32x32x32 voxel structures and converts them into OBJ meshes.

## 1. Installation

Create a virtual environment:

python -m venv .venv

Linux/macOS:

source .venv/bin/activate

Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## 2. Generate training data

Run:

python generate_dataset.py

This creates:

data/procedural.npz

## 3. Train

Run:

python train.py

The generator checkpoints will appear in:

checkpoints/

Generated voxel fields will appear in:

outputs/

## 4. Generate new structures

For example:

python generate.py \
    --checkpoint checkpoints/generator_0150.weights.h5 \
    --count 20

## 5. Convert a generated structure to OBJ

Run:

python export_obj.py \
    --input outputs/generated_0000.npy \
    --output outputs/generated_0000.obj

The OBJ can then be opened in:

Blender
Rhino
Grasshopper
MeshLab
Maya
Houdini
etc.

## 6. Inspect the generated volume

The .npy file contains a 3D scalar field.

Values near:

0 = empty

1 = occupied

The default mesh threshold is:

0.5

## Site drawings (Lockport, MB)

Scaled, data-driven drawings of natural processes at Lockport on the Red
River, in the spirit of *Architecture of Nature / Nature of Architecture*.
Sheets are SVG at true paper size (mm) for finishing in Illustrator/Inkscape.

- `drawing/` – shared toolkit: SVG writer and annotation (scale bar, north
  arrow, flags), Water Survey of Canada loader, season-based time axis.
- `sheets/` – one script per sheet.
- `data/lockport/` – where downloaded data goes (see its README).

Install: `pip install -r requirements-drawing.txt`

Sheet 01, water motion (estimated velocity on a typical day and the 1997 crest):

python -m sheets.velocity_field

Sheets 04a and 04b, winter and riverbank sections (with key plans and well-log geology):

python -m sheets.winter_section
python -m sheets.riverbank_section

Sheet 02, calendar cut (1 m lidar terrain flooded with monthly gauge levels):

python -m sheets.calendar_cut

Sheet 05, the 1997 flood as particles (one dot per 100,000 m³ of daily flow,
gathered along each day's water's edge):

python -m sheets.flood_particles

The 1974 flood, the one year with the floodway's own measured flow (05OJ014);
the level above the dam is estimated from James Avenue (05OJ015):

python -m sheets.flood_particles --start 1974-04-01 --end 1974-06-30 --floodway-flow data/lockport/daily_05OJ014.csv

Both default to Arch D (36 x 24 in) at 1:4,000. A request for the 1997
floodway records is drafted in `docs/data_request_1997_floodway.md`.

Sheet 03, hydrograph surface:

python -m sheets.hydrograph_surface --csv data/lockport/daily.csv

Without `--csv` it renders clearly labelled placeholder data.
