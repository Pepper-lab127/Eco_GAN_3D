# Eco_GAN_3D
generative adversarial network for ecological architecture

# 3D Ecological Architecture GAN

A TensorFlow 3D GAN for generating volumetric architectural/ecological forms.

The system learns from procedurally generated morphologies (branching,
cavitation, porosity, erosion, layering, clustering, fragmentation and
hybrids of these). Each sample is stored as a smooth signed distance field
and measured for ecological performance: how much water its pockets hold,
how much sheltered crevice surface it offers, and so on.

The generator is conditional: you ask for a combination of phenomena and
target values for those ecological metrics, and it produces a matching
structure, which can be exported as an OBJ mesh.

Default resolution is 32x32x32 (64x64x64 is supported).

## 1. Installation

Create a virtual environment:

python -m venv .venv

Linux/macOS:

source .venv/bin/activate

Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

### NVIDIA GPU (e.g. GTX 1060)

TensorFlow's GPU support depends on the operating system:

- Linux, or Windows through WSL2 (recommended):
  `pip install "tensorflow[and-cuda]"` with a current NVIDIA driver.
- Windows without WSL2: TensorFlow 2.10 is the last release with native
  GPU support. It needs Python 3.10, CUDA 11.2 and cuDNN 8.1:
  `pip install "tensorflow<2.11"`. The code is written to run on it.

Check the GPU is visible:

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

On a 6 GB GTX 1060 at 32^3, batch size 16 fits comfortably. Mixed precision
is deliberately not used: Pascal cards have no tensor cores and are slower
in float16.

## 2. Generate training data

Run:

python generate_dataset.py

Each sample is produced by simulating an ecological process rather than
drawing a shape that resembles one:

| Phenomenon | Simulation |
|---|---|
| branching | space colonisation growth; branch thickness from the pipe model |
| clustering | colonies accreting toward light, competing and shading each other |
| layering | deposited strata of varying hardness, weathered into ledges and recesses |
| erosion | rain (gully incision, drip lines), wind scour, or salt weathering (tafoni) |
| cavitation | karst dissolution by water percolating down, pooling on insoluble beds |
| porosity | Gray-Scott reaction-diffusion pores (vesicular, spongy or sparse) |
| fragmentation | jointed blocks (blocky, columnar, sheeted), frost cracking and spalling |
| hybridization | 2-3 of the above: growth processes first, then the weathering processes in turn |

Weathering processes act on a host form standing on the ground (outcrop,
boulder, wall or column). Earlier generator versions are in `legacy/`.

This creates 6000 samples (about 5-10 minutes; uses all CPU cores but one):

data/procedural.npz
data/procedural_metadata.json

Options: `--count`, `--resolution`, `--workers`, `--seed`, `--output`.

`procedural.npz` contains:

- `sdf`: signed distance fields in [-1, 1], positive inside material,
  surface at 0. This is what the GAN learns.
- `structures`: the binary occupancy volumes.
- `labels`: multi-hot phenomenon labels (hybrids switch on several).
- `metric_*`: one array per ecological metric (see below).

Preview the dataset (renders a gallery and prints metrics):

python inspect_dataset.py

### Ecological metrics

Computed in `ecology_metrics.py`. The z axis is vertical (index 0 is the
ground).

| Metric | Meaning |
|---|---|
| occupancy | fraction of the volume that is material |
| surface_to_volume | exposed voxel faces per material voxel |
| water_retention | volume of exterior pockets that cannot drain, per material volume |
| crevice_fraction | share of the surface that is sheltered (concave, crevice-like) |
| overhang_fraction | share of the surface facing downward onto open air |
| enclosed_void_fraction | sealed internal voids, per material volume |
| euler_number | topology: lower values mean more tunnels/handles |
| fractal_dimension | box-counting dimension of the surface |

The metrics listed in `condition_metrics` in `config.py` are given to the
GAN as conditions.

## 3. Train

Run:

python train.py

Training is measured in steps (`total_steps` in `config.py`, default
30000). Expect recognisable forms after roughly 10k steps.

Training resumes automatically from the latest saved state in
`checkpoints/train_state/`. Use `--fresh` to start over. Other options:
`--steps`, `--batch-size`, `--data`.

While training:

- `outputs/preview_XXXXXX.png` shows the same 8 conditions and latent
  vectors every `preview_every` steps, so successive images are directly
  comparable.
- Each preview prints the "condition error": how far the generated shapes'
  metrics are from the requested ones, in dataset standard deviations.
  It should fall as training progresses.
- `tensorboard --logdir logs` plots losses and condition errors.
- `checkpoints/generator_ema_XXXXXX.weights.h5` is saved every
  `checkpoint_every` steps with `checkpoints/condition_stats.json`.

The training setup: non-saturating GAN loss with an R1 gradient penalty,
a projection discriminator with a minibatch standard-deviation layer
(against mode collapse), FiLM conditioning at every generator resolution,
and an exponential moving average (EMA) of the generator weights, which is
what gets saved and sampled.

## 4. Generate new structures

For example:

python generate.py \
    --checkpoint checkpoints/generator_ema_030000.weights.h5 \
    --phenomena erosion,porosity \
    --metric water_retention=0.02 \
    --metric crevice_fraction=0.08 \
    --count 20 \
    --obj

- `--phenomena`: one or more processes to combine. Omit for a random one.
- `--metric NAME=VALUE`: target values in the metric's own units.
  Unset metrics use the dataset median. `condition_stats.json` lists the
  10th/50th/90th percentiles of each metric in the training data; targets
  far outside that range will not be honoured.
- `--truncation 0.7`: less varied, more typical shapes.
- `--obj`: also write an OBJ next to each `.npy`.

For each sample the script prints the requested and the achieved metrics.

## 5. Convert a generated structure to OBJ

Run:

python export_obj.py \
    --input outputs/generated_0000.npy \
    --output outputs/generated_0000.obj

The iso level defaults to 0 for signed distance volumes and 0.5 for
occupancy volumes; override with `--threshold`.

The OBJ can then be opened in:

Blender
Rhino
Grasshopper
MeshLab
Maya
Houdini
etc.

Coordinates are in [-1, 1] with z up (matches Rhino; in Blender's OBJ
importer choose Z as the up axis).

## 6. Inspect the generated volume

The .npy file contains a 3D signed distance field.

Values:

below 0 = empty

above 0 = material

0 = the surface

## 7. Surface detail (fine weathering)

The 3D GAN gives the overall form at 32^3, too coarse to hold fine
weathering. A second, 2D GAN learns high-resolution height maps of
weathered surfaces, and `apply_detail.py` carves them into the form at
128^3 (or higher) before meshing, so pits, ledges and rills are real
geometry.

The height maps come from 2D process simulations with the same phenomenon
labels as the volumes:

| Phenomenon | Surface process |
|---|---|
| branching | drainage or root networks from flow accumulation |
| clustering | colonies spreading into knobbed domes |
| layering | cross-bedded strata, soft beds recessed behind hard ledges |
| erosion | rain droplets carving rills, or salt weathering into honeycomb (tafoni) |
| cavitation | karst solution pans and runnels |
| porosity | reaction-diffusion vesicles |
| fragmentation | jointed blocks, open cracks, spalled blocks |

Every map tiles seamlessly, and the texture GAN uses wrap-around padding
so its output tiles too, which lets one texture wrap around a form.

Build the texture dataset (about a minute) and train the texture GAN
(2D, light enough for a laptop GPU):

python generate_textures.py

python train_textures.py

Previews go to `outputs_texture/`, weights to `checkpoints_texture/`.
Training resumes automatically; `--fresh` starts over.

Add detail to a generated structure:

python apply_detail.py --input outputs/generated_0000.npy

- The texture phenomena are read from `generated_0000.json`, which
  `generate.py` writes next to each sample; `--phenomena erosion,layering`
  overrides them.
- Without a trained texture GAN, or with `--procedural`, the textures are
  simulated directly instead.
- `--amplitude` sets the detail depth (fine-grid voxels, default 2.5),
  `--tile` how finely it repeats (default 1.5), `--resolution` the fine
  grid (default 128; 192 or 256 for print-scale detail), and
  `--weathering` how much stronger detail is on upward-facing surfaces.

The result is `generated_0000_detail.obj` (plus the fine `.npy` field).

## Repository layout

| File | Purpose |
|---|---|
| `config.py` | all settings |
| `generate_dataset.py` | procedural dataset |
| `ecology_metrics.py` | ecological performance metrics |
| `models.py` | conditional generator and discriminator |
| `train.py` | training loop, previews, checkpoints |
| `generate.py` | sampling with phenomenon/metric targets |
| `export_obj.py` | marching cubes to OBJ |
| `generate_textures.py` | surface-detail height map dataset |
| `texture_models.py` | tileable 2D texture generator and discriminator |
| `train_textures.py` | texture GAN training |
| `apply_detail.py` | carve surface detail into a generated form |
| `inspect_dataset.py` | dataset gallery |
| `legacy/` | earlier dataset generator versions |
