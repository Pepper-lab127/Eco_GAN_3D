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
