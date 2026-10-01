from dataclasses import dataclass
import os


@dataclass
class Config:

    # ---------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------

    # 32 is the practical default on a 6 GB laptop GPU (GTX 1060).
    # 64 works with batch_size 4-8 and is roughly 8x slower per step.
    resolution: int = 32

    # Latent vector fed to the generator
    latent_dim: int = 128

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset_path: str = "data/procedural.npz"

    # Procedural samples are cheap; thousands are needed for a GAN.
    dataset_size: int = 6000

    # Signed distance is clipped at this many voxels from the surface
    # and scaled into [-1, 1].
    tsdf_truncation: float = 3.0

    # Gaussian sigma (voxels) applied to the distance field to round
    # voxel stair-steps. 0 disables.
    tsdf_smoothing: float = 0.6

    batch_size: int = 16

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    epochs: int = 10

    learning_rate_generator: float = 0.0002
    learning_rate_discriminator: float = 0.0002

    beta_1: float = 0.5
    beta_2: float = 0.999

    # Encourage generated objects to remain spatially sparse.
    sparsity_weight: float = 0.15

    target_occupancy: float = 0.10

    # ---------------------------------------------------------
    # Checkpoints
    # ---------------------------------------------------------

    checkpoint_dir: str = "checkpoints"

    # ---------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------

    output_dir: str = "outputs"

    # ---------------------------------------------------------
    # Random seed
    # ---------------------------------------------------------

    seed: int = 42


CONFIG = Config()

os.makedirs(CONFIG.checkpoint_dir, exist_ok=True)
os.makedirs(CONFIG.output_dir, exist_ok=True)
