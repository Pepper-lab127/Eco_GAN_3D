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

    # Measured ecological metrics the generator is conditioned on, in
    # addition to the phenomenon labels. Any name from
    # ecology_metrics.METRIC_NAMES can be used.
    condition_metrics: tuple = (
        "occupancy",
        "surface_to_volume",
        "water_retention",
        "crevice_fraction",
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    batch_size: int = 16

    # Training length is counted in optimizer steps, not epochs.
    # Expect usable shapes after ~10k steps and better ones by 30k+.
    total_steps: int = 30000

    learning_rate_generator: float = 0.0002
    learning_rate_discriminator: float = 0.0002

    beta_1: float = 0.0
    beta_2: float = 0.99

    # R1 gradient penalty on real samples (stabilises the discriminator).
    # Applied every r1_interval steps, scaled to compensate.
    r1_gamma: float = 1.0
    r1_interval: int = 4

    # Exponential moving average of generator weights. The EMA generator
    # is used for previews and exported checkpoints.
    ema_decay: float = 0.999

    log_every: int = 100
    preview_every: int = 1000
    checkpoint_every: int = 2500

    log_dir: str = "logs"

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
