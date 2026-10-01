from dataclasses import dataclass
import os


@dataclass
class Config:

    # ---------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------

    resolution: int = 32

    # Latent vector fed to the generator
    latent_dim: int = 128

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset_size: int = 100
    batch_size: int = 8

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
