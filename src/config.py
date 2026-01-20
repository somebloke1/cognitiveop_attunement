"""
Project configuration and environment setup.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class HardwareConfig:
    """Hardware configuration for training and inference."""

    device: str = "cuda:0"  # RTX 3090
    dtype: str = "bfloat16"  # For modern GPUs
    max_memory_gb: int = 24

    # Quantization for training
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    r: int = 64  # Rank
    lora_alpha: int = 128  # Scaling factor
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Batch and gradient
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Learning rate
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03

    # Training duration
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use epochs

    # Optimization
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # Output
    output_dir: str = str(PROJECT_ROOT / "models" / "checkpoints")


@dataclass
class CognitiveConfig:
    """Configuration for cognitive operation levels."""

    # Base model to use (will be selected after benchmarking)
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Adapter names for each cognitive level
    adapters: dict = field(default_factory=lambda: {
        "attention": "adapters/attention",
        "understanding": "adapters/understanding",
        "judgment": "adapters/judgment",
        "decision": "adapters/decision"
    })

    # Maximum sequence length
    max_seq_length: int = 4096


@dataclass
class ProjectConfig:
    """Master project configuration."""

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)

    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")

    # API tokens
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    wandb_api_key: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_API_KEY"))


# Global config instance
config = ProjectConfig()
