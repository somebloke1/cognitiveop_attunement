"""
Training pipeline for cognitive operation LoRA adapters.

Uses TRL's SFTTrainer for supervised fine-tuning with PEFT/LoRA.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig

from .config import ProjectConfig, config
from .cognitive_levels import CognitiveLevel, get_level


def setup_environment():
    """Configure environment for training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # RTX 3090
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_base_model(
    model_name: str,
    hw_config: Dict[str, Any],
) -> tuple:
    """Load base model with 4-bit quantization for QLoRA training."""

    # BitsAndBytes 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=hw_config.get("load_in_4bit", True),
        bnb_4bit_quant_type=hw_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, hw_config.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=hw_config.get("bnb_4bit_use_double_quant", True),
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def create_lora_config(
    level: CognitiveLevel,
    lora_params: Dict[str, Any],
) -> LoraConfig:
    """Create LoRA configuration for a cognitive level adapter."""

    level_spec = get_level(level)

    return LoraConfig(
        r=lora_params.get("r", 64),
        lora_alpha=lora_params.get("lora_alpha", 128),
        lora_dropout=lora_params.get("lora_dropout", 0.05),
        target_modules=lora_params.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias=lora_params.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


def formatting_func(example: Dict[str, Any]) -> str:
    """Format a training example into the chat format string."""
    messages = example["messages"]

    # Use tokenizer's chat template if available
    # For now, use a simple format
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|system|>\n{content}\n"
        elif role == "user":
            text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}\n"

    return text


class CognitiveTrainer:
    """Trainer for cognitive operation adapters."""

    def __init__(
        self,
        project_config: Optional[ProjectConfig] = None,
    ):
        self.config = project_config or config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the base model for training."""
        setup_environment()

        print(f"Loading base model: {self.config.cognitive.base_model}")
        self.model, self.tokenizer = load_base_model(
            self.config.cognitive.base_model,
            self.config.hardware.__dict__,
        )
        print("Base model loaded successfully.")

    def train_adapter(
        self,
        level: CognitiveLevel,
        train_dataset_path: Path,
        output_dir: Optional[Path] = None,
        eval_dataset_path: Optional[Path] = None,
    ):
        """Train a LoRA adapter for a specific cognitive level."""

        if self.model is None:
            self.load_model()

        level_spec = get_level(level)
        print(f"\n{'='*60}")
        print(f"Training adapter for: {level_spec.name}")
        print(f"Transcendental precept: {level_spec.transcendental_precept}")
        print(f"{'='*60}")

        # Load dataset
        print(f"Loading dataset from: {train_dataset_path}")
        train_dataset = load_dataset(
            "json",
            data_files=str(train_dataset_path),
            split="train"
        )

        eval_dataset = None
        if eval_dataset_path and eval_dataset_path.exists():
            eval_dataset = load_dataset(
                "json",
                data_files=str(eval_dataset_path),
                split="train"
            )

        # Create LoRA config
        lora_config = create_lora_config(level, self.config.lora.__dict__)

        # Apply LoRA to model
        peft_model = get_peft_model(self.model, lora_config)
        peft_model.print_trainable_parameters()

        # Output directory
        if output_dir is None:
            output_dir = self.config.models_dir / "adapters" / level_spec.adapter_name

        # Training arguments using SFTConfig (TRL 0.27+)
        training_args = SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            warmup_ratio=self.config.training.warmup_ratio,
            num_train_epochs=self.config.training.num_train_epochs,
            max_steps=self.config.training.max_steps,
            optim=self.config.training.optim,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.training.eval_steps if eval_dataset else None,
            fp16=False,
            bf16=True,
            save_total_limit=3,
            load_best_model_at_end=eval_dataset is not None,
            report_to="none",  # Disable wandb for now
            max_length=self.config.cognitive.max_seq_length,
        )

        # Create trainer
        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Save adapter
        print(f"Saving adapter to: {output_dir}")
        peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"Training complete for {level_spec.name} adapter.")

        return trainer

    def train_all_adapters(
        self,
        data_dir: Path,
    ):
        """Train adapters for all cognitive levels."""

        levels_data = {
            CognitiveLevel.ATTENTION: "attention_samples.jsonl",
            CognitiveLevel.UNDERSTANDING: "understanding_samples.jsonl",
            CognitiveLevel.JUDGMENT: "judgment_samples.jsonl",
            CognitiveLevel.DECISION: "decision_samples.jsonl",
        }

        for level, data_file in levels_data.items():
            data_path = data_dir / data_file
            if data_path.exists():
                self.train_adapter(level, data_path)
            else:
                print(f"Skipping {level.name}: No data file at {data_path}")


if __name__ == "__main__":
    from pathlib import Path

    # Test training pipeline with Understanding level
    trainer = CognitiveTrainer()

    data_dir = Path(__file__).parent.parent / "data" / "synthetic"

    # Train Understanding adapter as a test
    trainer.train_adapter(
        CognitiveLevel.UNDERSTANDING,
        data_dir / "understanding_samples.jsonl",
    )
