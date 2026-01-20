#!/usr/bin/env python3
"""
TRL GRPO Training for Judgment Module

Uses TRL's GRPOTrainer to train Qwen2.5-7B with LoRA adapters
for judgment cognitive operation.
"""

import os
import sys
import logging
import re
from pathlib import Path
import json


def setup_logging(output_dir: Path, console_level="INFO", file_level="DEBUG"):
    """Configure dual logging: info to console, debug to file."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler (INFO level)
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, console_level))
    console.setFormatter(logging.Formatter('%(message)s'))

    # File handler (DEBUG level)
    from datetime import datetime
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, file_level))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)

    # Also configure transformers/trl logging
    logging.getLogger("transformers").setLevel(logging.DEBUG)
    logging.getLogger("trl").setLevel(logging.DEBUG)

    print(f"Logging: console={console_level}, file={file_level}")
    print(f"Log file: {log_file}")

    return log_file

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig


def load_judgment_data(data_path: Path) -> list[dict]:
    """Load judgment training data."""
    examples = []

    with open(data_path, 'r') as f:
        for line in f:
            sample = json.loads(line)

            # Extract messages
            messages = sample.get('messages', [])
            user_msg = next((m for m in messages if m['role'] == 'user'), None)
            assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)
            system_msg = next((m for m in messages if m['role'] == 'system'), None)

            if not user_msg or not assistant_msg:
                continue

            # Build prompt
            prompt = ""
            if system_msg:
                prompt = f"<|im_start|>system\n{system_msg['content']}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{user_msg['content']}<|im_end|>\n<|im_start|>assistant\n"

            # Extract the expected judgment
            expected = assistant_msg['content']

            examples.append({
                "prompt": prompt,
                "expected": expected,
                "domain": sample['metadata']['domain'],
                "judgment_type": sample['metadata']['judgment_type'],
            })

    return examples


def extract_judgment(completion: str) -> str | None:
    """
    Extract the judgment from a completion using flexible pattern matching.

    Returns: "Yes", "No", "Insufficient", or None if not found.
    """
    output_lower = completion.lower()

    # Pattern: "judgment" followed by separator then value
    # Handles: "Judgment: Yes", "Judgment - No", "judgment:\nInsufficient", etc.
    pattern = r"judgment[\s:\-]+\s*(yes|no|insufficient)"
    match = re.search(pattern, output_lower)

    if match:
        value = match.group(1)
        # Normalize to title case
        return value.capitalize()

    return None


def judgment_reward_fn(
    completions: list[str],
    expected_judgment: list[str] = None,
    domain: list[str] = None,
    **kwargs
) -> list[float]:
    """
    Reward function for judgment quality with correctness-dominant scoring.

    Key principle: Structure without correct judgment is empty form.
    Correctness gates structural rewards via multiplicative scoring.

    Args:
        completions: List of model completions
        expected_judgment: List of oracle-verified correct judgments ("Yes"/"No"/"Insufficient")
        domain: List of domains for domain-appropriate vocabulary bonus
        **kwargs: Additional fields passed by TRL (prompts, etc.)

    Returns:
        List of reward scores

    Scoring (max 1.0):
        - Correctness: 0.6 (correct) or -0.5 (wrong) -- PRIMARY SIGNAL
        - Structure: 0.15 (has explicit judgment marker)
        - Reasoning: 0.15 (conditions, fulfillment, causal language)
        - Domain: 0.1 (domain-appropriate vocabulary)

    Multiplicative: Structure/reasoning/domain only count if judgment is correct.
    Wrong judgment cannot achieve positive reward.
    """
    rewards = []

    # Domain-appropriate vocabulary
    DOMAIN_KEYWORDS = {
        "mathematical": ["proof", "axiom", "theorem", "derive", "follows", "definition"],
        "empirical": ["evidence", "data", "observation", "measure", "experiment", "verify"],
        "common_sense": ["practical", "situation", "works", "experience", "context"],
        "pop_science": ["unfulfilled", "claimed", "authority", "correlation", "alleged"],
        "philosophic": ["coherent", "performative", "self", "presuppose", "position"],
    }

    for i, completion in enumerate(completions):
        output_lower = completion.lower()

        # === EXTRACT MODEL'S JUDGMENT ===
        model_judgment = extract_judgment(completion)

        # === CORRECTNESS (primary signal) ===
        expected = expected_judgment[i] if expected_judgment and i < len(expected_judgment) else None

        if model_judgment and expected:
            is_correct = (model_judgment == expected)
        elif model_judgment is None:
            is_correct = False  # No judgment = wrong
        else:
            # No expected value available (shouldn't happen in training)
            is_correct = None

        # Correctness score
        if is_correct is True:
            correctness_score = 0.6
            correctness_multiplier = 1.0
        elif is_correct is False:
            correctness_score = -0.5
            correctness_multiplier = 0.0  # Zeros out structural rewards
        else:
            # Fallback if no expected value (shouldn't happen)
            correctness_score = 0.0
            correctness_multiplier = 0.5

        # === STRUCTURE (has explicit judgment marker) ===
        structure_score = 0.15 if model_judgment is not None else 0.0

        # === REASONING (conditions, fulfillment, causal language) ===
        reasoning_markers = [
            "condition" in output_lower,
            "fulfill" in output_lower or "met" in output_lower,
            "link" in output_lower,
            any(w in output_lower for w in ["because", "since", "therefore", "thus"]),
        ]
        reasoning_score = min(0.15, sum(reasoning_markers) * 0.05)

        # === DOMAIN (domain-appropriate vocabulary) ===
        domain_score = 0.0
        if domain and i < len(domain) and domain[i]:
            d = domain[i].lower()
            keywords = DOMAIN_KEYWORDS.get(d, [])
            if any(kw in output_lower for kw in keywords):
                domain_score = 0.1

        # === PENALTIES ===
        length_penalty = -0.1 if len(completion) < 50 else 0.0

        # === FINAL SCORE ===
        # Correctness is additive; structure/reasoning/domain are gated by correctness
        structural_bonus = (structure_score + reasoning_score + domain_score) * correctness_multiplier
        reward = correctness_score + structural_bonus + length_penalty

        rewards.append(reward)

    return rewards


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TRL GRPO training for judgment")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/oracle_generated/judgment_train.jsonl"),
        help="Training data path"
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/oracle_generated/judgment_val.jsonl"),
        help="Validation data path (for early stopping)"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop if no improvement for N evals (0 to disable)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/judgment_grpo"),
        help="Output directory"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just load data, don't train"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TRL GRPO TRAINING FOR JUDGMENT")
    print("=" * 70)

    # Set up logging (debug to file, info to console)
    args.output.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(args.output, console_level="INFO", file_level="DEBUG")

    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load data
    print(f"\nLoading data from {args.data}...")
    examples = load_judgment_data(args.data)
    print(f"Loaded {len(examples)} examples")

    # Show distribution
    domains = {}
    judgments = {}
    for ex in examples:
        domains[ex['domain']] = domains.get(ex['domain'], 0) + 1
        judgments[ex['judgment_type']] = judgments.get(ex['judgment_type'], 0) + 1
    print(f"By domain: {domains}")
    print(f"By judgment: {judgments}")

    if args.dry_run:
        print("\nDry run - sample prompt:")
        print("-" * 50)
        print(examples[0]['prompt'][:500] + "...")
        print("-" * 50)
        print(f"Expected: {examples[0]['expected'][:200]}...")
        return

    # Create dataset with expected judgment and domain for reward function
    dataset = Dataset.from_list([{
        "prompt": ex["prompt"],
        "expected_judgment": ex["judgment_type"],  # "Yes", "No", "Insufficient"
        "domain": ex["domain"],
    } for ex in examples])

    # Load validation data if early stopping enabled
    val_dataset = None
    if args.early_stopping_patience > 0 and args.val_data.exists():
        print(f"\nLoading validation data from {args.val_data}...")
        val_examples = load_judgment_data(args.val_data)
        val_dataset = Dataset.from_list([{
            "prompt": ex["prompt"],
            "expected_judgment": ex["judgment_type"],
            "domain": ex["domain"],
        } for ex in val_examples])
        print(f"Loaded {len(val_examples)} validation examples")

    # Model setup
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\nLoading model: {model_name}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # LoRA config (trainer will apply this)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj"
        ],
        task_type="CAUSAL_LM",
    )

    # Create logs directory
    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=str(args.output),
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=10,
        logging_dir=str(log_dir),
        save_steps=20,
        max_prompt_length=1024,
        max_completion_length=512,
        num_generations=4,  # Number of generations per prompt for GRPO
        temperature=0.7,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        log_level="info",
        log_completions=True,  # Log sample completions to see what model generates
        # Early stopping settings
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=20 if val_dataset is not None else None,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="reward" if val_dataset is not None else None,
        greater_is_better=True,
    )

    # Set up callbacks
    callbacks = []
    if val_dataset is not None and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        print(f"Early stopping enabled: patience={args.early_stopping_patience} evals")

    # Create trainer
    print(f"\nInitializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        reward_funcs=judgment_reward_fn,
        peft_config=lora_config,
        callbacks=callbacks if callbacks else None,
    )

    # Train
    print(f"\nStarting training for {args.steps} steps...")
    trainer.train()

    # Save
    print(f"\nSaving model to {args.output}...")
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
