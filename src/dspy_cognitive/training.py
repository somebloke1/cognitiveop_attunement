"""
GRPO Training for Cognitive Operations

Uses DSPy's ArborGRPO to train the judgment module through
reinforcement learning with multi-tier verification as the reward signal.

Architecture:
- Student model: Qwen2.5-7B with LoRA adapters
- Metric: Multi-tier verification (deterministic → heuristic → oracle)
- Oracle: Gemini Pro for difficult cases
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import dspy

from .modules import JudgmentModule, JudgmentProgram
from .metrics import judgment_metric, get_verifier


def create_judgment_examples(data_path: Path) -> List[dspy.Example]:
    """
    Load judgment training data and convert to DSPy Examples.

    Args:
        data_path: Path to JSONL file with judgment samples

    Returns:
        List of dspy.Example objects
    """
    import json

    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            sample = json.loads(line)

            # Extract from message format
            messages = sample.get('messages', [])
            user_msg = next((m for m in messages if m['role'] == 'user'), None)
            if not user_msg:
                continue

            content = user_msg['content']

            # Parse the structured content
            # Expected format:
            # Proposition: ...
            # Evidence: ...
            # Conditions for truth:
            # - condition 1
            # - condition 2

            lines = content.split('\n')
            proposition = ""
            evidence = ""
            conditions = []

            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith('Proposition:'):
                    current_section = 'proposition'
                    proposition = line.replace('Proposition:', '').strip()
                elif line.startswith('Evidence:'):
                    current_section = 'evidence'
                    evidence = line.replace('Evidence:', '').strip()
                elif line.startswith('Conditions for truth:'):
                    current_section = 'conditions'
                elif current_section == 'conditions' and line.startswith('-'):
                    conditions.append(line[1:].strip())
                elif current_section == 'evidence' and line and not line.startswith('Conditions'):
                    evidence += ' ' + line
                elif current_section == 'proposition' and line and not line.startswith('Evidence'):
                    proposition += ' ' + line

            # Get domain from metadata
            metadata = sample.get('metadata', {})
            domain = metadata.get('domain', 'unknown')

            # Create example
            example = dspy.Example(
                proposition=proposition.strip(),
                evidence=evidence.strip(),
                conditions=conditions,
                domain=domain,
            ).with_inputs('proposition', 'evidence', 'conditions', 'domain')

            examples.append(example)

    return examples


def setup_student_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    use_local: bool = True,
) -> dspy.LM:
    """
    Set up the student model for training.

    Args:
        model_name: HuggingFace model name
        use_local: Whether to run locally (vs API)

    Returns:
        Configured dspy.LM instance
    """
    if use_local:
        # For local training with GRPO, we use vLLM or HuggingFace
        # ArborGRPO handles the actual training loop
        return dspy.LM(
            f"huggingface/{model_name}",
            # These will be overridden by ArborGRPO
        )
    else:
        # API-based (for testing without local GPU)
        return dspy.LM(model_name)


def setup_oracle_model(
    model_name: str = "gemini/gemini-2.0-flash",
) -> dspy.LM:
    """
    Set up the oracle model for Tier 3 verification.

    Args:
        model_name: Model identifier (gemini/..., openai/..., anthropic/...)

    Returns:
        Configured dspy.LM instance
    """
    return dspy.LM(model_name)


def get_grpo_train_kwargs(
    num_gpus: int = 1,
    lora_r: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 1e-5,
    max_steps: int = 1000,
) -> Dict[str, Any]:
    """
    Get training kwargs for ArborGRPO.

    Configured for RTX 3090 (24GB VRAM).
    """
    return {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "temperature": 1.0,
        "top_k": -1,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "beta": 0.01,  # KL penalty
        "learning_rate": learning_rate,
        "gradient_checkpointing": True,
        "bf16": True,
        "lr_scheduler_type": "cosine",
        "loss_type": "grpo",  # or "dapo" for DAPO variant
        "max_steps": max_steps,
        "report_to": "none",  # or "wandb"
        "logging_steps": 10,
        "max_prompt_length": 2048,
        "max_completion_length": 512,
        "scale_rewards": True,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "lora_config": {
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.05,
            "r": lora_r,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj"
            ],
        },
        "num_training_gpus": num_gpus,
        "num_inference_gpus": 0,  # Use same GPUs for both
    }


def train_judgment_grpo(
    trainset: List[dspy.Example],
    valset: Optional[List[dspy.Example]] = None,
    student_model: str = "Qwen/Qwen2.5-7B-Instruct",
    oracle_model: str = "gemini/gemini-2.0-flash",
    output_dir: Path = Path("models/dspy_judgment"),
    num_train_steps: int = 500,
    use_oracle: bool = True,
):
    """
    Train judgment module using ArborGRPO.

    Args:
        trainset: Training examples
        valset: Validation examples (optional)
        student_model: Model to train
        oracle_model: Model for Tier 3 verification
        output_dir: Where to save trained model
        num_train_steps: Number of training steps
        use_oracle: Whether to use oracle for difficult cases
    """
    from dspy.teleprompt import ArborGRPO

    # Set up models
    student_lm = setup_student_model(student_model)
    dspy.configure(lm=student_lm)

    # Initialize verifier with oracle
    verifier = get_verifier(oracle_model=oracle_model, use_oracle=use_oracle)

    # Create judgment program
    program = JudgmentProgram()

    # Get training kwargs
    train_kwargs = get_grpo_train_kwargs(
        num_gpus=1,
        max_steps=num_train_steps,
    )

    # Create GRPO compiler
    compiler = ArborGRPO(
        metric=judgment_metric,
        num_dspy_examples_per_grpo_step=4,
        num_rollouts_per_grpo_step=16,
        exclude_demos=True,
        num_train_steps=num_train_steps,
        num_threads=8,
        use_train_as_val=valset is None,
        num_steps_for_val=50,
        train_kwargs=train_kwargs,
        checkpoint="single-best",
    )

    # Compile (train)
    print(f"Starting GRPO training for {num_train_steps} steps...")
    optimized_program = compiler.compile(
        student=program,
        trainset=trainset,
        valset=valset or trainset[:10],
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    optimized_program.save(output_dir / "judgment_program")
    print(f"Saved optimized program to {output_dir}")

    return optimized_program


# ============ SIMPLER TRAINING: MIPROv2 FOR PROMPT OPTIMIZATION ============

def train_judgment_mipro(
    trainset: List[dspy.Example],
    valset: Optional[List[dspy.Example]] = None,
    student_model: str = "Qwen/Qwen2.5-7B-Instruct",
    prompt_model: str = "gemini/gemini-2.0-flash",
    output_dir: Path = Path("models/dspy_judgment_mipro"),
):
    """
    Optimize judgment prompts using MIPROv2.

    This is lighter than GRPO - optimizes prompts without LoRA training.
    Good for iteration before committing to full GRPO.
    """
    # Set up models
    student_lm = setup_student_model(student_model, use_local=False)
    prompt_lm = dspy.LM(prompt_model)

    dspy.configure(lm=student_lm)

    # Create program
    program = JudgmentProgram()

    # Create MIPROv2 optimizer
    optimizer = dspy.MIPROv2(
        metric=judgment_metric,
        auto="light",  # or "medium", "heavy"
        num_threads=8,
        prompt_model=prompt_lm,
    )

    # Optimize
    config = dict(
        max_bootstrapped_demos=2,
        max_labeled_demos=4,
        minibatch_size=20,
    )

    print("Starting MIPROv2 prompt optimization...")
    optimized = optimizer.compile(
        program,
        trainset=trainset,
        valset=valset,
        **config
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    optimized.save(output_dir / "judgment_program_mipro")
    print(f"Saved optimized program to {output_dir}")

    return optimized


# ============ CLI ENTRY POINT ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train cognitive judgment with DSPy")
    parser.add_argument(
        "--method",
        choices=["grpo", "mipro"],
        default="mipro",
        help="Training method"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/synthetic/judgment_samples.jsonl"),
        help="Path to training data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/dspy_judgment"),
        help="Output directory"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Training steps (for GRPO)"
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Disable oracle (Tier 3) verification"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading training data from {args.data}...")
    examples = create_judgment_examples(args.data)
    print(f"Loaded {len(examples)} examples")

    # Split train/val
    split_idx = int(len(examples) * 0.8)
    trainset = examples[:split_idx]
    valset = examples[split_idx:]

    print(f"Train: {len(trainset)}, Val: {len(valset)}")

    if args.method == "grpo":
        train_judgment_grpo(
            trainset=trainset,
            valset=valset,
            output_dir=args.output,
            num_train_steps=args.steps,
            use_oracle=not args.no_oracle,
        )
    else:
        train_judgment_mipro(
            trainset=trainset,
            valset=valset,
            output_dir=args.output,
        )
