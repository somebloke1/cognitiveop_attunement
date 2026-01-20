#!/usr/bin/env python3
"""
Run GRPO Training for Judgment Module

Uses the oracle-generated dataset to train the judgment module
with reinforcement learning via ArborGRPO.
"""

import os
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import dspy


def load_oracle_dataset(data_path: Path) -> list[dspy.Example]:
    """
    Load oracle-generated JSONL and convert to DSPy Examples.

    The oracle dataset has messages format; we extract the structured fields.
    """
    examples = []

    with open(data_path, 'r') as f:
        for line in f:
            sample = json.loads(line)

            # Extract from user message
            user_msg = next(m for m in sample['messages'] if m['role'] == 'user')
            content = user_msg['content']

            # Parse structured content
            lines = content.split('\n')
            proposition = ""
            evidence = ""
            conditions = []

            current_section = None
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith('Proposition:'):
                    current_section = 'proposition'
                    proposition = line_stripped.replace('Proposition:', '').strip()
                elif line_stripped.startswith('Evidence:'):
                    current_section = 'evidence'
                    evidence = line_stripped.replace('Evidence:', '').strip()
                elif line_stripped.startswith('Conditions for truth:'):
                    current_section = 'conditions'
                elif current_section == 'conditions' and line_stripped.startswith('-'):
                    conditions.append(line_stripped[1:].strip())
                elif current_section == 'evidence' and line_stripped and not line_stripped.startswith('Conditions'):
                    evidence += ' ' + line_stripped
                elif current_section == 'proposition' and line_stripped and not line_stripped.startswith('Evidence'):
                    proposition += ' ' + line_stripped

            # Get domain from metadata
            domain = sample['metadata']['domain']

            # Create example with inputs
            example = dspy.Example(
                proposition=proposition.strip(),
                evidence=evidence.strip(),
                conditions=conditions,
                domain=domain,
            ).with_inputs('proposition', 'evidence', 'conditions', 'domain')

            examples.append(example)

    return examples


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run GRPO training for judgment")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/oracle_generated/judgment_training_full.jsonl"),
        help="Path to oracle-generated training data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/dspy_judgment_grpo"),
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps"
    )
    parser.add_argument(
        "--method",
        choices=["grpo", "mipro"],
        default="mipro",
        help="Training method (mipro is lighter, grpo does full LoRA)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just load data and show stats, don't train"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    examples = load_oracle_dataset(args.data)
    print(f"Loaded {len(examples)} examples")

    # Show distribution
    domains = {}
    for ex in examples:
        domains[ex.domain] = domains.get(ex.domain, 0) + 1
    print(f"By domain: {domains}")

    # Split train/val
    split_idx = int(len(examples) * 0.85)
    trainset = examples[:split_idx]
    valset = examples[split_idx:]
    print(f"Train: {len(trainset)}, Val: {len(valset)}")

    if args.dry_run:
        print("\nDry run - showing sample examples:")
        for i, ex in enumerate(trainset[:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Domain: {ex.domain}")
            print(f"Proposition: {ex.proposition[:100]}...")
            print(f"Conditions: {len(ex.conditions)} conditions")
        return

    # Import training functions
    from src.dspy_cognitive.training import train_judgment_grpo, train_judgment_mipro

    if args.method == "grpo":
        print(f"\nStarting GRPO training for {args.steps} steps...")
        train_judgment_grpo(
            trainset=trainset,
            valset=valset,
            output_dir=args.output,
            num_train_steps=args.steps,
            use_oracle=True,
        )
    else:
        print(f"\nStarting MIPROv2 prompt optimization...")
        train_judgment_mipro(
            trainset=trainset,
            valset=valset,
            output_dir=args.output,
        )

    print(f"\nTraining complete. Model saved to {args.output}")


if __name__ == "__main__":
    main()
