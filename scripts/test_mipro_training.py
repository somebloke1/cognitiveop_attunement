#!/usr/bin/env python3
"""
Test MIPROv2 Training

Tests the MIPROv2 prompt optimization with a small subset of data
to verify the training pipeline works before committing to full training.
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
from src.dspy_cognitive.modules import JudgmentModule
from src.dspy_cognitive.metrics import judgment_metric_no_oracle


def load_examples(data_path: Path, limit: int = 20) -> list[dspy.Example]:
    """Load a limited set of examples for testing."""
    examples = []

    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

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

            # Create example
            example = dspy.Example(
                proposition=proposition.strip(),
                evidence=evidence.strip(),
                conditions=conditions,
                domain=domain,
            ).with_inputs('proposition', 'evidence', 'conditions', 'domain')

            examples.append(example)

    return examples


def test_baseline(examples: list, lm):
    """Test baseline performance before optimization."""
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)

    dspy.configure(lm=lm)

    module = JudgmentModule(use_chain_of_thought=True)

    scores = []
    for i, ex in enumerate(examples[:10]):
        try:
            pred = module(
                proposition=ex.proposition,
                evidence=ex.evidence,
                conditions=ex.conditions,
                domain=ex.domain,
            )
            score = judgment_metric_no_oracle(ex, pred)
            scores.append(score)
            print(f"  Example {i+1}: score={score:.2f}, judgment={pred.judgment}")
        except Exception as e:
            print(f"  Example {i+1}: ERROR - {e}")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\nBaseline average score: {avg_score:.2f}")
    return avg_score


def test_mipro_optimization(trainset: list, valset: list, lm, prompt_lm):
    """Test MIPROv2 prompt optimization."""
    print("\n" + "=" * 70)
    print("MIPRO OPTIMIZATION")
    print("=" * 70)

    dspy.configure(lm=lm)

    # Create program
    module = JudgmentModule(use_chain_of_thought=True)

    # Create MIPROv2 optimizer
    print("\nInitializing MIPROv2...")
    optimizer = dspy.MIPROv2(
        metric=judgment_metric_no_oracle,
        auto="light",  # Start with light optimization
        num_threads=4,
        prompt_model=prompt_lm,
        verbose=True,
    )

    # Optimize with minimal config for testing
    config = dict(
        max_bootstrapped_demos=1,
        max_labeled_demos=2,
        minibatch_size=5,
    )

    print(f"\nStarting optimization with {len(trainset)} train, {len(valset)} val examples...")
    print(f"Config: {config}")

    try:
        optimized = optimizer.compile(
            module,
            trainset=trainset,
            valset=valset,
            **config
        )
        print("\nOptimization completed!")
        return optimized
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_optimized(optimized, examples: list, lm):
    """Evaluate the optimized module."""
    print("\n" + "=" * 70)
    print("OPTIMIZED EVALUATION")
    print("=" * 70)

    dspy.configure(lm=lm)

    scores = []
    for i, ex in enumerate(examples[:10]):
        try:
            pred = optimized(
                proposition=ex.proposition,
                evidence=ex.evidence,
                conditions=ex.conditions,
                domain=ex.domain,
            )
            score = judgment_metric_no_oracle(ex, pred)
            scores.append(score)
            print(f"  Example {i+1}: score={score:.2f}, judgment={pred.judgment}")
        except Exception as e:
            print(f"  Example {i+1}: ERROR - {e}")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\nOptimized average score: {avg_score:.2f}")
    return avg_score


def main():
    print("=" * 70)
    print("MIPRO TRAINING TEST")
    print("=" * 70)

    # Load data
    data_path = Path("data/oracle_generated/judgment_training_full.jsonl")
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        return

    examples = load_examples(data_path, limit=30)
    print(f"\nLoaded {len(examples)} examples")

    # Split
    trainset = examples[:20]
    valset = examples[20:]
    print(f"Train: {len(trainset)}, Val: {len(valset)}")

    # Configure LMs
    # Using Gemini for both student and prompt model (for testing)
    print("\nConfiguring language models...")
    student_lm = dspy.LM("gemini/gemini-2.0-flash")
    prompt_lm = dspy.LM("gemini/gemini-2.0-flash")

    # Test baseline
    baseline_score = test_baseline(trainset, student_lm)

    # Run MIPROv2 optimization
    optimized = test_mipro_optimization(trainset, valset, student_lm, prompt_lm)

    if optimized:
        # Evaluate optimized
        optimized_score = evaluate_optimized(optimized, valset, student_lm)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Baseline score: {baseline_score:.2f}")
        print(f"Optimized score: {optimized_score:.2f}")
        print(f"Improvement: {optimized_score - baseline_score:+.2f}")

        # Save optimized module
        output_dir = Path("models/dspy_judgment_mipro_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        optimized.save(output_dir / "judgment_module")
        print(f"\nSaved optimized module to {output_dir}")
    else:
        print("\nOptimization failed - no results to report")


if __name__ == "__main__":
    main()
