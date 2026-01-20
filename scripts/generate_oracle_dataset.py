#!/usr/bin/env python3
"""
Generate Oracle Training Dataset

Uses the Lonergan-augmented oracle pipeline to generate verified
judgment training data across all domains.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from src.dspy_cognitive.data_generation import OracleDataPipeline


def generate_dataset(
    num_per_cell: int = 3,
    output_path: Path = None,
    verify: bool = True,
    domains: list = None,
    difficulties: list = None,
    judgments: list = None,
):
    """
    Generate verified training dataset.

    Args:
        num_per_cell: Examples per (domain, difficulty, judgment) combination
        output_path: Where to save the JSONL output
        verify: Whether to run verification pass
        domains: Which domains to include
        difficulties: Which difficulties to include
        judgments: Which judgment types to include
    """
    # Defaults
    if output_path is None:
        output_path = Path("data/oracle_generated/judgment_training.jsonl")

    if domains is None:
        domains = ["mathematical", "empirical", "common_sense", "pop_science", "philosophic"]

    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]

    if judgments is None:
        judgments = ["Yes", "No", "Insufficient"]

    # Calculate total
    total_cells = len(domains) * len(difficulties) * len(judgments)
    total_expected = total_cells * num_per_cell

    print("=" * 70)
    print("ORACLE TRAINING DATA GENERATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Domains: {domains}")
    print(f"  Difficulties: {difficulties}")
    print(f"  Judgment types: {judgments}")
    print(f"  Examples per cell: {num_per_cell}")
    print(f"  Total cells: {total_cells}")
    print(f"  Target examples: {total_expected}")
    print(f"  Verification: {verify}")
    print(f"  Output: {output_path}")
    print("\n" + "-" * 70)

    # Initialize pipeline with augmented oracle
    pipeline = OracleDataPipeline(
        scenario_model="gemini/gemini-2.0-flash",
        judgment_model="gemini/gemini-2.0-flash",
        verification_model="gemini/gemini-2.0-flash",
        verify_judgments=verify,
    )

    # Generate dataset
    examples = pipeline.generate_dataset(
        num_examples_per_cell=num_per_cell,
        domains=domains,
        difficulties=difficulties,
        judgments=judgments,
        output_path=output_path,
        only_verified=verify,
    )

    # Summary statistics
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    # Count by domain
    domain_counts = {}
    for ex in examples:
        domain_counts[ex.domain] = domain_counts.get(ex.domain, 0) + 1

    print(f"\nTotal examples generated: {len(examples)}")
    print(f"\nBy domain:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")

    # Count by judgment type
    judgment_counts = {}
    for ex in examples:
        judgment_counts[ex.judgment] = judgment_counts.get(ex.judgment, 0) + 1

    print(f"\nBy judgment type:")
    for judgment, count in sorted(judgment_counts.items()):
        print(f"  {judgment}: {count}")

    # Verification stats
    if verify:
        verified_count = sum(1 for ex in examples if ex.verification_passed)
        print(f"\nVerification:")
        print(f"  Passed: {verified_count}/{len(examples)} ({100*verified_count/len(examples):.1f}%)")

        confidence_counts = {}
        for ex in examples:
            confidence_counts[ex.verification_confidence] = confidence_counts.get(ex.verification_confidence, 0) + 1

        print(f"\nBy confidence:")
        for conf, count in sorted(confidence_counts.items()):
            print(f"  {conf}: {count}")

    print(f"\nData saved to: {output_path}")

    return examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate oracle training data")
    parser.add_argument(
        "--num-per-cell",
        type=int,
        default=3,
        help="Examples per (domain, difficulty, judgment) cell"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/oracle_generated/judgment_training.jsonl"),
        help="Output path"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification pass"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Specific domains to generate for"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run (1 per cell, limited domains)"
    )

    args = parser.parse_args()

    if args.quick:
        # Quick test run
        examples = generate_dataset(
            num_per_cell=1,
            output_path=Path("data/oracle_generated/judgment_training_quick.jsonl"),
            verify=True,
            domains=["pop_science", "philosophic"],  # Most important for our use case
            difficulties=["medium"],
            judgments=["Yes", "No"],
        )
    else:
        examples = generate_dataset(
            num_per_cell=args.num_per_cell,
            output_path=args.output,
            verify=not args.no_verify,
            domains=args.domains,
        )
