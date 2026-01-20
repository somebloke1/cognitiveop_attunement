#!/usr/bin/env python3
"""
Generate Large Judgment Dataset

Creates a 500+ example dataset for comprehensive training.
Combines oracle-generated examples with level-themed challenges.
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

from src.dspy_cognitive.data_generation import OracleDataPipeline
from src.dspy_cognitive.invulnerability import LevelThemedChallengeGenerator


def generate_oracle_dataset(
    output_path: Path,
    num_per_cell: int = 10,
):
    """Generate standard oracle dataset across all domains."""
    print("=" * 70)
    print("ORACLE DATASET GENERATION")
    print("=" * 70)

    pipeline = OracleDataPipeline(
        scenario_model="gemini/gemini-2.0-flash",
        judgment_model="gemini/gemini-2.0-flash",
        verification_model="gemini/gemini-2.0-flash",
        verify_judgments=True,
    )

    domains = ["mathematical", "empirical", "common_sense", "pop_science", "philosophic"]
    difficulties = ["easy", "medium", "hard"]
    judgments = ["Yes", "No", "Insufficient"]

    # Calculate target
    total_cells = len(domains) * len(difficulties) * len(judgments)
    total_target = total_cells * num_per_cell  # 45 cells × 10 = 450

    print(f"\nConfiguration:")
    print(f"  Domains: {len(domains)}")
    print(f"  Difficulties: {len(difficulties)}")
    print(f"  Judgments: {len(judgments)}")
    print(f"  Per cell: {num_per_cell}")
    print(f"  Target: {total_target}")

    examples = pipeline.generate_dataset(
        num_examples_per_cell=num_per_cell,
        domains=domains,
        difficulties=difficulties,
        judgments=judgments,
        output_path=output_path,
        only_verified=True,
    )

    return examples


def generate_level_challenges(
    output_path: Path,
    per_cell: int = 2,
):
    """Generate level-themed challenges."""
    print("\n" + "=" * 70)
    print("LEVEL-THEMED CHALLENGE GENERATION")
    print("=" * 70)

    generator = LevelThemedChallengeGenerator(model="gemini/gemini-2.0-flash")

    levels = ["attention", "understanding", "judgment", "decision"]
    domains = ["mathematical", "empirical", "common_sense", "pop_science", "philosophic"]
    difficulties = ["medium", "hard"]

    total_target = len(levels) * len(domains) * len(difficulties) * per_cell

    print(f"\nConfiguration:")
    print(f"  Levels: {len(levels)}")
    print(f"  Domains: {len(domains)}")
    print(f"  Difficulties: {len(difficulties)}")
    print(f"  Per cell: {per_cell}")
    print(f"  Target: {total_target}")

    challenges = []

    for level in levels:
        for domain in domains:
            for difficulty in difficulties:
                cell_challenges = []
                attempts = 0
                max_attempts = per_cell * 3

                while len(cell_challenges) < per_cell and attempts < max_attempts:
                    attempts += 1
                    try:
                        result = generator(
                            level_theme=level,
                            domain=domain,
                            difficulty=difficulty,
                        )

                        challenge = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": """You are a Judgment cognitive operation (krisis/mishpat/viveka). Evaluate whether a proposition is true by grasping the virtually unconditioned.

If conditions are fulfilled, affirm (Yes). If identifiably not fulfilled, deny (No). If cannot determine, withhold (Insufficient).

Be reasonable (episteme, not doxa). Demand sufficient reason."""
                                },
                                {
                                    "role": "user",
                                    "content": f"""Proposition: {result.proposition}

Evidence: {result.evidence}

Conditions for truth:
{chr(10).join(f"- {c}" for c in result.conditions)}"""
                                },
                                {
                                    "role": "assistant",
                                    "content": f"""Judgment: {result.correct_judgment}

Vulnerability note: This scenario contains a {level}-level vulnerability. {result.vulnerability_explanation}"""
                                }
                            ],
                            "metadata": {
                                "level": "judgment",
                                "domain": domain,
                                "judgment_type": result.correct_judgment,
                                "difficulty": difficulty,
                                "vulnerability_level": level,
                                "vulnerability_explanation": result.vulnerability_explanation,
                                "generated_by": "level_themed_generator",
                            }
                        }

                        cell_challenges.append(challenge)

                    except Exception as e:
                        print(f"  Error generating {level}/{domain}/{difficulty}: {e}")
                        continue

                challenges.extend(cell_challenges)
                if len(cell_challenges) > 0:
                    print(f"  {level}/{domain}/{difficulty}: {len(cell_challenges)} challenges")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for c in challenges:
            f.write(json.dumps(c) + '\n')

    print(f"\nGenerated {len(challenges)} level-themed challenges")
    return challenges


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate large judgment dataset")
    parser.add_argument(
        "--oracle-per-cell",
        type=int,
        default=10,
        help="Oracle examples per (domain, difficulty, judgment) cell"
    )
    parser.add_argument(
        "--level-per-cell",
        type=int,
        default=2,
        help="Level challenges per (level, domain, difficulty) cell"
    )
    parser.add_argument(
        "--oracle-only",
        action="store_true",
        help="Only generate oracle dataset"
    )
    parser.add_argument(
        "--level-only",
        action="store_true",
        help="Only generate level-themed challenges"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/oracle_generated"),
        help="Output directory"
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_examples = 0

    if not args.level_only:
        # Generate oracle dataset
        oracle_path = output_dir / "judgment_large_oracle.jsonl"
        oracle_examples = generate_oracle_dataset(
            output_path=oracle_path,
            num_per_cell=args.oracle_per_cell,
        )
        total_examples += len(oracle_examples)

    if not args.oracle_only:
        # Generate level-themed challenges
        level_path = output_dir / "judgment_large_level.jsonl"
        level_challenges = generate_level_challenges(
            output_path=level_path,
            per_cell=args.level_per_cell,
        )
        total_examples += len(level_challenges)

    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal examples generated: {total_examples}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
