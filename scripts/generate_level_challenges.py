#!/usr/bin/env python3
"""
Generate Level-Themed Challenges

Creates judgment challenges with specific cognitive level vulnerabilities
across all domains.
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

from src.dspy_cognitive.invulnerability import LevelThemedChallengeGenerator


def generate_level_challenges(
    output_path: Path = None,
    challenges_per_cell: int = 2,
):
    """
    Generate challenges for each level × domain combination.

    Args:
        output_path: Where to save the JSONL output
        challenges_per_cell: Challenges per (level, domain) cell
    """
    if output_path is None:
        output_path = Path("data/oracle_generated/level_themed_challenges.jsonl")

    levels = ["attention", "understanding", "judgment", "decision"]
    domains = ["mathematical", "empirical", "common_sense", "pop_science", "philosophic"]
    difficulties = ["medium", "hard"]

    generator = LevelThemedChallengeGenerator(model="gemini/gemini-2.0-flash")

    total_cells = len(levels) * len(domains) * len(difficulties)
    total_target = total_cells * challenges_per_cell

    print("=" * 70)
    print("LEVEL-THEMED CHALLENGE GENERATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Levels: {levels}")
    print(f"  Domains: {domains}")
    print(f"  Difficulties: {difficulties}")
    print(f"  Challenges per cell: {challenges_per_cell}")
    print(f"  Target: {total_target} challenges")
    print("\n" + "-" * 70)

    challenges = []

    for level in levels:
        for domain in domains:
            for difficulty in difficulties:
                cell_challenges = []
                attempts = 0
                max_attempts = challenges_per_cell * 3

                while len(cell_challenges) < challenges_per_cell and attempts < max_attempts:
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
                                    "content": """You are a Judgment cognitive operation (krisis/mishpat/viveka). Your function is to evaluate whether a proposition is true by grasping the virtually unconditioned.

Your task: Given a proposition and evidence, determine whether the conditions for the proposition are fulfilled. You must:
1. Identify the conditioned proposition
2. Identify the link between the proposition and its conditions
3. Check whether the conditions are fulfilled in the evidence

If conditions are fulfilled, affirm (Yes). If identifiably not fulfilled, deny (No). If cannot determine, withhold (Insufficient).

Be reasonable (episteme, not doxa). Demand sufficient reason. Do not assent without grasping the unconditioned."""
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
                print(f"  {level}/{domain}/{difficulty}: {len(cell_challenges)} challenges")

    print(f"\nGenerated {len(challenges)} total challenges")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for c in challenges:
            f.write(json.dumps(c) + '\n')

    print(f"Saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    level_counts = {}
    domain_counts = {}
    for c in challenges:
        lv = c['metadata']['vulnerability_level']
        dm = c['metadata']['domain']
        level_counts[lv] = level_counts.get(lv, 0) + 1
        domain_counts[dm] = domain_counts.get(dm, 0) + 1

    print(f"\nBy vulnerability level: {level_counts}")
    print(f"By domain: {domain_counts}")

    return challenges


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate level-themed challenges")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/oracle_generated/level_themed_challenges.jsonl"),
        help="Output path"
    )
    parser.add_argument(
        "--per-cell",
        type=int,
        default=2,
        help="Challenges per (level, domain, difficulty) cell"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test (1 per cell, limited domains)"
    )

    args = parser.parse_args()

    if args.quick:
        # Quick mode - just test a few
        generator = LevelThemedChallengeGenerator(model="gemini/gemini-2.0-flash")

        for level in ["attention", "judgment"]:
            print(f"\n--- {level.upper()} challenge ---")
            result = generator(level_theme=level, domain="empirical", difficulty="medium")
            print(f"Proposition: {result.proposition[:80]}...")
            print(f"Correct: {result.correct_judgment}")
            print(f"Vulnerability: {result.vulnerability_explanation[:100]}...")
    else:
        generate_level_challenges(
            output_path=args.output,
            challenges_per_cell=args.per_cell,
        )
