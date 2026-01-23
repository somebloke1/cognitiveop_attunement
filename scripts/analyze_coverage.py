#!/usr/bin/env python3
"""
Analyze coverage of existing training data.

This script:
1. Loads existing JSONL training data
2. Transforms nested metadata to flat format
3. Runs CoverageAnalyzer to identify gaps
4. Produces a detailed report

Usage:
    python scripts/analyze_coverage.py [--data PATH] [--min-count N]
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coverage.analyzer import (
    CoverageAnalyzer,
    build_coverage_matrix,
    find_coverage_gaps,
    get_coverage_percentage,
    get_most_sparse_cells,
    generate_target_specification,
    CoverageGap,
    DOMAIN_GUIDANCE,
)


# =============================================================================
# Data Transformation
# =============================================================================

# Map string difficulties to integers
DIFFICULTY_MAP = {
    "introductory": 1,
    "easy": 2,
    "medium": 3,
    "hard": 4,
    "expert": 5,
}

# Map judgment_type variations to canonical form
JUDGMENT_MAP = {
    "yes": "Yes",
    "no": "No",
    "insufficient": "Insufficient",
    "Yes": "Yes",
    "No": "No",
    "Insufficient": "Insufficient",
}


def transform_example(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw JSONL example to flat format for coverage analysis.

    Input format (from oracle pipeline):
        {
            "messages": [...],
            "metadata": {
                "domain": "mathematical",
                "judgment_type": "Yes",
                "difficulty": "medium",
                ...
            }
        }

    Output format (for CoverageAnalyzer):
        {
            "domain": "mathematical",
            "judgment": "Yes",
            "difficulty": 3,
            "has_distractor": False
        }
    """
    meta = raw.get("metadata", {})

    # Get domain
    domain = meta.get("domain", "empirical")

    # Get and normalize judgment
    jtype = meta.get("judgment_type", "").strip()
    judgment = JUDGMENT_MAP.get(jtype, JUDGMENT_MAP.get(jtype.lower(), "Yes"))

    # Get and convert difficulty
    diff_str = meta.get("difficulty", "medium")
    if isinstance(diff_str, int):
        difficulty = max(1, min(5, diff_str))
    else:
        difficulty = DIFFICULTY_MAP.get(diff_str.lower(), 3)

    # Check for distractor (current data doesn't have this)
    has_distractor = meta.get("has_distractor", False)

    return {
        "domain": domain,
        "judgment": judgment,
        "difficulty": difficulty,
        "has_distractor": has_distractor,
    }


def load_and_transform(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and transform all examples."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                transformed = transform_example(raw)
                examples.append(transformed)
    return examples


# =============================================================================
# Reporting
# =============================================================================

def print_coverage_report(
    matrix: Dict,
    gaps: List[CoverageGap],
    total_examples: int,
    min_count: int
):
    """Print detailed coverage report."""
    print("=" * 70)
    print("COVERAGE ANALYSIS REPORT")
    print("=" * 70)
    print()

    # Calculate basic stats
    total_cells = len(matrix)
    filled_cells = sum(1 for count in matrix.values() if count > 0)
    coverage_pct = (filled_cells / total_cells) * 100.0 if total_cells > 0 else 0.0

    print(f"Total examples:     {total_examples}")
    print(f"Total cells:        {total_cells}")
    print(f"Filled cells:       {filled_cells}")
    print(f"Coverage:           {coverage_pct:.1f}%")
    print(f"Threshold:          {min_count} examples per cell")
    print()

    # Distribution by dimension
    print("-" * 70)
    print("DISTRIBUTION BY DOMAIN")
    print("-" * 70)
    domain_counts = {}
    for (domain, judgment, diff, distractor), count in matrix.items():
        domain_counts[domain] = domain_counts.get(domain, 0) + count
    for domain in sorted(domain_counts.keys()):
        print(f"  {domain:20s}: {domain_counts[domain]:4d}")
    print()

    print("-" * 70)
    print("DISTRIBUTION BY JUDGMENT")
    print("-" * 70)
    judgment_counts = {}
    for (domain, judgment, diff, distractor), count in matrix.items():
        judgment_counts[judgment] = judgment_counts.get(judgment, 0) + count
    for judgment in sorted(judgment_counts.keys()):
        print(f"  {judgment:20s}: {judgment_counts[judgment]:4d}")
    print()

    print("-" * 70)
    print("DISTRIBUTION BY DIFFICULTY")
    print("-" * 70)
    diff_counts = {}
    for (domain, judgment, diff, distractor), count in matrix.items():
        diff_counts[diff] = diff_counts.get(diff, 0) + count
    for diff in sorted(diff_counts.keys()):
        print(f"  Level {diff}:              {diff_counts[diff]:4d}")
    print()

    print("-" * 70)
    print("DISTRIBUTION BY DISTRACTOR STATUS")
    print("-" * 70)
    distractor_counts = {False: 0, True: 0}
    for (domain, judgment, diff, distractor), count in matrix.items():
        distractor_counts[distractor] += count
    print(f"  Positive examples:    {distractor_counts[False]:4d}")
    print(f"  Distractor examples:  {distractor_counts[True]:4d}")
    print()

    # Gaps summary
    print("-" * 70)
    print(f"COVERAGE GAPS (cells with < {min_count} examples)")
    print("-" * 70)
    print(f"Total gaps: {len(gaps)}")
    print()

    # Group gaps by domain
    gaps_by_domain = {}
    for gap in gaps:
        gaps_by_domain.setdefault(gap.domain, []).append(gap)

    for domain in sorted(gaps_by_domain.keys()):
        domain_gaps = gaps_by_domain[domain]
        print(f"\n{domain.upper()} ({len(domain_gaps)} gaps):")

        # Group by distractor status
        positive_gaps = [g for g in domain_gaps if not g.has_distractor]
        distractor_gaps = [g for g in domain_gaps if g.has_distractor]

        if positive_gaps:
            print("  Positive examples needed:")
            for gap in positive_gaps[:5]:  # Show top 5
                print(f"    {gap.judgment:12s} Diff={gap.difficulty}: {gap.current_count}/{gap.target_count}")
            if len(positive_gaps) > 5:
                print(f"    ... and {len(positive_gaps) - 5} more")

        if distractor_gaps:
            print("  Distractor examples needed:")
            for gap in distractor_gaps[:5]:  # Show top 5
                print(f"    {gap.judgment:12s} Diff={gap.difficulty}: {gap.current_count}/{gap.target_count}")
            if len(distractor_gaps) > 5:
                print(f"    ... and {len(distractor_gaps) - 5} more")

    print()

    # Most sparse cells
    print("-" * 70)
    print("TOP 10 MOST SPARSE CELLS")
    print("-" * 70)
    sparse = get_most_sparse_cells(matrix, n=10)
    for i, cell in enumerate(sparse, 1):
        domain, judgment, diff, distractor = cell["cell_id"]
        dtype = "DISTRACTOR" if distractor else "positive"
        print(f"  {i:2d}. {domain:15s} {judgment:12s} D={diff} {dtype:10s}: {cell['count']}")
    print()

    # Generate target specs for worst gaps
    print("-" * 70)
    print("PRIORITY GENERATION TARGETS (for gap filling)")
    print("-" * 70)
    priority_gaps = sorted(gaps, key=lambda g: (-g.priority, g.current_count))[:10]
    for i, gap in enumerate(priority_gaps, 1):
        spec = generate_target_specification(gap)
        dtype = "DISTRACTOR" if spec["has_distractor"] else "positive"
        print(f"\n{i}. {spec['domain']} / {spec['judgment']} / Diff={spec['difficulty']} / {dtype}")
        print(f"   Need: {spec['count_needed']} examples")
        print(f"   Guidance: {spec['domain_guidance'][:60]}...")
        if spec['hints']:
            print(f"   Hint: {spec['hints'][0]}")

    print()
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze training data coverage")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/oracle_generated/judgment_train.jsonl"),
        help="Path to training data JSONL"
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum examples required per cell"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable"
    )
    args = parser.parse_args()

    # Load and transform data
    print(f"Loading data from: {args.data}")
    examples = load_and_transform(args.data)
    print(f"Loaded {len(examples)} examples")
    print()

    # Build coverage matrix
    matrix = build_coverage_matrix(examples)

    # Find gaps
    gaps = find_coverage_gaps(matrix, min_count=args.min_count)

    if args.json:
        # Output as JSON for programmatic use
        result = {
            "total_examples": len(examples),
            "total_cells": len(matrix),
            "filled_cells": sum(1 for c in matrix.values() if c > 0),
            "coverage_percentage": get_coverage_percentage(matrix),
            "gaps": [
                {
                    "domain": g.domain,
                    "judgment": g.judgment,
                    "difficulty": g.difficulty,
                    "has_distractor": g.has_distractor,
                    "current_count": g.current_count,
                    "target_count": g.target_count,
                    "priority": g.priority,
                }
                for g in gaps
            ],
        }
        print(json.dumps(result, indent=2))
    else:
        # Human-readable report
        print_coverage_report(matrix, gaps, len(examples), args.min_count)


if __name__ == "__main__":
    main()
