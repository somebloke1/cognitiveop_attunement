#!/usr/bin/env python3
"""
Test Augmented Oracle

Validates that the Lonergan-augmented oracle produces correct judgments
on cases that previously failed (particularly pop science domain).
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import dspy
from src.dspy_cognitive.data_generation import JudgmentGenerator
from src.dspy_cognitive.lonergan_context import (
    get_judgment_context,
    load_full_synopsis,
    JUDGMENT_CONTEXT,
    POP_SCIENCE_CONTEXT,
)


# Test cases that previously failed
TEST_CASES = [
    {
        "name": "Pop Science - Chocolate Memory (previously failed: said Insufficient instead of No)",
        "proposition": "Eating chocolate is proven to improve memory, according to a new study.",
        "evidence": "A 2024 study in the Journal of Nutritional Neuroscience found that participants who ate 50g of dark chocolate daily for 6 weeks showed a 15% improvement in working memory tests compared to baseline. The study had 30 participants, no control group, and was funded by a chocolate industry consortium.",
        "conditions": [
            "Controlled experiment with proper control group",
            "Causal mechanism established (not just correlation)",
            "Results replicated by independent researchers",
            "No conflicts of interest affecting interpretation"
        ],
        "domain": "pop_science",
        "expected_judgment": "No",
        "expected_reason": "The claim of PROOF is false - conditions for causal proof are identifiably unfulfilled (no control group, industry funding, no replication)"
    },
    {
        "name": "Pop Science - Quantum Consciousness (previously failed: said Insufficient instead of No)",
        "proposition": "Quantum effects in microtubules prove consciousness is non-computable, according to prominent scientists.",
        "evidence": "Roger Penrose and Stuart Hameroff have proposed that quantum coherence in neural microtubules could explain consciousness. The theory suggests that quantum gravity effects in microtubules cause 'orchestrated objective reduction' moments of conscious experience. Experimental evidence for quantum coherence in microtubules at brain temperature remains controversial, with some studies showing decoherence times too short for cognitive relevance.",
        "conditions": [
            "Quantum coherence demonstrated at biological temperatures",
            "Causal mechanism connecting quantum effects to consciousness established",
            "Experimental evidence supporting the specific predictions of the theory",
            "Alternative explanations ruled out"
        ],
        "domain": "pop_science",
        "expected_judgment": "No",
        "expected_reason": "The claim of PROOF is false - the theory is speculative, coherence at brain temperature not demonstrated, no experimental proof of the mechanism"
    },
    {
        "name": "Pop Science - Vaccines Autism (clear case)",
        "proposition": "Vaccines have been proven to cause autism, according to research.",
        "evidence": "Andrew Wakefield's 1998 Lancet paper claimed a link between MMR vaccine and autism. The paper was retracted for ethical violations and scientific fraud. Subsequent large-scale studies involving millions of children found no association between vaccines and autism.",
        "conditions": [
            "Valid scientific methodology in original research",
            "Results replicated by independent researchers",
            "No evidence of fraud or ethical violations",
            "No contradicting evidence from larger studies"
        ],
        "domain": "pop_science",
        "expected_judgment": "No",
        "expected_reason": "All conditions are definitively NOT fulfilled - fraud proven, results contradicted by massive subsequent studies"
    },
    {
        "name": "Counterposition - Knowledge as Neural Activity",
        "proposition": "Human knowledge is nothing but electrochemical processes in the brain.",
        "evidence": "Neuroscience has mapped correlations between neural activity and cognitive functions. Brain imaging shows activation patterns during thinking. Damage to specific brain regions impairs specific cognitive abilities.",
        "conditions": [
            "The claim can be affirmed reasonably without presupposing more than neural activity",
            "The act of knowing the claim does not require capacities beyond those the claim allows",
            "The claim is coherent with the performance of affirming it"
        ],
        "domain": "philosophic",
        "expected_judgment": "No",
        "expected_reason": "Self-defeating counterposition - affirming this reasonably presupposes that affirmation is more than 'nothing but' neural activity"
    },
    {
        "name": "Empirical - Replication Crisis Case",
        "proposition": "Power posing increases testosterone and improves performance outcomes.",
        "evidence": "Amy Cuddy's 2010 study claimed power poses increased testosterone by 20% and risk tolerance. The study had 42 participants. Multiple replication attempts with larger samples (n=200+) failed to reproduce the testosterone effect. A 2017 meta-analysis found no reliable evidence for hormonal changes from power posing.",
        "conditions": [
            "Original findings replicated with adequate sample size",
            "Effect persists under methodological scrutiny",
            "Hormonal mechanism confirmed by independent labs"
        ],
        "domain": "empirical",
        "expected_judgment": "No",
        "expected_reason": "Conditions are definitively NOT fulfilled - replications failed, meta-analysis found no effect"
    },
]


def run_tests(use_full_synopsis: bool = False):
    """Run validation tests on the augmented oracle."""

    print("=" * 70)
    print("AUGMENTED ORACLE VALIDATION TEST")
    print("=" * 70)
    print(f"\nUsing full synopsis: {use_full_synopsis}")
    print(f"Lonergan context loaded: {len(JUDGMENT_CONTEXT)} chars (base)")
    print(f"Pop science context: {len(POP_SCIENCE_CONTEXT)} chars")

    if use_full_synopsis:
        synopsis = load_full_synopsis()
        print(f"Full synopsis: {len(synopsis)} chars")

    print("\n" + "-" * 70)

    # Initialize the judgment generator
    generator = JudgmentGenerator(model="gemini/gemini-2.0-flash")

    results = []
    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}/{len(TEST_CASES)}] {test['name']}")
        print(f"  Domain: {test['domain']}")
        print(f"  Expected: {test['expected_judgment']}")
        print(f"  Reason: {test['expected_reason'][:80]}...")

        try:
            result = generator(
                proposition=test['proposition'],
                evidence=test['evidence'],
                conditions=test['conditions'],
                domain=test['domain'],
            )

            actual_judgment = result.judgment
            passed = actual_judgment == test['expected_judgment']

            print(f"\n  RESULT: {actual_judgment}")
            print(f"  STATUS: {'✓ PASSED' if passed else '✗ FAILED'}")

            if hasattr(result, 'reasoning'):
                reasoning_preview = result.reasoning[:200] if len(result.reasoning) > 200 else result.reasoning
                print(f"  Reasoning: {reasoning_preview}...")

            results.append({
                'name': test['name'],
                'expected': test['expected_judgment'],
                'actual': actual_judgment,
                'passed': passed,
                'reasoning': getattr(result, 'reasoning', 'N/A'),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'name': test['name'],
                'expected': test['expected_judgment'],
                'actual': 'ERROR',
                'passed': False,
                'reasoning': str(e),
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r['passed'])
    total = len(results)

    print(f"\nPassed: {passed}/{total} ({100*passed/total:.1f}%)")
    print("\nDetailed Results:")

    for r in results:
        status = "✓" if r['passed'] else "✗"
        print(f"  {status} {r['name'][:50]}: expected={r['expected']}, got={r['actual']}")

    # Previously failed cases
    print("\n" + "-" * 70)
    print("PREVIOUSLY FAILED CASES STATUS:")
    for r in results:
        if "previously failed" in r['name'].lower():
            status = "✓ NOW PASSING" if r['passed'] else "✗ STILL FAILING"
            print(f"  {status}: {r['name']}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test augmented oracle")
    parser.add_argument(
        "--full-synopsis",
        action="store_true",
        help="Use full Lonergan synopsis (more tokens, potentially more aligned)"
    )
    args = parser.parse_args()

    run_tests(use_full_synopsis=args.full_synopsis)
