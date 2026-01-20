#!/usr/bin/env python3
"""
Enhanced Reward Function for Judgment Module Training

This module implements a correctness-dominant reward function with multiplicative
gating. Form without correct judgment is empty form.

Architecture:
    - Correctness (PRIMARY): +0.6 correct, -0.5 wrong, 0.0 no judgment
    - Evidence Grounding (Insight 3): 0.0 to 0.15
    - Domain Verification (Insight 7): 0.0 to 0.10
    - Structure (existing): 0.0 to 0.15

Total range: -0.5 to +1.0

Multiplicative Gating: Bonuses only apply if judgment is correct.
"""

import re
import json
from typing import List, Dict, Any, Optional


# =============================================================================
# DOMAIN VERIFICATION TEMPLATES (Insight 7)
# =============================================================================

VERIFICATION_TEMPLATES = {
    "mathematical": {
        "verification_type": "proof",
        "required_elements": ["axioms_referenced", "logical_steps", "conclusion_marker"],
        "output_markers": ["qed", "therefore", "thus", "it follows that", "by definition", "axiom", "proof", "theorem"],
    },
    "empirical": {
        "verification_type": "experimental",
        "required_elements": ["data_points_cited", "methodology_stated", "replication_status"],
        "output_markers": ["data shows", "measured", "observed", "replicated", "experiment", "evidence shows", "data", "measure", "observe"],
    },
    "common_sense": {
        "verification_type": "pragmatic",
        "required_elements": ["situation_described", "practical_outcome"],
        "output_markers": ["works", "practical", "experience shows", "in this situation", "typically", "experience"],
    },
    "pop_science": {
        "verification_type": "counterposition_detection",
        "required_elements": ["claimed_proof_identified", "conditions_unfulfilled_listed"],
        "output_markers": ["claim unfounded", "correlation not causation", "conditions unfulfilled", "unfulfilled", "alleged", "correlation"],
    },
    "philosophic": {
        "verification_type": "performative_consistency",
        "required_elements": ["position_stated", "self_consistency_check"],
        "output_markers": ["self-defeating", "performative contradiction", "presupposes", "coherent", "counterposition", "performative", "self-consistency"],
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_judgment(completion: str) -> Optional[str]:
    """
    Extract the judgment from a completion using flexible pattern matching.

    Args:
        completion: The model's completion text

    Returns:
        "Yes", "No", "Insufficient", or None if not found.
    """
    if not completion:
        return None

    output_lower = completion.lower()

    # Pattern: "judgment" followed by optional "is" and separator then value
    # Handles: "Judgment: Yes", "Judgment - No", "judgment:\nInsufficient",
    # "The judgment is: Yes", "Final judgment: Yes.", "judgment:yes", etc.
    # The pattern allows for:
    # - Optional whitespace after "judgment"
    # - Optional "is" word
    # - Separators like ":", "-", or just whitespace
    # - Optional whitespace before the value
    pattern = r"judgment(?:\s+is)?[\s:\-]+\s*(yes|no|insufficient)"
    matches = list(re.finditer(pattern, output_lower))

    if matches:
        # Use the last match (in case of multiple judgment markers)
        value = matches[-1].group(1)
        # Normalize to title case
        return value.capitalize()

    return None


def extract_quotes(text: str) -> List[str]:
    """
    Extract quoted text segments from a completion.

    Args:
        text: The text to extract quotes from

    Returns:
        List of quoted strings found in the text
    """
    if not text:
        return []

    # Find text within double quotes
    pattern = r'"([^"]+)"'
    matches = re.findall(pattern, text)
    return matches


# =============================================================================
# COMPONENT SCORING FUNCTIONS
# =============================================================================

def score_correctness(completion: str, expected_judgment: str) -> float:
    """
    Score the correctness of the judgment (PRIMARY signal).

    Args:
        completion: The model's completion text
        expected_judgment: The oracle-verified correct judgment ("Yes"/"No"/"Insufficient")

    Returns:
        +0.6 if judgment matches expected
        -0.5 if judgment doesn't match expected
        0.0 if no judgment detected

    Key Principle: This is the PRIMARY signal. Form without correct
    judgment is empty form.
    """
    # Extract judgment from completion
    model_judgment = extract_judgment(completion)

    # No judgment detected
    if model_judgment is None:
        return 0.0

    # Normalize expected for comparison (case-insensitive)
    expected_normalized = expected_judgment.strip().capitalize() if expected_judgment else ""

    # Compare judgments
    if model_judgment == expected_normalized:
        return 0.6
    else:
        return -0.5


def score_evidence_grounding(
    completion: str,
    input_text: str,
    citations: List[Dict[str, Any]]
) -> float:
    """
    Score evidence grounding (Insight 3).

    Checks:
    1. Does completion contain quotation markers that match input? (+0.05)
    2. Do quoted segments exist in input text? (+0.05)
    3. Are quotes relevant to conditions? (+0.05)

    Note: Hallucinated quotes (not from input) do NOT count.

    Args:
        completion: The model's completion text
        input_text: The original input evidence/proposition
        citations: Expected citations (may be empty)

    Returns:
        Score from 0.0 to 0.15
    """
    if not completion:
        return 0.0

    score = 0.0

    # Extract quotes from completion
    quotes = extract_quotes(completion)

    if not quotes:
        return 0.0

    # Check if ANY quotes exist in input text
    # This is the threshold for getting any credit at all
    input_lower = input_text.lower() if input_text else ""
    matching_quotes = []
    for quote in quotes:
        quote_lower = quote.lower()
        if quote_lower in input_lower:
            matching_quotes.append(quote)

    # If no quotes match the input, this is hallucination - no credit
    if not matching_quotes:
        return 0.0

    # 1. Citation present AND matches input (+0.05)
    score += 0.05

    # 2. Additional bonus for having matching quotes (+0.05)
    # (Already verified by being in matching_quotes)
    score += 0.05

    # 3. Check relevance to conditions (+0.05)
    # If we have explicit citations that match, or if quotes contain
    # condition-relevant language
    condition_relevant = False

    # Check against provided citations
    if citations:
        for citation in citations:
            if isinstance(citation, dict):
                citation_text = citation.get("text", "").lower()
                for quote in matching_quotes:
                    if citation_text in quote.lower() or quote.lower() in citation_text:
                        condition_relevant = True
                        break
            if condition_relevant:
                break

    # If no explicit citations provided, check for condition-related language
    # in the completion context where matching quotes appear
    if not condition_relevant:
        # Check if quotes relate to conditions mentioned in completion
        completion_lower = completion.lower()
        if "condition" in completion_lower or "fulfilled" in completion_lower:
            condition_relevant = True

    if condition_relevant:
        score += 0.05

    return min(0.15, score)


def score_domain_verification(
    completion: str,
    domain: str,
    verification_chain: Dict[str, Any]
) -> float:
    """
    Score domain-specific verification structure (Insight 7).

    Checks if completion follows domain-appropriate verification pattern:
    - Mathematical: proof structure (QED, axioms, steps)
    - Empirical: data reference (measured, observed, data shows)
    - Philosophic: performative assessment (self-defeating, presupposes)
    - Common Sense: practical structure (works, experience)
    - Pop Science: counterposition detection (unfulfilled, correlation)

    Args:
        completion: The model's completion text
        domain: The domain type
        verification_chain: Domain-specific verification requirements

    Returns:
        Score from 0.0 to 0.10
    """
    if not completion or not domain:
        return 0.0

    domain_lower = domain.lower()

    # Check for unknown domain
    if domain_lower not in VERIFICATION_TEMPLATES:
        return 0.0

    completion_lower = completion.lower()

    # Get the template for this domain
    template = VERIFICATION_TEMPLATES[domain_lower]

    # Get output markers - combine template defaults with any provided in verification_chain
    markers = template.get("output_markers", [])
    if verification_chain and "output_markers" in verification_chain:
        markers = list(set(markers + verification_chain["output_markers"]))

    # Count marker matches
    marker_count = 0
    for marker in markers:
        if marker.lower() in completion_lower:
            marker_count += 1

    # Score based on marker presence
    # At least one marker = 0.05, multiple markers = up to 0.10
    if marker_count == 0:
        return 0.0
    elif marker_count == 1:
        return 0.05
    else:
        return 0.10


def score_structure(completion: str) -> float:
    """
    Score structural quality of the completion.

    Checks:
    - Has explicit judgment marker ("Judgment: X")
    - Identifies conditions
    - Shows fulfillment reasoning

    Args:
        completion: The model's completion text

    Returns:
        Score from 0.0 to 0.15
    """
    if not completion:
        return 0.0

    score = 0.0
    completion_lower = completion.lower()

    # 1. Has explicit judgment marker (+0.05)
    if extract_judgment(completion) is not None:
        score += 0.05

    # 2. Identifies conditions (+0.05)
    condition_patterns = [
        r"condition",
        r"conditions",
        r"criterion",
        r"criteria",
    ]
    has_conditions = any(re.search(p, completion_lower) for p in condition_patterns)
    if has_conditions:
        score += 0.05

    # 3. Shows fulfillment reasoning (+0.05)
    fulfillment_patterns = [
        "fulfilled",
        "fulfillment",
        "met",
        "not met",
        "satisfied",
        "therefore",
        "thus",
        "because",
        "since",
        "link",
    ]
    has_fulfillment = any(p in completion_lower for p in fulfillment_patterns)
    if has_fulfillment:
        score += 0.05

    return min(0.15, score)


# =============================================================================
# COMPOSITE REWARD FUNCTION
# =============================================================================

def compute_reward(
    completion: str,
    expected_judgment: str,
    domain: str,
    input_text: str,
    citations: List[Dict[str, Any]],
    verification_chain: Dict[str, Any]
) -> float:
    """
    Compute the composite reward with multiplicative gating.

    Architecture:
        - Correctness: +0.6 (correct) / -0.5 (wrong) / 0.0 (no judgment)
        - Evidence Grounding: 0.0 to 0.15 (gated by correctness)
        - Domain Verification: 0.0 to 0.10 (gated by correctness)
        - Structure: 0.0 to 0.15 (gated by correctness)

    Multiplicative Gating: If correctness <= 0, bonuses are multiplied by 0.
    Wrong judgment cannot achieve positive reward.

    Args:
        completion: The model's completion text
        expected_judgment: The oracle-verified correct judgment
        domain: The domain type
        input_text: The original input evidence/proposition
        citations: Expected citations
        verification_chain: Domain-specific verification requirements

    Returns:
        Reward score from -0.5 to +1.0
    """
    # 1. Get correctness score (PRIMARY)
    correctness = score_correctness(completion, expected_judgment)

    # 2. Determine gating multiplier (1.0 if correct, 0.0 if not)
    gate = 1.0 if correctness > 0 else 0.0

    # 3. Get component scores
    evidence_score = score_evidence_grounding(completion, input_text, citations)
    domain_score = score_domain_verification(completion, domain, verification_chain)
    structure_score = score_structure(completion)

    # 4. Apply gating to bonuses
    bonuses = (evidence_score + domain_score + structure_score) * gate

    # 5. Apply length penalty for very short completions
    length_penalty = -0.1 if completion and len(completion) < 50 else 0.0
    if not completion or len(completion) == 0:
        length_penalty = -0.1

    # 6. Sum and return
    total = correctness + bonuses + length_penalty

    # Clamp to expected range
    return max(-0.5, min(1.0, total))


# =============================================================================
# TRL-COMPATIBLE BATCH REWARD FUNCTION
# =============================================================================

def reward_fn(
    completions: List[str],
    prompts: List[str],
    expected_judgment: List[str] = None,
    domain: List[str] = None,
    input_text: List[str] = None,
    citations: List[str] = None,  # JSON-encoded
    verification_chain: List[str] = None,  # JSON-encoded
    **kwargs
) -> List[float]:
    """
    TRL-compatible reward function for batch processing.

    This is the main entry point for the TRL GRPOTrainer.

    Args:
        completions: List of model completions
        prompts: List of prompts (required by TRL, may not be used)
        expected_judgment: List of oracle-verified correct judgments
        domain: List of domain types
        input_text: List of input evidence/proposition texts
        citations: List of JSON-encoded citation lists
        verification_chain: List of JSON-encoded verification chains
        **kwargs: Additional TRL-provided fields

    Returns:
        List of reward floats, one per completion
    """
    # 1. Handle empty input
    if not completions:
        return []

    n = len(completions)
    rewards = []

    # 2. Handle missing optional kwargs with defaults
    if expected_judgment is None:
        expected_judgment = [""] * n
    if domain is None:
        domain = ["empirical"] * n  # Default domain
    if input_text is None:
        input_text = [""] * n
    if citations is None:
        citations = ["[]"] * n
    if verification_chain is None:
        verification_chain = ["{}"] * n

    for i in range(n):
        # Get values for this item
        completion = completions[i]
        expected = expected_judgment[i] if i < len(expected_judgment) else ""
        dom = domain[i] if i < len(domain) else "empirical"
        inp_text = input_text[i] if i < len(input_text) else ""

        # 3. Decode JSON-encoded fields
        cit_raw = citations[i] if i < len(citations) else "[]"
        chain_raw = verification_chain[i] if i < len(verification_chain) else "{}"

        try:
            if isinstance(cit_raw, str):
                cit = json.loads(cit_raw)
            else:
                cit = cit_raw
        except (json.JSONDecodeError, TypeError):
            cit = []

        try:
            if isinstance(chain_raw, str):
                chain = json.loads(chain_raw)
            else:
                chain = chain_raw
        except (json.JSONDecodeError, TypeError):
            chain = {}

        # 4. Call compute_reward for each item
        reward = compute_reward(
            completion=completion,
            expected_judgment=expected,
            domain=dom,
            input_text=inp_text,
            citations=cit,
            verification_chain=chain
        )

        rewards.append(float(reward))

    # 5. Return list of rewards
    return rewards
