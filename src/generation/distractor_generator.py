"""
Distractor Generator Module for Contrastive Learning.

This module implements Insight 2 from the test architecture:
"Negative Space Definition Through Typed Distractors"

Key principle: Learning what P3 (Judgment) IS requires learning what it is NOT.
Distractors define the negative space of correct cognitive operation.

Distractor Types:
- MISALIGNED_PHASE: P2 (Understanding) output in P3 (Judgment) context
- INVERTED_JUDGMENT: Wrong Yes/No/Insufficient with plausible reasoning
- REASONING_ERROR: Correct judgment with faulty reasoning chain
- EVIDENCE_HALLUCINATION: Cites non-existent evidence
"""

import re
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


class DistractorType(Enum):
    """Types of distractor completions for contrastive learning."""
    MISALIGNED_PHASE = "misaligned_phase"
    INVERTED_JUDGMENT = "inverted_judgment"
    REASONING_ERROR = "reasoning_error"
    EVIDENCE_HALLUCINATION = "evidence_hallucination"
    DOMAIN_MISMATCH = "domain_mismatch"


@dataclass
class JudgmentSample:
    """Judgment sample for distractor generation."""
    proposition: str
    evidence: str
    conditions: List[str]
    conditions_fulfilled: Dict[str, bool]
    judgment: str  # "Yes", "No", "Insufficient"
    judgment_reasoning: str
    domain: str
    difficulty: int = 3


@dataclass
class DistractorSample:
    """A distractor completion with typed error."""
    original_sample: JudgmentSample
    distractor_completion: str
    distractor_type: DistractorType
    error_explanation: str


# =============================================================================
# P2 (Understanding) Language Markers
# =============================================================================

P2_MARKERS = [
    "perhaps",
    "might be",
    "might",
    "could be",
    "could indicate",
    "could",
    "suggests that",
    "possibly",
    "hypothesis",
    "one possibility",
    "it seems",
    "appears to",
    "what if",
    "could mean",
    "may indicate",
    "potentially",
    "speculation",
    "conjecture",
    "?",  # Questions indicate P2 inquiry
]

# =============================================================================
# P3 (Judgment) Language Markers
# =============================================================================

P3_MARKERS = [
    "judgment:",
    "judgment: yes",
    "judgment: no",
    "judgment: insufficient",
    "therefore",
    "thus",
    "consequently",
    "it is the case that",
    "the conditions are",
    "conditions fulfilled",
    "conditions not fulfilled",
    "virtually unconditioned",
    "affirm",
    "deny",
    "withhold",
    "is true",
    "is false",
    "is verified",
    "is falsified",
    "conclude that",
]


class DistractorGenerator:
    """
    Generates typed distractor completions for contrastive learning.

    Key principle: Learn P2/P3 boundary by seeing what P3 is NOT.
    """

    def __init__(self, oracle_model: str = "gemini/gemini-2.0-flash"):
        """Initialize with oracle model for distractor generation."""
        self.oracle_model = oracle_model

    def generate_misaligned_phase(
        self,
        sample: JudgmentSample
    ) -> DistractorSample:
        """
        Generate P2 output for P3 task.

        P2 characteristics:
        - Generates hypotheses, not judgments
        - Offers multiple possibilities
        - No commitment to yes/no/insufficient
        - Asks questions instead of answering
        """
        correct_completion = self._build_correct_completion(sample)
        distractor_text = generate_misaligned_phase_distractor(correct_completion)

        return DistractorSample(
            original_sample=sample,
            distractor_completion=distractor_text,
            distractor_type=DistractorType.MISALIGNED_PHASE,
            error_explanation="Response generates hypotheses (P2) instead of judging (P3). "
                            "Lacks commitment to Yes/No/Insufficient. Asks questions instead of answering."
        )

    def generate_inverted_judgment(
        self,
        sample: JudgmentSample
    ) -> DistractorSample:
        """
        Generate opposite judgment with plausible-sounding reasoning.

        This teaches the model that reasoning must actually support judgment.
        """
        correct_completion = self._build_correct_completion(sample)
        distractor_text = generate_inverted_judgment_distractor(correct_completion)

        return DistractorSample(
            original_sample=sample,
            distractor_completion=distractor_text,
            distractor_type=DistractorType.INVERTED_JUDGMENT,
            error_explanation=f"Original judgment was {sample.judgment}, distractor inverts it "
                            "while maintaining plausible-sounding reasoning structure."
        )

    def generate_reasoning_error(
        self,
        sample: JudgmentSample
    ) -> DistractorSample:
        """
        Generate correct judgment with flawed reasoning.

        Types of reasoning errors:
        - Condition not actually checked
        - Evidence misrepresented
        - Logical non-sequitur
        """
        correct_completion = self._build_correct_completion(sample)
        distractor_text = generate_reasoning_error_distractor(correct_completion)

        return DistractorSample(
            original_sample=sample,
            distractor_completion=distractor_text,
            distractor_type=DistractorType.REASONING_ERROR,
            error_explanation="Correct judgment but flawed reasoning - appeals to authority, "
                            "skips condition checking, or uses logical fallacies."
        )

    def generate_evidence_hallucination(
        self,
        sample: JudgmentSample
    ) -> DistractorSample:
        """
        Generate response that cites non-existent evidence.

        Key for Insight 3 (evidence grounding).
        """
        correct_completion = self._build_correct_completion(sample)
        distractor_text = generate_evidence_hallucination_distractor(
            correct_completion,
            sample.evidence
        )

        return DistractorSample(
            original_sample=sample,
            distractor_completion=distractor_text,
            distractor_type=DistractorType.EVIDENCE_HALLUCINATION,
            error_explanation="Response cites evidence that does not exist in the provided input. "
                            "Fabricated quotes and data points."
        )

    def _build_correct_completion(self, sample: JudgmentSample) -> str:
        """Build a correct completion from a JudgmentSample."""
        lines = ["Condition fulfillment:"]
        for cond, fulfilled in sample.conditions_fulfilled.items():
            status = "Fulfilled" if fulfilled else "Not fulfilled"
            lines.append(f"- {cond}: {status}")

        lines.append("")
        lines.append(sample.judgment_reasoning)
        lines.append("")
        lines.append(f"Judgment: {sample.judgment}")

        return "\n".join(lines)


# =============================================================================
# Standalone Functions (for simpler API)
# =============================================================================

def generate_misaligned_phase_distractor(correct_completion: str) -> str:
    """
    Produce P2 language in P3 context.

    Args:
        correct_completion: A correct P3 judgment completion

    Returns:
        A P2-style completion that lacks firm judgment
    """
    # Extract the topic/content from the correct completion
    lines = correct_completion.split('\n')

    # Build a P2-style response that hedges and explores instead of commits
    p2_response = """The evidence presents several interesting patterns that warrant consideration.

Perhaps this indicates a relationship between the variables, though this would require further investigation. It might be explained by underlying mechanisms we haven't yet identified.

Alternatively, another possibility is that the observed pattern could reflect measurement artifacts rather than a genuine phenomenon. The data might be interpreted in multiple ways.

On the other hand, one could hypothesize that the conditions are met, but this remains to be determined. What additional evidence would help clarify the situation?

These hypotheses suggest several lines of inquiry. The question remains open for further investigation."""

    return p2_response


def generate_inverted_judgment_distractor(correct_completion: str) -> str:
    """
    Flip Yes to No or vice versa.

    Args:
        correct_completion: A correct P3 judgment completion

    Returns:
        A completion with inverted judgment
    """
    # Detect the original judgment
    text_lower = correct_completion.lower()

    # Find the original judgment
    judgment_pattern = r"judgment:\s*(yes|no|insufficient)"
    match = re.search(judgment_pattern, text_lower)

    if match:
        original_judgment = match.group(1).capitalize()
    else:
        # Default assumption
        original_judgment = "Yes"

    # Determine the inverted judgment
    if original_judgment == "Yes":
        inverted_judgment = random.choice(["No", "Insufficient"])
    elif original_judgment == "No":
        inverted_judgment = random.choice(["Yes", "Insufficient"])
    else:  # Insufficient
        inverted_judgment = random.choice(["Yes", "No"])

    # Build inverted response maintaining structure
    if inverted_judgment == "No":
        distractor = """Condition fulfillment:
- Primary condition: Not fulfilled - the evidence does not adequately support the claim
- Secondary condition: Not fulfilled - key requirements are missing
- Verification condition: Not fulfilled - cannot verify the necessary elements

The conditions for affirmation are NOT met. Upon careful reflection, the evidence
fails to establish the required link between the proposition and its fulfillment.

Judgment: No

The proposition is denied because the conditions were not fulfilled in the evidence
provided. The reasoning chain does not establish the virtually unconditioned."""

    elif inverted_judgment == "Insufficient":
        distractor = """Condition fulfillment:
- Primary condition: Cannot determine - evidence is ambiguous
- Secondary condition: Cannot determine - more data needed
- Verification condition: Cannot determine - insufficient information

The virtually unconditioned cannot be grasped. The evidence provided does not
allow for a definitive assessment of whether conditions are fulfilled or not.

Judgment: Insufficient

I withhold judgment because the conditions for determining fulfillment cannot
be verified from the evidence provided. Further investigation is required."""

    else:  # Yes
        distractor = """Condition fulfillment:
- Primary condition: Fulfilled - the evidence adequately supports the claim
- Secondary condition: Fulfilled - all key requirements are present
- Verification condition: Fulfilled - the necessary elements are verified

The conditions for the virtually unconditioned are met. The evidence demonstrates
that all necessary conditions are satisfied.

Judgment: Yes

The proposition is affirmed because all conditions linking it to truth are fulfilled
in the evidence provided. The reasoning chain establishes the virtually unconditioned."""

    return distractor


def generate_reasoning_error_distractor(correct_completion: str) -> str:
    """
    Valid judgment with faulty reasoning.

    Args:
        correct_completion: A correct P3 judgment completion

    Returns:
        Same judgment but with flawed reasoning chain
    """
    # Extract the original judgment to preserve it
    text_lower = correct_completion.lower()
    judgment_pattern = r"judgment:\s*(yes|no|insufficient)"
    match = re.search(judgment_pattern, text_lower)

    if match:
        original_judgment = match.group(1).capitalize()
    else:
        original_judgment = "Yes"

    # Generate faulty reasoning that appeals to authority, tradition, or popularity
    # rather than checking conditions
    faulty_reasonings = [
        f"""The proposition must be correct because:
- This has been widely accepted for a long time
- Many experts in the field believe this to be true
- The argument looks sophisticated and well-formed
- Similar claims have been made before without challenge

The general consensus supports this view. The tradition of scholarship
has long accepted this position, and there is no reason to doubt it now.

Judgment: {original_judgment}

The judgment follows from the weight of authority and established practice.""",

        f"""Considering the evidence presented:
- The source appears credible at first glance
- Other people have reached similar conclusions
- The reasoning seems plausible on the surface
- It would be inconvenient if this were not true

The proposition aligns with common understanding. Since no obvious
contradictions are apparent, the conclusion seems warranted.

Judgment: {original_judgment}

The judgment is based on the general plausibility of the claim.""",

        f"""Analysis of the proposition:
- The claim is stated confidently
- The evidence exists (though not directly verified against conditions)
- Counterarguments were not immediately obvious
- The conclusion feels right intuitively

Since the claim appears reasonable and no immediate objections arise,
we can proceed with confidence in our assessment.

Judgment: {original_judgment}

The judgment follows from intuitive plausibility and lack of obvious error.""",
    ]

    return random.choice(faulty_reasonings)


def generate_evidence_hallucination_distractor(
    correct_completion: str,
    input_text: str
) -> str:
    """
    Cite text not in input.

    Args:
        correct_completion: A correct P3 judgment completion
        input_text: The original input evidence

    Returns:
        A completion that cites non-existent text
    """
    # Extract the original judgment to preserve structure
    text_lower = correct_completion.lower()
    judgment_pattern = r"judgment:\s*(yes|no|insufficient)"
    match = re.search(judgment_pattern, text_lower)

    if match:
        original_judgment = match.group(1).capitalize()
    else:
        original_judgment = "Yes"

    # Generate hallucinated citations that are plausible but not in the input
    hallucinated_quotes = [
        '"the definitive analysis confirms all hypotheses were validated"',
        '"complete verification was achieved across all test conditions"',
        '"the methodology exceeded all established standards of rigor"',
        '"unanimous expert consensus supports these findings"',
    ]

    selected_quotes = random.sample(hallucinated_quotes, min(2, len(hallucinated_quotes)))

    distractor = f"""Condition fulfillment analysis based on the evidence:

The evidence clearly states {selected_quotes[0]}, which directly addresses
the primary condition for judgment.

Furthermore, the text indicates {selected_quotes[1]}, providing additional
support for the assessment of condition fulfillment.

These citations from the evidence establish that the conditions are verified
in the data provided.

Judgment: {original_judgment}

The judgment is supported by the quoted evidence demonstrating fulfillment."""

    return distractor


# =============================================================================
# P2/P3 Language Detection
# =============================================================================

def is_p2_language(text: str) -> bool:
    """
    Detect Understanding/hypothesis language.

    P2 markers: "perhaps", "might be", "could indicate", "suggests that",
    "possibly", "hypothesis", "one possibility", etc.

    Args:
        text: Text to analyze

    Returns:
        True if P2 language markers are present
    """
    text_lower = text.lower()

    # Check for P2 markers
    for marker in P2_MARKERS:
        if marker in text_lower:
            return True

    return False


def is_p3_language(text: str) -> bool:
    """
    Detect Judgment/commitment language.

    P3 markers: "Judgment:", "therefore", "thus", "consequently",
    "it is the case that", "conditions are fulfilled", etc.

    Args:
        text: Text to analyze

    Returns:
        True if P3 language markers are present
    """
    text_lower = text.lower()

    # Check for P3 markers
    for marker in P3_MARKERS:
        if marker in text_lower:
            return True

    return False


# =============================================================================
# Distractor Quality Validation
# =============================================================================

def is_valid_distractor(correct: str, distractor: str) -> bool:
    """
    Check if distractor differs meaningfully from correct completion.

    Rejects:
    - Identical text
    - Whitespace-only differences
    - Trivial word replacements

    Args:
        correct: The correct completion
        distractor: The proposed distractor

    Returns:
        True if distractor is meaningfully different
    """
    # Normalize whitespace for comparison
    correct_normalized = ' '.join(correct.split())
    distractor_normalized = ' '.join(distractor.split())

    # Reject identical (after normalization)
    if correct_normalized == distractor_normalized:
        return False

    # Reject whitespace-only differences
    if correct.strip() == distractor.strip():
        return False

    # Check for meaningful vocabulary differences
    correct_words = set(correct_normalized.lower().split())
    distractor_words = set(distractor_normalized.lower().split())

    # Calculate symmetric difference
    symmetric_diff = correct_words.symmetric_difference(distractor_words)

    # Define trivial words that don't count as meaningful changes
    trivial_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "to", "of", "in", "for", "on", "with", "at", "by"}

    # Filter out trivial word changes
    meaningful_diff = symmetric_diff - trivial_words

    # Require at least 1 meaningful word difference for valid distractor
    # But also check if there are key semantic changes (judgment words)
    key_words = {"yes", "no", "insufficient", "fulfilled", "not", "denied",
                 "affirmed", "true", "false", "verified", "falsified"}

    key_changes = meaningful_diff & key_words

    # Valid if: (1) key judgment words changed, OR (2) significant vocabulary change
    if len(key_changes) >= 1:
        return True

    if len(meaningful_diff) >= 5:
        return True

    # Also reject if only trivial words changed
    if len(symmetric_diff) > 0 and len(meaningful_diff) == 0:
        return False

    # If there are some non-trivial changes, accept
    if len(meaningful_diff) >= 1:
        return True

    return False


def distractor_preserves_structure(correct: str, distractor: str) -> bool:
    """
    Check if distractor maintains structural similarity.

    Checks:
    - Approximate length (lenient for structured content)
    - Similar section markers
    - Not gibberish

    Args:
        correct: The correct completion
        distractor: The proposed distractor

    Returns:
        True if structural similarity is maintained
    """
    correct_len = len(correct)
    distractor_len = len(distractor)

    # Check length ratio - but be more lenient
    if correct_len == 0:
        return distractor_len > 0

    ratio = distractor_len / correct_len

    # Check for structural markers that indicate similar structure
    structure_markers = ["condition", "judgment:", "fulfillment", "fulfilled",
                        "not fulfilled", "evidence", "therefore"]

    distractor_lower = distractor.lower()
    has_structure = any(marker in distractor_lower for marker in structure_markers)

    # If the distractor has structural markers, be more lenient on length
    if has_structure:
        # Accept if within 0.15x to 4x when structured
        if ratio < 0.15 or ratio > 4.0:
            return False
    else:
        # For unstructured, require closer length match
        if ratio < 0.3 or ratio > 3.0:
            return False

    # Check for gibberish - should have reasonable word distribution
    distractor_words = distractor.split()
    if not distractor_words:
        return False

    # Check for repetitive gibberish (same short word repeated)
    unique_words = set(distractor_words)
    if len(unique_words) < 5 and len(distractor_words) > 10:
        return False

    # Check for reasonable word lengths (gibberish often has uniform short words)
    avg_word_len = sum(len(w) for w in distractor_words) / len(distractor_words)
    if avg_word_len < 2:
        return False

    return True


# =============================================================================
# Pipeline Function
# =============================================================================

def generate_distractors_for_example(
    example: JudgmentSample,
    count: int = 2
) -> List[DistractorSample]:
    """
    Generate multiple typed distractors for a training example.

    Args:
        example: The judgment sample to create distractors for
        count: Number of distractors to generate (default 2)

    Returns:
        List of DistractorSample objects with different distractor types
    """
    if count == 0:
        return []

    # Build a correct completion from the sample
    generator = DistractorGenerator()

    # Available distractor generators
    distractor_generators = [
        (DistractorType.MISALIGNED_PHASE, generator.generate_misaligned_phase),
        (DistractorType.INVERTED_JUDGMENT, generator.generate_inverted_judgment),
        (DistractorType.REASONING_ERROR, generator.generate_reasoning_error),
        (DistractorType.EVIDENCE_HALLUCINATION, generator.generate_evidence_hallucination),
    ]

    # Generate distractors, cycling through types for balance
    distractors = []
    for i in range(count):
        # Select distractor type (cycle through for balance)
        distractor_type, gen_func = distractor_generators[i % len(distractor_generators)]

        try:
            distractor = gen_func(example)
            distractors.append(distractor)
        except Exception as e:
            # Fallback to misaligned phase if other types fail
            distractor = generator.generate_misaligned_phase(example)
            distractors.append(distractor)

    return distractors
