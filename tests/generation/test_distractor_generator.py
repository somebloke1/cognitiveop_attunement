"""
TDD Tests for Distractor Generator Module.

These tests define the specification for the distractor generation system
that produces contrastive examples for cognitive operation training.

The distractor generator implements Insight 2 from the test architecture:
"Negative Space Definition Through Typed Distractors"

Key principle: Learning what P3 (Judgment) IS requires learning what it is NOT.
Distractors define the negative space of correct cognitive operation.

NOTE: These tests are written BEFORE the implementation exists.
They will FAIL until the module is implemented.
"""

import pytest
import re
from typing import List, Dict, Optional

# Import fixtures from conftest - they define the data structures
from tests.conftest import (
    DistractorType,
    JudgmentSample,
    DistractorSample,
)


# =============================================================================
# Module Under Test (will be implemented in src/generation/distractor_generator.py)
# =============================================================================

# These imports will fail until the module is implemented
try:
    from src.generation.distractor_generator import (
        DistractorGenerator,
        generate_misaligned_phase_distractor,
        generate_inverted_judgment_distractor,
        generate_reasoning_error_distractor,
        generate_evidence_hallucination_distractor,
        is_p2_language,
        is_p3_language,
        is_valid_distractor,
        distractor_preserves_structure,
        generate_distractors_for_example,
    )
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False
    # Placeholder stubs for type hints - tests will skip if not implemented
    DistractorGenerator = None


# =============================================================================
# Skip decorator for unimplemented module
# =============================================================================

requires_implementation = pytest.mark.skipif(
    not MODULE_AVAILABLE,
    reason="DistractorGenerator module not yet implemented"
)


# =============================================================================
# SECTION 1: DISTRACTOR TYPE GENERATION
# =============================================================================

class TestMisalignedPhaseDistractor:
    """
    Tests for generate_misaligned_phase_distractor().

    This distractor produces P2 (Understanding) language in a P3 (Judgment) context.

    P2 characteristics:
    - Generates hypotheses, not judgments
    - Offers multiple possibilities ("could be", "might indicate")
    - No commitment to yes/no/insufficient
    - Asks questions instead of answering
    - Uses hedging language extensively
    """

    @requires_implementation
    def test_misaligned_phase_contains_p2_language(
        self,
        judgment_sample_mathematical_yes,
        p3_judgment_yes_mathematical
    ):
        """Misaligned phase distractor should contain P2 language markers."""
        distractor = generate_misaligned_phase_distractor(p3_judgment_yes_mathematical)

        p2_markers = ["perhaps", "might", "could", "possibly", "hypothesis", "suggests"]
        text_lower = distractor.lower()

        has_p2_markers = any(marker in text_lower for marker in p2_markers)
        assert has_p2_markers, (
            f"Distractor should contain P2 language markers. "
            f"Got: {distractor[:200]}..."
        )

    @requires_implementation
    def test_misaligned_phase_lacks_firm_judgment(
        self,
        p3_judgment_yes_mathematical
    ):
        """Misaligned phase distractor should NOT contain firm judgment statements."""
        distractor = generate_misaligned_phase_distractor(p3_judgment_yes_mathematical)

        # Should not have the judgment pattern
        judgment_pattern = r"judgment:\s*(yes|no|insufficient)"
        match = re.search(judgment_pattern, distractor.lower())

        assert match is None, (
            f"Misaligned phase distractor should NOT have firm judgment. "
            f"Found: {match.group() if match else 'N/A'}"
        )

    @requires_implementation
    def test_misaligned_phase_asks_questions(
        self,
        p3_judgment_no_empirical
    ):
        """Misaligned phase distractor may ask questions rather than commit."""
        distractor = generate_misaligned_phase_distractor(p3_judgment_no_empirical)

        # P2 often ends with questions or open inquiry
        inquiry_markers = ["?", "further investigation", "remains to be", "warrants inquiry"]
        has_inquiry = any(marker in distractor for marker in inquiry_markers)

        # This is a soft assertion - not all P2 outputs ask questions
        # but they should NOT have firm commitments
        text_lower = distractor.lower()
        has_commitment = "judgment: yes" in text_lower or "judgment: no" in text_lower

        assert not has_commitment or has_inquiry, (
            "Misaligned phase should either ask questions or avoid firm commitment"
        )

    @requires_implementation
    def test_misaligned_phase_offers_alternatives(
        self,
        judgment_sample_empirical_no,
        p3_judgment_no_empirical
    ):
        """Misaligned phase distractor should offer multiple possibilities."""
        distractor = generate_misaligned_phase_distractor(p3_judgment_no_empirical)

        alternative_markers = [
            "alternatively", "another possibility", "on the other hand",
            "could also", "or perhaps", "one interpretation", "another view"
        ]
        text_lower = distractor.lower()
        has_alternatives = any(marker in text_lower for marker in alternative_markers)

        # Soft requirement - P2 often presents alternatives
        # Main test is that it doesn't have firm P3 judgment
        judgment_pattern = r"judgment:\s*(yes|no|insufficient)"
        has_judgment = bool(re.search(judgment_pattern, text_lower))

        assert has_alternatives or not has_judgment, (
            "P2 distractor should present alternatives or avoid firm judgment"
        )

    @requires_implementation
    def test_misaligned_phase_returns_distractor_sample(
        self,
        p3_judgment_yes_mathematical
    ):
        """The function should return a properly typed DistractorSample."""
        result = generate_misaligned_phase_distractor(p3_judgment_yes_mathematical)

        # Result should be a string (the distractor completion text)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 50, "Distractor should be substantial"


class TestInvertedJudgmentDistractor:
    """
    Tests for generate_inverted_judgment_distractor().

    This distractor flips Yes to No, No to Yes, or produces a different judgment.
    The key is that it maintains plausible-sounding reasoning while having
    the WRONG judgment.
    """

    @requires_implementation
    def test_inverted_judgment_flips_yes_to_no(
        self,
        p3_judgment_yes_mathematical
    ):
        """Inverted judgment distractor should flip Yes to No or Insufficient."""
        distractor = generate_inverted_judgment_distractor(p3_judgment_yes_mathematical)

        # Extract judgment from distractor
        judgment_pattern = r"judgment:\s*(yes|no|insufficient)"
        match = re.search(judgment_pattern, distractor.lower())

        assert match is not None, "Inverted judgment distractor should contain a judgment"
        extracted_judgment = match.group(1).capitalize()

        # Original was Yes, so inverted should be No or Insufficient
        assert extracted_judgment in ["No", "Insufficient"], (
            f"Original judgment was Yes, inverted should be No or Insufficient, "
            f"got: {extracted_judgment}"
        )

    @requires_implementation
    def test_inverted_judgment_flips_no_to_yes(
        self,
        p3_judgment_no_empirical
    ):
        """Inverted judgment distractor should flip No to Yes or Insufficient."""
        distractor = generate_inverted_judgment_distractor(p3_judgment_no_empirical)

        judgment_pattern = r"judgment:\s*(yes|no|insufficient)"
        match = re.search(judgment_pattern, distractor.lower())

        assert match is not None, "Inverted judgment distractor should contain a judgment"
        extracted_judgment = match.group(1).capitalize()

        # Original was No, so inverted should be Yes or Insufficient
        assert extracted_judgment in ["Yes", "Insufficient"], (
            f"Original judgment was No, inverted should be Yes or Insufficient, "
            f"got: {extracted_judgment}"
        )

    @requires_implementation
    def test_inverted_judgment_has_plausible_reasoning(
        self,
        p3_judgment_yes_mathematical
    ):
        """Inverted judgment should have reasoning that sounds plausible."""
        distractor = generate_inverted_judgment_distractor(p3_judgment_yes_mathematical)

        # Should have condition fulfillment discussion
        reasoning_markers = [
            "condition", "fulfillment", "evidence", "reasoning", "because", "therefore"
        ]
        text_lower = distractor.lower()
        reasoning_count = sum(1 for m in reasoning_markers if m in text_lower)

        assert reasoning_count >= 2, (
            f"Inverted judgment should have plausible reasoning structure. "
            f"Found {reasoning_count} reasoning markers."
        )

    @requires_implementation
    def test_inverted_judgment_maintains_structure(
        self,
        p3_judgment_no_empirical
    ):
        """Inverted judgment should maintain P3 structural elements."""
        distractor = generate_inverted_judgment_distractor(p3_judgment_no_empirical)

        # Should have structure markers
        has_condition_section = "condition" in distractor.lower()
        has_judgment = re.search(r"judgment:", distractor.lower()) is not None

        assert has_condition_section, "Should have condition discussion"
        assert has_judgment, "Should have explicit judgment marker"

    @requires_implementation
    def test_inverted_insufficient_to_yes_or_no(
        self,
        p3_judgment_insufficient_philosophic
    ):
        """Inverted judgment should flip Insufficient to Yes or No."""
        distractor = generate_inverted_judgment_distractor(
            p3_judgment_insufficient_philosophic
        )

        judgment_pattern = r"judgment:\s*(yes|no|insufficient)"
        match = re.search(judgment_pattern, distractor.lower())

        assert match is not None, "Should contain a judgment"
        extracted_judgment = match.group(1).capitalize()

        # Original was Insufficient, so inverted should be Yes or No
        assert extracted_judgment in ["Yes", "No"], (
            f"Original was Insufficient, inverted should be Yes or No, "
            f"got: {extracted_judgment}"
        )


class TestReasoningErrorDistractor:
    """
    Tests for generate_reasoning_error_distractor().

    This distractor produces the CORRECT judgment but with FAULTY reasoning.
    Types of reasoning errors:
    - Condition not actually checked in evidence
    - Evidence misrepresented or misquoted
    - Logical non-sequitur (conclusion doesn't follow from premises)
    - Ignoring unfulfilled conditions
    """

    @requires_implementation
    def test_reasoning_error_has_correct_judgment(
        self,
        p3_judgment_yes_mathematical
    ):
        """Reasoning error distractor should have the CORRECT judgment."""
        distractor = generate_reasoning_error_distractor(p3_judgment_yes_mathematical)

        judgment_pattern = r"judgment:\s*(yes|no|insufficient)"
        match = re.search(judgment_pattern, distractor.lower())

        assert match is not None, "Should contain a judgment"
        # Original was Yes, distractor should also be Yes
        assert match.group(1).lower() == "yes", (
            "Reasoning error distractor should have same judgment as original"
        )

    @requires_implementation
    def test_reasoning_error_has_faulty_logic(
        self,
        p3_judgment_yes_mathematical
    ):
        """Reasoning error distractor should have detectable logical flaw."""
        distractor = generate_reasoning_error_distractor(p3_judgment_yes_mathematical)

        # This is hard to test automatically - we check for common error patterns
        # A well-constructed reasoning error might:
        # 1. Skip checking a condition
        # 2. Claim fulfillment without evidence
        # 3. Use circular reasoning

        # For now, just verify it's different from a valid response
        # and has some content
        assert len(distractor) > 100, "Should be substantial response"

        # Should NOT be identical to correct completion
        # (actual comparison would need the correct completion)

    @requires_implementation
    def test_reasoning_error_maintains_p3_form(
        self,
        p3_judgment_no_empirical
    ):
        """Reasoning error should maintain P3 judgment form."""
        distractor = generate_reasoning_error_distractor(p3_judgment_no_empirical)

        # Should have judgment marker
        assert re.search(r"judgment:", distractor.lower()), (
            "Reasoning error should have P3 form with judgment marker"
        )

        # Should NOT have P2 hedging
        p2_hedges = ["perhaps", "might be", "possibly", "hypothesis"]
        hedge_count = sum(1 for h in p2_hedges if h in distractor.lower())

        # Allow at most 1 hedge (could appear in quoted evidence)
        assert hedge_count <= 1, (
            f"Reasoning error should not hedge like P2. Found {hedge_count} hedges."
        )

    @requires_implementation
    def test_reasoning_error_different_from_correct(
        self,
        p3_judgment_yes_mathematical
    ):
        """Reasoning error distractor must differ from correct completion."""
        correct_completion = p3_judgment_yes_mathematical
        distractor = generate_reasoning_error_distractor(correct_completion)

        # Should not be identical
        assert distractor != correct_completion, (
            "Distractor must be different from correct completion"
        )

        # Should have meaningful differences (not just whitespace)
        correct_words = set(correct_completion.lower().split())
        distractor_words = set(distractor.lower().split())

        # Some difference in vocabulary
        symmetric_diff = correct_words.symmetric_difference(distractor_words)
        assert len(symmetric_diff) >= 5, (
            "Distractor should have meaningful vocabulary differences"
        )


class TestEvidenceHallucinationDistractor:
    """
    Tests for generate_evidence_hallucination_distractor().

    This distractor cites text that does NOT exist in the input evidence.
    Key for testing evidence grounding (Insight 3).
    """

    @requires_implementation
    def test_hallucination_cites_nonexistent_text(
        self,
        p3_judgment_yes_mathematical,
        input_text_mathematical_proof
    ):
        """Hallucination distractor should cite text not in input."""
        distractor = generate_evidence_hallucination_distractor(
            p3_judgment_yes_mathematical,
            input_text_mathematical_proof
        )

        # Extract quoted text from distractor
        quote_pattern = r'"([^"]+)"'
        quotes = re.findall(quote_pattern, distractor)

        # At least one quote should NOT be in the input
        hallucinated_quotes = [
            q for q in quotes
            if q not in input_text_mathematical_proof
        ]

        assert len(hallucinated_quotes) > 0, (
            f"Hallucination distractor should cite nonexistent text. "
            f"Found quotes: {quotes}, all exist in input."
        )

    @requires_implementation
    def test_hallucination_maintains_judgment_structure(
        self,
        p3_judgment_no_empirical,
        input_text_empirical_study
    ):
        """Hallucination distractor should maintain P3 structure."""
        distractor = generate_evidence_hallucination_distractor(
            p3_judgment_no_empirical,
            input_text_empirical_study
        )

        # Should have judgment marker
        assert re.search(r"judgment:", distractor.lower()), (
            "Hallucination distractor should maintain judgment structure"
        )

        # Should have some form of condition discussion
        condition_markers = ["condition", "fulfillment", "evidence", "verified"]
        has_structure = any(m in distractor.lower() for m in condition_markers)
        assert has_structure, "Should maintain structural elements"

    @requires_implementation
    def test_hallucination_produces_plausible_fake_quotes(
        self,
        p3_judgment_yes_mathematical,
        input_text_mathematical_proof
    ):
        """Hallucinated quotes should be domain-appropriate."""
        distractor = generate_evidence_hallucination_distractor(
            p3_judgment_yes_mathematical,
            input_text_mathematical_proof
        )

        # Extract quotes
        quote_pattern = r'"([^"]+)"'
        quotes = re.findall(quote_pattern, distractor)

        # Should have at least one quote (hallucinated or not)
        assert len(quotes) > 0, "Should produce quoted citations"

        # Quotes should be plausible (not obviously nonsense)
        for quote in quotes:
            assert len(quote) >= 5, f"Quote '{quote}' is too short to be plausible"
            assert len(quote) <= 200, f"Quote '{quote}' is suspiciously long"

    @requires_implementation
    def test_hallucination_with_different_domains(
        self,
        judgment_sample_empirical_no,
        input_text_empirical_study
    ):
        """Hallucination should work across different domains."""
        # Create a simple correct completion for the empirical sample
        correct_completion = """Condition fulfillment:
- Study has adequate power: Fulfilled based on n=200
- Statistical significance shown: Fulfilled (p < 0.001)

Judgment: Yes"""

        distractor = generate_evidence_hallucination_distractor(
            correct_completion,
            input_text_empirical_study
        )

        # Should produce output
        assert len(distractor) > 50, "Should produce substantial output"


# =============================================================================
# SECTION 2: P2/P3 MISALIGNMENT DETECTION
# =============================================================================

class TestP2LanguageDetection:
    """
    Tests for is_p2_language() - detects Understanding/hypothesis language.

    P2 (Understanding) language characteristics:
    - Hedging: "perhaps", "might be", "could indicate"
    - Questions: "What if?", "Could this mean?"
    - Multiple possibilities offered
    - Lack of firm commitment
    """

    @requires_implementation
    def test_detects_clear_p2_perhaps(self):
        """Should detect 'perhaps' as P2 language."""
        text = "Perhaps this indicates a correlation between the variables."
        assert is_p2_language(text), f"'Perhaps' should be P2 marker"

    @requires_implementation
    def test_detects_clear_p2_might_be(self):
        """Should detect 'might be' as P2 language."""
        text = "This might be explained by the underlying mechanism."
        assert is_p2_language(text), f"'might be' should be P2 marker"

    @requires_implementation
    def test_detects_clear_p2_suggests_that(self):
        """Should detect 'suggests that' as P2 language."""
        text = "The data suggests that there could be a relationship."
        assert is_p2_language(text), f"'suggests that' should be P2 marker"

    @requires_implementation
    def test_detects_clear_p2_could_indicate(self):
        """Should detect 'could indicate' as P2 language."""
        text = "The pattern could indicate measurement error."
        assert is_p2_language(text), f"'could indicate' should be P2 marker"

    @requires_implementation
    def test_detects_hypothesis_language(self):
        """Should detect explicit hypothesis language."""
        text = "Our hypothesis is that X causes Y, though further testing is needed."
        assert is_p2_language(text), "Hypothesis language is P2"

    @requires_implementation
    def test_detects_question_as_p2(self):
        """Should detect inquiry questions as P2 (not P3 commitment)."""
        text = "What mechanism could explain this distribution?"
        assert is_p2_language(text), "Inquiry questions are P2"

    @requires_implementation
    def test_p2_in_p3_context_still_detected(self, mixed_p2_p3_text):
        """Should detect P2 language even when mixed with P3 markers."""
        # The mixed text has both P2 and P3 language
        assert is_p2_language(mixed_p2_p3_text), (
            "Should detect P2 language even in mixed text"
        )

    @requires_implementation
    def test_pure_p3_not_detected_as_p2(self, p3_judgment_yes_mathematical):
        """Pure P3 completion should NOT be detected as P2."""
        # A well-formed P3 judgment should not have P2 markers
        # This may return True if P3 contains ANY hedging
        # The test is about the detection, not the classification
        result = is_p2_language(p3_judgment_yes_mathematical)
        # Note: pure P3 might still have trace P2 language, so this is
        # more about understanding the detection than a hard assertion
        # The key is that HEAVY P2 is clearly detected
        pass  # Document behavior rather than assert

    @requires_implementation
    def test_multiple_p2_markers_detected(self, p2_understanding_hypothesis):
        """Should detect text with multiple P2 markers."""
        assert is_p2_language(p2_understanding_hypothesis), (
            "Clear P2 text with multiple markers should be detected"
        )


class TestP3LanguageDetection:
    """
    Tests for is_p3_language() - detects Judgment/commitment language.

    P3 (Judgment) language characteristics:
    - Explicit judgment markers: "Judgment: Yes/No/Insufficient"
    - Commitment language: "therefore", "thus", "consequently"
    - Condition fulfillment assessment
    - Virtually unconditioned language
    """

    @requires_implementation
    def test_detects_judgment_yes(self):
        """Should detect 'Judgment: Yes' as P3 language."""
        text = "Based on the evidence, Judgment: Yes"
        assert is_p3_language(text), "'Judgment: Yes' is P3 marker"

    @requires_implementation
    def test_detects_judgment_no(self):
        """Should detect 'Judgment: No' as P3 language."""
        text = "The conditions are not met. Judgment: No"
        assert is_p3_language(text), "'Judgment: No' is P3 marker"

    @requires_implementation
    def test_detects_judgment_insufficient(self):
        """Should detect 'Judgment: Insufficient' as P3 language."""
        text = "Cannot determine fulfillment. Judgment: Insufficient"
        assert is_p3_language(text), "'Judgment: Insufficient' is P3 marker"

    @requires_implementation
    def test_detects_therefore_as_p3(self):
        """Should detect 'therefore' commitment language."""
        text = "All conditions are met, therefore the proposition is true."
        assert is_p3_language(text), "'therefore' indicates P3 commitment"

    @requires_implementation
    def test_detects_thus_as_p3(self):
        """Should detect 'thus' as P3 language."""
        text = "The evidence confirms the hypothesis. Thus, we affirm it."
        assert is_p3_language(text), "'thus' indicates P3 commitment"

    @requires_implementation
    def test_detects_conditions_fulfilled_language(self):
        """Should detect condition fulfillment language."""
        text = "The conditions are fulfilled in the evidence provided."
        assert is_p3_language(text), "Condition fulfillment is P3"

    @requires_implementation
    def test_detects_virtually_unconditioned(self):
        """Should detect 'virtually unconditioned' technical term."""
        text = "The virtually unconditioned has been grasped."
        assert is_p3_language(text), "'virtually unconditioned' is P3 term"

    @requires_implementation
    def test_detects_affirm_deny_language(self):
        """Should detect affirm/deny commitment language."""
        text_affirm = "We affirm the proposition based on fulfilled conditions."
        text_deny = "We deny the proposition; conditions are not met."

        assert is_p3_language(text_affirm), "'affirm' is P3"
        assert is_p3_language(text_deny), "'deny' is P3"

    @requires_implementation
    def test_pure_p2_not_detected_as_p3(self, p2_understanding_hypothesis):
        """Pure P2 text should NOT be detected as P3."""
        assert not is_p3_language(p2_understanding_hypothesis), (
            "P2 hypothesis text should not be detected as P3"
        )

    @requires_implementation
    def test_detects_p3_in_mixed_text(self, mixed_p2_p3_text):
        """Should detect P3 markers even in mixed text."""
        assert is_p3_language(mixed_p2_p3_text), (
            "Should detect P3 language in mixed text"
        )


class TestP2P3Boundary:
    """
    Tests for the critical P2/P3 boundary detection.

    This is the most important operational distinction:
    - P2 offers hypotheses for consideration
    - P3 commits to judgment

    Conflating these is a critical failure mode.
    """

    @requires_implementation
    def test_clear_p2_is_not_p3(self, p2_completions, p3_markers):
        """Clear P2 completions should be P2, not P3."""
        for name, p2_text in p2_completions.items():
            assert is_p2_language(p2_text), f"{name} should be detected as P2"
            # May or may not be detected as P3 depending on content

    @requires_implementation
    def test_clear_p3_is_not_primarily_p2(self, p3_completions, p2_markers):
        """Clear P3 completions should be primarily P3."""
        for name, p3_text in p3_completions.items():
            assert is_p3_language(p3_text), f"{name} should be detected as P3"

    @requires_implementation
    def test_mixed_text_detected(self, mixed_p2_p3_text):
        """Mixed P2/P3 text should be flagged by both detectors."""
        is_p2 = is_p2_language(mixed_p2_p3_text)
        is_p3 = is_p3_language(mixed_p2_p3_text)

        # Both should fire - this is the problematic case
        assert is_p2 and is_p3, (
            "Mixed P2/P3 text should trigger both detectors, "
            "indicating a problematic conflation"
        )

    @requires_implementation
    def test_weak_commitment_detected(self, borderline_p3_weak_commitment):
        """Weak commitment (tentative) should be flagged."""
        # Has P3 structure but weak language
        is_p2 = is_p2_language(borderline_p3_weak_commitment)
        is_p3 = is_p3_language(borderline_p3_weak_commitment)

        # Should have P3 markers (has judgment) but also P2 hedging
        assert is_p3, "Has explicit judgment, so should be P3"
        # The "tentative" and "probably" should trigger P2 detection
        # This is a borderline case worth flagging


# =============================================================================
# SECTION 3: DISTRACTOR QUALITY VALIDATION
# =============================================================================

class TestDistractorValidity:
    """
    Tests for is_valid_distractor() - ensures distractor differs meaningfully.

    A valid distractor must:
    - Be different from the correct completion
    - Not accidentally produce the correct answer
    - Have meaningful differences (not just formatting)
    """

    @requires_implementation
    def test_rejects_identical_distractor(self, p3_judgment_yes_mathematical):
        """Should reject distractor identical to correct completion."""
        correct = p3_judgment_yes_mathematical
        distractor = correct  # Identical

        assert not is_valid_distractor(correct, distractor), (
            "Identical text should not be valid distractor"
        )

    @requires_implementation
    def test_rejects_whitespace_only_difference(self, p3_judgment_yes_mathematical):
        """Should reject distractor differing only in whitespace."""
        correct = p3_judgment_yes_mathematical
        distractor = correct + "   \n\n"  # Only whitespace difference

        assert not is_valid_distractor(correct, distractor), (
            "Whitespace-only difference should not be valid"
        )

    @requires_implementation
    def test_accepts_meaningful_difference(self, p3_judgment_yes_mathematical):
        """Should accept distractor with meaningful content difference."""
        correct = p3_judgment_yes_mathematical
        distractor = correct.replace("Yes", "No").replace(
            "Fulfilled", "Not Fulfilled"
        )

        assert is_valid_distractor(correct, distractor), (
            "Meaningful judgment flip should be valid distractor"
        )

    @requires_implementation
    def test_rejects_trivial_word_replacement(self, p3_judgment_yes_mathematical):
        """Should reject distractor with only trivial word changes."""
        correct = p3_judgment_yes_mathematical
        # Just replace "the" with "a" - trivial change
        distractor = correct.replace(" the ", " a ")

        assert not is_valid_distractor(correct, distractor), (
            "Trivial word replacement should not be valid"
        )

    @requires_implementation
    def test_accepts_p2_distractor_for_p3_correct(
        self,
        p3_judgment_yes_mathematical,
        p2_understanding_hypothesis
    ):
        """P2 output is a valid distractor for P3 correct completion."""
        assert is_valid_distractor(
            p3_judgment_yes_mathematical,
            p2_understanding_hypothesis
        ), "P2 output should be valid distractor for P3 task"


class TestDistractorStructurePreservation:
    """
    Tests for distractor_preserves_structure() - structural similarity check.

    Distractors should:
    - Maintain approximate length
    - Have similar section structure
    - Not be completely unrelated gibberish
    """

    @requires_implementation
    def test_similar_length_preserved(self, p3_judgment_yes_mathematical):
        """Distractor should have similar length to correct."""
        correct = p3_judgment_yes_mathematical
        # Good distractor: similar length
        good_distractor = correct.replace("Yes", "No").replace(
            "Fulfilled", "Not fulfilled"
        )

        assert distractor_preserves_structure(correct, good_distractor), (
            "Similar length distractor should preserve structure"
        )

    @requires_implementation
    def test_rejects_much_shorter(self, p3_judgment_yes_mathematical):
        """Should reject distractor that is much shorter."""
        correct = p3_judgment_yes_mathematical
        short_distractor = "Judgment: No"  # Way too short

        assert not distractor_preserves_structure(correct, short_distractor), (
            "Much shorter distractor should not preserve structure"
        )

    @requires_implementation
    def test_rejects_much_longer(self, p3_judgment_yes_mathematical):
        """Should reject distractor that is much longer."""
        correct = p3_judgment_yes_mathematical
        long_distractor = correct * 5  # Way too long

        assert not distractor_preserves_structure(correct, long_distractor), (
            "Much longer distractor should not preserve structure"
        )

    @requires_implementation
    def test_rejects_gibberish(self, p3_judgment_yes_mathematical):
        """Should reject gibberish distractor."""
        correct = p3_judgment_yes_mathematical
        gibberish = "asdf qwer zxcv " * 50

        assert not distractor_preserves_structure(correct, gibberish), (
            "Gibberish should not preserve structure"
        )

    @requires_implementation
    def test_section_markers_considered(self, p3_judgment_yes_mathematical):
        """Structural similarity should consider section markers."""
        correct = p3_judgment_yes_mathematical

        # Distractor with similar structure (has condition/judgment sections)
        structured_distractor = """Condition fulfillment:
- Condition A: Not fulfilled
- Condition B: Not fulfilled

The conditions are not met.

Judgment: No"""

        # Distractor without structure
        unstructured_distractor = """The data looks interesting but I am not
sure what to make of it. There might be some relationship but who knows.
Maybe further analysis would help. I'll say No I guess."""

        assert distractor_preserves_structure(correct, structured_distractor), (
            "Structured distractor should preserve structure"
        )
        # Unstructured might still pass if length is similar


# =============================================================================
# SECTION 4: DISTRACTOR GENERATION PIPELINE
# =============================================================================

class TestDistractorGenerationPipeline:
    """
    Tests for generate_distractors_for_example() - the main pipeline.

    This function should:
    - Generate specified number of distractors
    - Return typed distractors with explanations
    - Balance distractor types
    """

    @requires_implementation
    def test_generates_specified_count(self, judgment_sample_mathematical_yes):
        """Should generate the specified number of distractors."""
        result = generate_distractors_for_example(
            judgment_sample_mathematical_yes,
            count=2
        )

        assert len(result) == 2, f"Should generate 2 distractors, got {len(result)}"

    @requires_implementation
    def test_generates_different_counts(self, judgment_sample_empirical_no):
        """Should handle different count requests."""
        for count in [1, 2, 3, 4]:
            result = generate_distractors_for_example(
                judgment_sample_empirical_no,
                count=count
            )
            assert len(result) == count, f"Should generate {count} distractors"

    @requires_implementation
    def test_returns_typed_distractors(self, judgment_sample_mathematical_yes):
        """Each distractor should have a type."""
        result = generate_distractors_for_example(
            judgment_sample_mathematical_yes,
            count=3
        )

        for distractor in result:
            assert hasattr(distractor, 'distractor_type') or 'type' in distractor, (
                "Distractor should have a type"
            )

    @requires_implementation
    def test_returns_explanations(self, judgment_sample_empirical_no):
        """Each distractor should have an explanation."""
        result = generate_distractors_for_example(
            judgment_sample_empirical_no,
            count=2
        )

        for distractor in result:
            # Check for explanation attribute or key
            has_explanation = (
                hasattr(distractor, 'error_explanation') or
                hasattr(distractor, 'explanation') or
                (isinstance(distractor, dict) and 'explanation' in distractor)
            )
            assert has_explanation, "Distractor should have explanation"

    @requires_implementation
    def test_balances_distractor_types(self, judgment_sample_philosophic_yes):
        """Should try to balance distractor types when generating multiple."""
        result = generate_distractors_for_example(
            judgment_sample_philosophic_yes,
            count=4
        )

        # Extract types
        types = []
        for d in result:
            if hasattr(d, 'distractor_type'):
                types.append(d.distractor_type)
            elif isinstance(d, dict) and 'type' in d:
                types.append(d['type'])

        # Should have at least 2 different types when generating 4
        unique_types = set(types)
        assert len(unique_types) >= 2, (
            f"Should balance types. Got: {types}"
        )

    @requires_implementation
    def test_links_to_original_sample(self, judgment_sample_mathematical_yes):
        """Distractors should link back to original sample."""
        result = generate_distractors_for_example(
            judgment_sample_mathematical_yes,
            count=1
        )

        distractor = result[0]
        has_link = (
            hasattr(distractor, 'original_sample') or
            (isinstance(distractor, dict) and 'original_sample' in distractor)
        )

        assert has_link, "Distractor should link to original sample"

    @requires_implementation
    def test_handles_edge_case_count_zero(self, judgment_sample_empirical_no):
        """Should handle count=0 gracefully."""
        result = generate_distractors_for_example(
            judgment_sample_empirical_no,
            count=0
        )

        assert result == [], "Count 0 should return empty list"

    @requires_implementation
    def test_handles_all_domains(self, judgment_samples):
        """Should work for all domain types."""
        for name, sample in judgment_samples.items():
            result = generate_distractors_for_example(sample, count=1)
            assert len(result) == 1, f"Should work for {name}"


# =============================================================================
# SECTION 5: INTEGRATION TESTS
# =============================================================================

class TestDistractorGeneratorIntegration:
    """
    Integration tests for the complete distractor generation system.
    """

    @requires_implementation
    def test_full_pipeline_mathematical(
        self,
        judgment_sample_mathematical_yes,
        p3_judgment_yes_mathematical
    ):
        """Full pipeline test for mathematical domain."""
        # Generate distractors
        distractors = generate_distractors_for_example(
            judgment_sample_mathematical_yes,
            count=3
        )

        # All should be valid
        for d in distractors:
            completion = d.distractor_completion if hasattr(d, 'distractor_completion') else d.get('completion', '')
            assert is_valid_distractor(p3_judgment_yes_mathematical, completion)

    @requires_implementation
    def test_full_pipeline_empirical(
        self,
        judgment_sample_empirical_no,
        p3_judgment_no_empirical
    ):
        """Full pipeline test for empirical domain."""
        distractors = generate_distractors_for_example(
            judgment_sample_empirical_no,
            count=2
        )

        # Should produce valid distractors
        assert len(distractors) == 2

    @requires_implementation
    def test_full_pipeline_pop_science(
        self,
        judgment_sample_pop_science_no
    ):
        """Full pipeline test for pop science domain."""
        distractors = generate_distractors_for_example(
            judgment_sample_pop_science_no,
            count=2
        )

        # Pop science distractors might include ones that
        # wrongly affirm the unfounded claim
        types = [
            d.distractor_type if hasattr(d, 'distractor_type') else d.get('type')
            for d in distractors
        ]
        assert len(types) == 2

    @requires_implementation
    def test_distractors_are_distinguishable(self, judgment_sample_empirical_no):
        """Multiple distractors should be distinguishable from each other."""
        distractors = generate_distractors_for_example(
            judgment_sample_empirical_no,
            count=3
        )

        completions = [
            d.distractor_completion if hasattr(d, 'distractor_completion') else d.get('completion', '')
            for d in distractors
        ]

        # All completions should be different
        unique_completions = set(completions)
        assert len(unique_completions) == len(completions), (
            "All distractors should be different from each other"
        )


# =============================================================================
# SECTION 6: EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestDistractorEdgeCases:
    """
    Tests for edge cases and error handling.
    """

    @requires_implementation
    def test_handles_empty_evidence(self, make_judgment_sample):
        """Should handle sample with empty evidence."""
        sample = make_judgment_sample(evidence="")

        # Should not raise exception
        result = generate_distractors_for_example(sample, count=1)
        assert len(result) == 1

    @requires_implementation
    def test_handles_very_short_proposition(self, make_judgment_sample):
        """Should handle very short proposition."""
        sample = make_judgment_sample(proposition="X = Y")

        result = generate_distractors_for_example(sample, count=1)
        assert len(result) == 1

    @requires_implementation
    def test_handles_special_characters(self, make_judgment_sample):
        """Should handle propositions with special characters."""
        sample = make_judgment_sample(
            proposition="If A && B || !C, then D (where D > E * F)"
        )

        result = generate_distractors_for_example(sample, count=1)
        assert len(result) == 1

    @requires_implementation
    def test_consistent_output_type(self, judgment_samples):
        """Output type should be consistent across calls."""
        first_result = None

        for sample in judgment_samples.values():
            result = generate_distractors_for_example(sample, count=1)
            if first_result is None:
                first_result = type(result[0])
            else:
                assert type(result[0]) == first_result, (
                    "Output type should be consistent"
                )
