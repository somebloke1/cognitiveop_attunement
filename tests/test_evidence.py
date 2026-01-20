"""
TDD Tests for Evidence Grounding (Insight 3: Mandatory Textual Reversion)

These tests define the expected behavior of the evidence grounding module
BEFORE implementation. Tests should FAIL initially.

Evidence Grounding operationalizes Lonergan's principle that "fulfillment is
found by REVERTING from formulation to the more rudimentary state" - every
judgment must cite specific textual evidence from the input.

Tests are organized by functionality:
1. Citation extraction - finding quoted text in completions
2. Citation validation - checking citations against input
3. Evidence grounding score - computing overall grounding quality

The implementation will be at: src/training/evidence_grounding.py
"""

import pytest
from typing import List, Dict, Any, Optional

# These imports will fail until modules are implemented
# This is expected for TDD - we write tests first
try:
    from src.training.evidence_grounding import (
        extract_citations,
        extract_paraphrases,
        citation_exists_in_input,
        citation_approximately_in_input,
        citation_is_relevant,
        compute_evidence_score,
        EvidenceCitation,
    )
    EVIDENCE_MODULE_AVAILABLE = True
except ImportError:
    EVIDENCE_MODULE_AVAILABLE = False
    # Define placeholder types for test discovery
    def extract_citations(completion): raise NotImplementedError
    def extract_paraphrases(completion, input_text): raise NotImplementedError
    def citation_exists_in_input(citation, input_text): raise NotImplementedError
    def citation_approximately_in_input(citation, input_text, threshold=0.85): raise NotImplementedError
    def citation_is_relevant(citation, conditions): raise NotImplementedError
    def compute_evidence_score(completion, input_text, conditions): raise NotImplementedError


# Skip all tests if module not implemented yet (expected for TDD)
pytestmark = pytest.mark.skipif(
    not EVIDENCE_MODULE_AVAILABLE,
    reason="Evidence grounding module not yet implemented (TDD)"
)


# =============================================================================
# PART 1: Citation Extraction Tests
# =============================================================================

class TestExtractCitations:
    """Tests for extract_citations(completion) function."""

    def test_returns_list(self, completion_with_valid_citations):
        """extract_citations should return a list."""
        citations = extract_citations(completion_with_valid_citations)
        assert isinstance(citations, list)

    def test_extracts_double_quoted_text(self):
        """Should extract text within double quotes."""
        completion = 'The evidence states "this is quoted" which proves the point.'
        citations = extract_citations(completion)

        assert "this is quoted" in citations

    def test_extracts_multiple_quotes(self):
        """Should extract all quoted segments."""
        completion = '''First quote: "alpha beta"
        Second quote: "gamma delta"
        Third quote: "epsilon zeta"'''

        citations = extract_citations(completion)

        assert len(citations) >= 3
        assert "alpha beta" in citations
        assert "gamma delta" in citations
        assert "epsilon zeta" in citations

    def test_empty_completion_returns_empty_list(self):
        """Empty completion should return empty list."""
        citations = extract_citations("")
        assert citations == []

    def test_no_quotes_returns_empty_list(self, completion_with_no_citations):
        """Completion without quotes should return empty list."""
        citations = extract_citations(completion_with_no_citations)
        assert citations == []

    def test_preserves_quote_content_exactly(self):
        """Extracted quote should match content exactly."""
        completion = 'The data shows "95.5% correlation with p < 0.001" clearly.'
        citations = extract_citations(completion)

        assert "95.5% correlation with p < 0.001" in citations

    def test_handles_nested_quotes(self):
        """Should handle quotes within quotes reasonably."""
        # Single quotes inside double quotes
        completion = '''The report states "subjects said 'yes' to all questions"'''
        citations = extract_citations(completion)

        assert len(citations) >= 1
        # Should capture the outer quote at minimum
        assert any("subjects said" in c for c in citations)

    def test_handles_multiline_quotes(self):
        """Should handle quotes spanning multiple lines."""
        completion = '''The conclusion reads "The experiment
        demonstrated significant results
        across all conditions" as expected.'''

        citations = extract_citations(completion)
        assert len(citations) >= 1

    def test_extracts_single_word_quotes(self):
        """Should extract single-word quotes."""
        completion = 'The term "significance" was used throughout.'
        citations = extract_citations(completion)

        assert "significance" in citations

    def test_ignores_empty_quotes(self):
        """Should ignore empty quote pairs."""
        completion = 'There was "" nothing between those quotes.'
        citations = extract_citations(completion)

        # Empty string should not be in citations
        assert "" not in citations or len(citations) == 0


class TestExtractParaphrases:
    """Tests for extract_paraphrases(completion, input_text) function."""

    def test_returns_list(self, completion_with_paraphrases, evidence_with_quotable_segments):
        """extract_paraphrases should return a list."""
        paraphrases = extract_paraphrases(
            completion_with_paraphrases,
            evidence_with_quotable_segments
        )
        assert isinstance(paraphrases, list)

    def test_identifies_close_paraphrases(self, evidence_with_quotable_segments):
        """Should identify text that closely paraphrases input."""
        input_text = "The study examined 150 participants over a 6-month period."
        completion = "The research tracked 150 subjects for six months."

        paraphrases = extract_paraphrases(completion, input_text)

        # Should identify the close match
        assert len(paraphrases) >= 1

    def test_paraphrase_includes_similarity_score(self, evidence_with_quotable_segments):
        """Each paraphrase should include a similarity score."""
        completion = "The study tracked 150 subjects for six months."
        paraphrases = extract_paraphrases(completion, evidence_with_quotable_segments)

        if paraphrases:
            # Paraphrases might be dicts or tuples with scores
            first = paraphrases[0]
            if isinstance(first, dict):
                assert 'similarity' in first or 'score' in first
            elif isinstance(first, tuple):
                assert len(first) >= 2  # (text, score)

    def test_empty_completion_returns_empty(self, evidence_with_quotable_segments):
        """Empty completion should return empty list."""
        paraphrases = extract_paraphrases("", evidence_with_quotable_segments)
        assert paraphrases == []

    def test_no_match_returns_empty(self):
        """Unrelated text should return empty list."""
        input_text = "The sky is blue and water is wet."
        completion = "Quantum mechanics explains electron behavior in atoms."

        paraphrases = extract_paraphrases(completion, input_text)
        assert paraphrases == []

    def test_paraphrase_maps_to_input_segment(self, evidence_with_quotable_segments):
        """Each paraphrase should reference the input segment it matches."""
        completion = "The research showed ninety-five percent correlation."
        paraphrases = extract_paraphrases(completion, evidence_with_quotable_segments)

        if paraphrases:
            first = paraphrases[0]
            if isinstance(first, dict):
                assert 'input_segment' in first or 'original' in first


# =============================================================================
# PART 2: Citation Validation Tests
# =============================================================================

class TestCitationExistsInInput:
    """Tests for citation_exists_in_input(citation, input_text) - exact match."""

    def test_exact_match_returns_true(self, simple_evidence):
        """Exact citation should return True."""
        citation = "100 degrees Celsius"
        result = citation_exists_in_input(citation, simple_evidence)
        assert result is True

    def test_no_match_returns_false(self, simple_evidence):
        """Non-existent citation should return False."""
        citation = "200 degrees Fahrenheit"
        result = citation_exists_in_input(citation, simple_evidence)
        assert result is False

    def test_partial_match_returns_true(self, simple_evidence):
        """Partial exact match should return True."""
        citation = "standard pressure"
        result = citation_exists_in_input(citation, simple_evidence)
        assert result is True

    def test_case_sensitivity(self, simple_evidence):
        """Should handle case differences appropriately."""
        # This tests the implementation choice - either case-sensitive or not
        citation = "CELSIUS"  # Original is "Celsius"

        result = citation_exists_in_input(citation, simple_evidence)

        # Behavior depends on implementation - document expectation
        # Most implementations should be case-insensitive for robustness
        assert isinstance(result, bool)

    def test_empty_citation_returns_false(self, simple_evidence):
        """Empty citation should return False."""
        result = citation_exists_in_input("", simple_evidence)
        assert result is False

    def test_whitespace_handling(self):
        """Should handle whitespace variations reasonably."""
        input_text = "The   measurement  showed   results."
        citation = "measurement showed"

        result = citation_exists_in_input(citation, input_text)

        # Should normalize whitespace for matching
        assert isinstance(result, bool)

    def test_punctuation_handling(self):
        """Should handle punctuation at boundaries."""
        input_text = "The result was 'significant'."
        citation = "significant"

        result = citation_exists_in_input(citation, input_text)
        assert result is True


class TestCitationApproximatelyInInput:
    """Tests for citation_approximately_in_input(citation, input_text, threshold)."""

    def test_exact_match_highest_score(self, simple_evidence):
        """Exact match should pass any threshold."""
        citation = "100 degrees Celsius"
        result = citation_approximately_in_input(citation, simple_evidence, threshold=0.99)
        assert result is True

    def test_close_match_passes_threshold(self):
        """Close match should pass reasonable threshold."""
        input_text = "The experiment demonstrated significant results."
        citation = "The experiment showed significant results."  # Minor variation

        result = citation_approximately_in_input(citation, input_text, threshold=0.85)
        assert result is True

    def test_distant_match_fails_threshold(self):
        """Very different text should fail threshold."""
        input_text = "The sky is blue."
        citation = "Water is wet."

        result = citation_approximately_in_input(citation, input_text, threshold=0.85)
        assert result is False

    def test_threshold_parameter_respected(self):
        """Different thresholds should produce different results."""
        input_text = "The temperature was measured at one hundred degrees."
        citation = "temperature measured at 100 degrees"

        # Low threshold should pass
        low_result = citation_approximately_in_input(citation, input_text, threshold=0.5)

        # High threshold may or may not pass depending on similarity
        high_result = citation_approximately_in_input(citation, input_text, threshold=0.95)

        assert isinstance(low_result, bool)
        assert isinstance(high_result, bool)
        # If anything, low threshold should be more permissive
        if high_result:
            assert low_result is True

    def test_default_threshold_reasonable(self, evidence_with_quotable_segments):
        """Default threshold (0.85) should be reasonable for typical use."""
        citation = "95% correlation between exercise frequency and mood improvements"
        result = citation_approximately_in_input(
            citation,
            evidence_with_quotable_segments,
            # Using default threshold
        )
        assert result is True

    def test_returns_boolean(self, simple_evidence):
        """Should return boolean, not similarity score."""
        result = citation_approximately_in_input("anything", simple_evidence)
        assert isinstance(result, bool)


class TestCitationIsRelevant:
    """Tests for citation_is_relevant(citation, conditions) - semantic relevance."""

    def test_relevant_citation_returns_true(self, conditions_list):
        """Citation relevant to conditions should return True."""
        citation = "100 degrees Celsius"
        # First condition mentions temperature >= 100
        result = citation_is_relevant(citation, conditions_list)
        assert result is True

    def test_irrelevant_citation_returns_false(self, conditions_list):
        """Citation irrelevant to conditions should return False."""
        citation = "The sky was cloudy that day."
        result = citation_is_relevant(citation, conditions_list)
        assert result is False

    def test_empty_conditions_returns_true(self):
        """With no conditions, any citation is vacuously relevant."""
        citation = "any citation text"
        result = citation_is_relevant(citation, conditions=[])
        assert result is True

    def test_partial_relevance_counts(self, conditions_list):
        """Citation relevant to some conditions should be relevant."""
        citation = "standard atmospheric pressure"
        # Relevant to condition about pressure
        result = citation_is_relevant(citation, conditions_list)
        assert result is True

    def test_relevance_check_is_semantic(self):
        """Should do semantic matching, not just keyword."""
        conditions = ["The solution must have high acidity (low pH)"]
        citation = "pH measured at 2.3"

        result = citation_is_relevant(citation, conditions)
        # pH 2.3 is low, indicating high acidity
        assert result is True

    def test_returns_boolean(self, conditions_list):
        """Should return boolean."""
        result = citation_is_relevant("any text", conditions_list)
        assert isinstance(result, bool)


# =============================================================================
# PART 3: Evidence Grounding Score Tests
# =============================================================================

class TestComputeEvidenceScore:
    """Tests for compute_evidence_score(completion, input_text, conditions)."""

    def test_returns_float_between_0_and_1(
        self,
        completion_with_valid_citations,
        evidence_with_quotable_segments,
        conditions_list
    ):
        """Score should be float between 0.0 and 1.0."""
        score = compute_evidence_score(
            completion_with_valid_citations,
            evidence_with_quotable_segments,
            conditions_list
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_valid_citations_high_score(
        self,
        completion_with_valid_citations,
        evidence_with_quotable_segments,
        conditions_list
    ):
        """Valid, relevant citations should yield high score."""
        score = compute_evidence_score(
            completion_with_valid_citations,
            evidence_with_quotable_segments,
            conditions_list
        )
        # Full credit requires: citations present, valid, and relevant
        assert score >= 0.5  # Should be substantial

    def test_hallucinated_citations_low_score(
        self,
        completion_with_hallucinated_citations,
        evidence_with_quotable_segments,
        conditions_list
    ):
        """Hallucinated citations should yield low/zero score."""
        score = compute_evidence_score(
            completion_with_hallucinated_citations,
            evidence_with_quotable_segments,
            conditions_list
        )
        # Hallucinated citations should be penalized
        assert score < 0.3

    def test_no_citations_zero_score(
        self,
        completion_with_no_citations,
        evidence_with_quotable_segments,
        conditions_list
    ):
        """No citations should yield zero score."""
        score = compute_evidence_score(
            completion_with_no_citations,
            evidence_with_quotable_segments,
            conditions_list
        )
        assert score == 0.0

    def test_mixed_citations_partial_score(
        self,
        completion_with_mixed_citations,
        evidence_with_quotable_segments,
        conditions_list
    ):
        """Mix of valid and hallucinated should yield partial score."""
        score = compute_evidence_score(
            completion_with_mixed_citations,
            evidence_with_quotable_segments,
            conditions_list
        )
        # Some valid citations = some credit
        # Some invalid = penalty
        assert 0.0 < score < 0.8

    def test_paraphrases_get_partial_credit(
        self,
        completion_with_paraphrases,
        evidence_with_quotable_segments,
        conditions_list
    ):
        """Paraphrases (not exact quotes) should get partial credit."""
        score = compute_evidence_score(
            completion_with_paraphrases,
            evidence_with_quotable_segments,
            conditions_list
        )
        # Paraphrases are less reliable than direct quotes
        # But better than nothing
        assert 0.0 < score < 0.8

    def test_empty_completion_zero_score(
        self,
        evidence_with_quotable_segments,
        conditions_list
    ):
        """Empty completion should yield zero score."""
        score = compute_evidence_score(
            "",
            evidence_with_quotable_segments,
            conditions_list
        )
        assert score == 0.0

    def test_empty_evidence_special_handling(self, conditions_list):
        """With no evidence, citations cannot be valid."""
        completion = 'The data shows "important result" which proves it.'
        score = compute_evidence_score(
            completion,
            input_text="",  # Empty evidence
            conditions=conditions_list
        )
        # Cannot have valid citations if no evidence exists
        assert score == 0.0


# =============================================================================
# PART 4: Domain-Specific Evidence Tests
# =============================================================================

class TestMathematicalDomainEvidence:
    """Tests for evidence grounding in mathematical domain."""

    def test_proof_step_citation(self, mathematical_evidence, conditions_list):
        """Mathematical citations should reference proof steps."""
        evidence = mathematical_evidence["evidence"]
        expected = mathematical_evidence["expected_citation"]

        completion = f'By the proof, "{expected}" therefore the condition is met. Judgment: Yes.'

        score = compute_evidence_score(completion, evidence, conditions_list)
        assert score > 0.4

    def test_axiom_reference_valid(self, mathematical_evidence):
        """Citation of axioms should be valid."""
        evidence = mathematical_evidence["evidence"]
        completion = 'As stated in "Axiom 1: The sum of angles in a Euclidean triangle equals 180 degrees"...'

        # Should find exact match
        result = citation_exists_in_input(
            "The sum of angles in a Euclidean triangle equals 180 degrees",
            evidence
        )
        assert result is True


class TestEmpiricalDomainEvidence:
    """Tests for evidence grounding in empirical domain."""

    def test_measurement_citation(self, empirical_evidence, conditions_list):
        """Empirical citations should reference measurements."""
        evidence = empirical_evidence["evidence"]
        expected = empirical_evidence["expected_citation"]

        completion = f'The observation shows "{expected}" confirming the hypothesis.'

        score = compute_evidence_score(completion, evidence, conditions_list)
        assert score > 0.3

    def test_data_point_reference_valid(self, empirical_evidence):
        """Citation of data points should be valid."""
        evidence = empirical_evidence["evidence"]

        result = citation_exists_in_input("Temperature = 100.0C", evidence)
        assert result is True


class TestCommonSenseDomainEvidence:
    """Tests for evidence grounding in common sense domain."""

    def test_situational_citation(self, common_sense_evidence, conditions_list):
        """Common sense citations should reference situation details."""
        evidence = common_sense_evidence["evidence"]
        expected = common_sense_evidence["expected_citation"]

        completion = f'Given that "{expected}" the practical choice is clear.'

        score = compute_evidence_score(completion, evidence, conditions_list)
        assert score > 0.3


class TestPopScienceDomainEvidence:
    """Tests for evidence grounding in pop science domain."""

    def test_unfulfilled_conditions_cited(self, pop_science_evidence, conditions_list):
        """Pop science detection should cite unfulfilled conditions."""
        evidence = pop_science_evidence["evidence"]
        expected_unfulfilled = pop_science_evidence["expected_unfulfilled"]

        # Good detection cites what's MISSING
        completion = f'''The claim is undermined because "{expected_unfulfilled}"
        which means confounders were not addressed. Judgment: No.'''

        score = compute_evidence_score(completion, evidence, conditions_list)
        assert score > 0.3

    def test_correlation_causation_fallacy_detection(self, pop_science_evidence):
        """Should be able to cite evidence of correlation-causation confusion."""
        evidence = pop_science_evidence["evidence"]

        # The evidence says "correlation is undeniable" but claims causation
        result = citation_exists_in_input("correlation", evidence)
        assert result is True


class TestPhilosophicDomainEvidence:
    """Tests for evidence grounding in philosophic domain."""

    def test_performative_analysis_citation(self, philosophic_evidence, conditions_list):
        """Philosophic citations should reference performative analysis."""
        evidence = philosophic_evidence["evidence"]
        expected = philosophic_evidence["expected_citation"]

        completion = f'The position is self-defeating because "{expected}"'

        score = compute_evidence_score(completion, evidence, conditions_list)
        assert score > 0.3

    def test_counterposition_detection_evidence(self, philosophic_evidence):
        """Should cite evidence of self-referential incoherence."""
        evidence = philosophic_evidence["evidence"]

        result = citation_exists_in_input("self-referentially incoherent", evidence)
        assert result is True


# =============================================================================
# PART 5: Edge Cases
# =============================================================================

class TestEvidenceEdgeCases:
    """Edge case tests for evidence grounding."""

    def test_special_characters_in_citation(self):
        """Should handle special characters in citations."""
        evidence = 'The formula shows: E = mc^2 (Einstein, 1905).'
        completion = 'As the evidence states "E = mc^2" we can conclude...'

        citations = extract_citations(completion)
        assert "E = mc^2" in citations

        result = citation_exists_in_input("E = mc^2", evidence)
        assert result is True

    def test_unicode_in_citation(self):
        """Should handle Unicode characters."""
        evidence = "The Greek letter pi (\u03c0) represents the ratio."
        completion = 'The evidence mentions "\u03c0" as the ratio constant.'

        result = citation_exists_in_input("\u03c0", evidence)
        assert result is True

    def test_very_long_citation(self, evidence_with_quotable_segments):
        """Should handle citations of significant length."""
        # Full sentence citation
        citation = "Results showed a \"95% correlation between exercise frequency and reported mood improvements.\""
        completion = f'The key finding is that {citation}'

        citations = extract_citations(completion)
        assert len(citations) >= 1

    def test_multiple_identical_citations(self, simple_evidence):
        """Should handle same text cited multiple times."""
        completion = '''First, "100 degrees" confirms temperature.
        Again, "100 degrees" is significant.
        And "100 degrees" appears again.'''

        citations = extract_citations(completion)
        # Should find all instances (or dedupe, depending on implementation)
        assert len(citations) >= 1
        assert "100 degrees" in citations

    def test_citation_at_boundary(self):
        """Should handle citations at start/end of text."""
        evidence = "Start. Middle content here. End."

        result_start = citation_exists_in_input("Start", evidence)
        result_end = citation_exists_in_input("End", evidence)

        assert result_start is True
        assert result_end is True

    def test_overlapping_potential_citations(self):
        """Should handle when multiple citation patterns could match."""
        evidence = 'The "quick brown fox" jumps over the "lazy dog".'
        completion = 'Citing "quick brown fox" and "lazy dog" from evidence.'

        citations = extract_citations(completion)
        assert "quick brown fox" in citations
        assert "lazy dog" in citations


class TestEvidenceScoringNuances:
    """Tests for nuanced evidence scoring behavior."""

    def test_relevance_weighted_higher_than_existence(self):
        """Relevant citations should score higher than irrelevant ones."""
        evidence = "Temperature was 100C. The sky was blue that day."
        conditions = ["Temperature must be at least 100C"]

        # Relevant citation
        relevant_completion = 'The evidence shows "Temperature was 100C" meeting the condition.'
        # Irrelevant citation
        irrelevant_completion = 'The evidence mentions "The sky was blue that day" interestingly.'

        relevant_score = compute_evidence_score(relevant_completion, evidence, conditions)
        irrelevant_score = compute_evidence_score(irrelevant_completion, evidence, conditions)

        assert relevant_score > irrelevant_score

    def test_more_citations_not_always_better(self):
        """Quantity of citations shouldn't outweigh quality."""
        evidence = "The result was 95%. The weather was nice. It was Tuesday."
        conditions = ["Result must be above 90%"]

        # One relevant citation
        focused = 'The data shows "result was 95%" confirming the condition.'

        # Many irrelevant citations
        unfocused = '''The evidence states "weather was nice" and "It was Tuesday"
        and we note "The result" appeared.'''

        focused_score = compute_evidence_score(focused, evidence, conditions)
        unfocused_score = compute_evidence_score(unfocused, evidence, conditions)

        assert focused_score >= unfocused_score

    def test_fabricated_but_plausible_citation_penalized(self):
        """Fabricated citations that sound plausible should still be caught."""
        evidence = "The experiment showed 95% success rate."

        # Fabricated but plausible
        completion = 'The experiment showed "96% success rate" which is excellent.'

        score = compute_evidence_score(completion, evidence, conditions=[])
        # 96% is not in the evidence (it says 95%)
        assert score < 0.5


# =============================================================================
# PART 6: Integration Tests
# =============================================================================

class TestEvidenceIntegration:
    """Integration tests for evidence grounding workflow."""

    def test_full_workflow_valid_citations(
        self,
        completion_with_valid_citations,
        evidence_with_quotable_segments,
        conditions_list
    ):
        """Test complete workflow with valid citations."""
        # Step 1: Extract citations
        citations = extract_citations(completion_with_valid_citations)
        assert len(citations) > 0

        # Step 2: Validate each citation
        valid_count = sum(
            1 for c in citations
            if citation_exists_in_input(c, evidence_with_quotable_segments)
        )
        assert valid_count > 0

        # Step 3: Check relevance
        relevant_count = sum(
            1 for c in citations
            if citation_is_relevant(c, conditions_list)
        )
        assert relevant_count >= 0  # May or may not be relevant to these conditions

        # Step 4: Compute overall score
        score = compute_evidence_score(
            completion_with_valid_citations,
            evidence_with_quotable_segments,
            conditions_list
        )
        assert score > 0.4

    def test_full_workflow_hallucinated_citations(
        self,
        completion_with_hallucinated_citations,
        evidence_with_quotable_segments,
        conditions_list
    ):
        """Test complete workflow with hallucinated citations."""
        # Step 1: Extract citations
        citations = extract_citations(completion_with_hallucinated_citations)
        assert len(citations) > 0

        # Step 2: Validate - should find NO valid citations
        valid_count = sum(
            1 for c in citations
            if citation_exists_in_input(c, evidence_with_quotable_segments)
        )
        assert valid_count == 0, "Hallucinated citations should not validate"

        # Step 3: Score should be low
        score = compute_evidence_score(
            completion_with_hallucinated_citations,
            evidence_with_quotable_segments,
            conditions_list
        )
        assert score < 0.3

    def test_reward_function_integration(
        self,
        completion_with_valid_citations,
        completion_with_no_citations,
        evidence_with_quotable_segments,
        conditions_list
    ):
        """Test that evidence score integrates with reward function logic."""
        valid_score = compute_evidence_score(
            completion_with_valid_citations,
            evidence_with_quotable_segments,
            conditions_list
        )

        no_citation_score = compute_evidence_score(
            completion_with_no_citations,
            evidence_with_quotable_segments,
            conditions_list
        )

        # In reward function: evidence_score * 0.2 bonus
        # Valid citations should contribute more than no citations
        valid_reward_contrib = valid_score * 0.2
        no_citation_reward_contrib = no_citation_score * 0.2

        assert valid_reward_contrib > no_citation_reward_contrib
        assert no_citation_reward_contrib == 0.0
