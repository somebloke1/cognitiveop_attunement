"""
TDD Tests for Extended Schema Module

These tests define the SPECIFICATION for the extended schema. They are written
BEFORE implementation (TDD approach). Tests should FAIL until the schema module
is implemented.

The schema implements data structures for:
1. EnhancedJudgmentSample - Extended judgment sample with all fields
2. DistractorSample - Contrastive distractor completions
3. StudentPacket / EvaluatorPacket - Dual-packet architecture
4. VerificationChain - Domain-specific verification structures
5. CoverageCell - Coverage matrix cell identification

Key Principle: Form without substance is empty form.
These tests ensure the schema operationalizes Lonergan's cognitive theory.
"""

import pytest
from typing import List, Dict, Any, Optional
from dataclasses import FrozenInstanceError


# =============================================================================
# Test Imports - These will fail until modules are implemented
# =============================================================================

class TestSchemaImports:
    """Test that all schema classes can be imported."""

    def test_import_enhanced_judgment_sample(self):
        """EnhancedJudgmentSample should be importable from schema module."""
        try:
            from src.schema.extended_schema import EnhancedJudgmentSample
        except ImportError as e:
            pytest.fail(f"Cannot import EnhancedJudgmentSample: {e}")

    def test_import_evidence_citation(self):
        """EvidenceCitation should be importable from schema module."""
        try:
            from src.schema.extended_schema import EvidenceCitation
        except ImportError as e:
            pytest.fail(f"Cannot import EvidenceCitation: {e}")

    def test_import_distractor_type_enum(self):
        """DistractorType enum should be importable from schema module."""
        try:
            from src.schema.extended_schema import DistractorType
        except ImportError as e:
            pytest.fail(f"Cannot import DistractorType: {e}")

    def test_import_distractor_sample(self):
        """DistractorSample should be importable from schema module."""
        try:
            from src.schema.extended_schema import DistractorSample
        except ImportError as e:
            pytest.fail(f"Cannot import DistractorSample: {e}")

    def test_import_verification_chain(self):
        """DomainVerificationChain should be importable from schema module."""
        try:
            from src.schema.extended_schema import DomainVerificationChain
        except ImportError as e:
            pytest.fail(f"Cannot import DomainVerificationChain: {e}")

    def test_import_student_packet(self):
        """StudentPacket should be importable from schema module."""
        try:
            from src.schema.extended_schema import StudentPacket
        except ImportError as e:
            pytest.fail(f"Cannot import StudentPacket: {e}")

    def test_import_evaluator_packet(self):
        """EvaluatorPacket should be importable from schema module."""
        try:
            from src.schema.extended_schema import EvaluatorPacket
        except ImportError as e:
            pytest.fail(f"Cannot import EvaluatorPacket: {e}")

    def test_import_dual_packet(self):
        """DualPacket should be importable from schema module."""
        try:
            from src.schema.extended_schema import DualPacket
        except ImportError as e:
            pytest.fail(f"Cannot import DualPacket: {e}")

    def test_import_coverage_cell(self):
        """CoverageCell should be importable from schema module."""
        try:
            from src.schema.extended_schema import CoverageCell
        except ImportError as e:
            pytest.fail(f"Cannot import CoverageCell: {e}")


# =============================================================================
# EnhancedJudgmentSample Tests
# =============================================================================

class TestEnhancedJudgmentSample:
    """Tests for the EnhancedJudgmentSample dataclass."""

    # -------------------------------------------------------------------------
    # Field Existence Tests
    # -------------------------------------------------------------------------

    def test_has_required_core_fields(self, sample_factory):
        """EnhancedJudgmentSample must have all core fields."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory()
        sample = EnhancedJudgmentSample(
            proposition=sample_data["proposition"],
            domain=sample_data["domain"],
            evidence_text=sample_data["evidence_text"],
            conditions=sample_data["conditions"],
            expected_judgment=sample_data["expected_judgment"],
            difficulty=sample_data["difficulty"],
        )

        # Core fields must exist
        assert hasattr(sample, "proposition")
        assert hasattr(sample, "domain")
        assert hasattr(sample, "evidence_text")
        assert hasattr(sample, "conditions")
        assert hasattr(sample, "expected_judgment")

    def test_has_difficulty_field(self, sample_factory):
        """EnhancedJudgmentSample must have difficulty field (1-5)."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(difficulty=3)
        sample = EnhancedJudgmentSample(**sample_data)

        assert hasattr(sample, "difficulty")
        assert sample.difficulty == 3

    def test_has_evidence_citations_field(self, sample_factory):
        """EnhancedJudgmentSample must have evidence_citations list."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        citations = [{"text": "test citation", "start_idx": 0, "end_idx": 13}]
        sample_data = sample_factory(evidence_citations=citations)
        sample = EnhancedJudgmentSample(**sample_data)

        assert hasattr(sample, "evidence_citations")
        assert isinstance(sample.evidence_citations, list)

    def test_has_verification_chain_field(self, sample_factory):
        """EnhancedJudgmentSample must have verification_chain field."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        chain = {"domain": "empirical", "verification_type": "experimental"}
        sample_data = sample_factory(verification_chain=chain)
        sample = EnhancedJudgmentSample(**sample_data)

        assert hasattr(sample, "verification_chain")

    def test_has_distractor_type_field(self, sample_factory):
        """EnhancedJudgmentSample must have optional distractor_type field."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        # Without distractor
        sample_data = sample_factory(distractor_type=None)
        sample = EnhancedJudgmentSample(**sample_data)
        assert hasattr(sample, "distractor_type")
        assert sample.distractor_type is None

        # With distractor
        sample_data = sample_factory(distractor_type="inverted_judgment")
        sample = EnhancedJudgmentSample(**sample_data)
        assert sample.distractor_type == "inverted_judgment"

    # -------------------------------------------------------------------------
    # Difficulty Validation Tests
    # -------------------------------------------------------------------------

    def test_difficulty_accepts_valid_values(self, sample_factory):
        """Difficulty field must accept values 1-5."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        for level in [1, 2, 3, 4, 5]:
            sample_data = sample_factory(difficulty=level)
            sample = EnhancedJudgmentSample(**sample_data)
            assert sample.difficulty == level

    def test_difficulty_rejects_zero(self, sample_factory):
        """Difficulty field must reject 0."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(difficulty=0)
        with pytest.raises((ValueError, TypeError)):
            EnhancedJudgmentSample(**sample_data)

    def test_difficulty_rejects_negative(self, sample_factory):
        """Difficulty field must reject negative values."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(difficulty=-1)
        with pytest.raises((ValueError, TypeError)):
            EnhancedJudgmentSample(**sample_data)

    def test_difficulty_rejects_above_five(self, sample_factory):
        """Difficulty field must reject values > 5."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(difficulty=6)
        with pytest.raises((ValueError, TypeError)):
            EnhancedJudgmentSample(**sample_data)

    def test_difficulty_rejects_float(self, sample_factory):
        """Difficulty field must reject non-integer values."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(difficulty=2.5)
        with pytest.raises((ValueError, TypeError)):
            EnhancedJudgmentSample(**sample_data)

    # -------------------------------------------------------------------------
    # Domain Validation Tests
    # -------------------------------------------------------------------------

    def test_domain_accepts_mathematical(self, sample_factory):
        """Domain field must accept 'mathematical'."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(domain="mathematical")
        sample = EnhancedJudgmentSample(**sample_data)
        assert sample.domain == "mathematical"

    def test_domain_accepts_empirical(self, sample_factory):
        """Domain field must accept 'empirical'."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(domain="empirical")
        sample = EnhancedJudgmentSample(**sample_data)
        assert sample.domain == "empirical"

    def test_domain_accepts_common_sense(self, sample_factory):
        """Domain field must accept 'common_sense'."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(domain="common_sense")
        sample = EnhancedJudgmentSample(**sample_data)
        assert sample.domain == "common_sense"

    def test_domain_accepts_pop_science(self, sample_factory):
        """Domain field must accept 'pop_science'."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(domain="pop_science")
        sample = EnhancedJudgmentSample(**sample_data)
        assert sample.domain == "pop_science"

    def test_domain_accepts_philosophic(self, sample_factory):
        """Domain field must accept 'philosophic'."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(domain="philosophic")
        sample = EnhancedJudgmentSample(**sample_data)
        assert sample.domain == "philosophic"

    def test_domain_rejects_invalid_value(self, sample_factory):
        """Domain field must reject invalid values."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(domain="invalid_domain")
        with pytest.raises((ValueError, TypeError)):
            EnhancedJudgmentSample(**sample_data)

    def test_domain_rejects_empty_string(self, sample_factory):
        """Domain field must reject empty string."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(domain="")
        with pytest.raises((ValueError, TypeError)):
            EnhancedJudgmentSample(**sample_data)

    def test_domain_is_case_sensitive(self, sample_factory):
        """Domain field should be lowercase (case sensitive)."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        # Uppercase should be rejected or normalized
        sample_data = sample_factory(domain="EMPIRICAL")
        with pytest.raises((ValueError, TypeError)):
            EnhancedJudgmentSample(**sample_data)

    # -------------------------------------------------------------------------
    # Judgment Validation Tests
    # -------------------------------------------------------------------------

    def test_judgment_accepts_yes(self, sample_factory):
        """expected_judgment field must accept 'yes'."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(expected_judgment="yes")
        sample = EnhancedJudgmentSample(**sample_data)
        assert sample.expected_judgment == "yes"

    def test_judgment_accepts_no(self, sample_factory):
        """expected_judgment field must accept 'no'."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(expected_judgment="no")
        sample = EnhancedJudgmentSample(**sample_data)
        assert sample.expected_judgment == "no"

    def test_judgment_accepts_insufficient(self, sample_factory):
        """expected_judgment field must accept 'insufficient'."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(expected_judgment="insufficient")
        sample = EnhancedJudgmentSample(**sample_data)
        assert sample.expected_judgment == "insufficient"

    def test_judgment_rejects_invalid_value(self, sample_factory):
        """expected_judgment field must reject invalid values."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(expected_judgment="maybe")
        with pytest.raises((ValueError, TypeError)):
            EnhancedJudgmentSample(**sample_data)

    def test_judgment_rejects_empty_string(self, sample_factory):
        """expected_judgment field must reject empty string."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample_data = sample_factory(expected_judgment="")
        with pytest.raises((ValueError, TypeError)):
            EnhancedJudgmentSample(**sample_data)

    # -------------------------------------------------------------------------
    # Full Sample Construction Tests
    # -------------------------------------------------------------------------

    def test_construct_from_mathematical_sample(self, mathematical_sample_data):
        """Should construct EnhancedJudgmentSample from mathematical domain data."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample = EnhancedJudgmentSample(**mathematical_sample_data)

        assert sample.proposition == mathematical_sample_data["proposition"]
        assert sample.domain == "mathematical"
        assert sample.expected_judgment == "yes"
        assert sample.difficulty == 2

    def test_construct_from_empirical_sample(self, empirical_sample_data):
        """Should construct EnhancedJudgmentSample from empirical domain data."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample = EnhancedJudgmentSample(**empirical_sample_data)

        assert sample.domain == "empirical"
        assert sample.expected_judgment == "yes"
        assert len(sample.evidence_citations) == 2

    def test_construct_from_pop_science_sample(self, pop_science_sample_data):
        """Should construct EnhancedJudgmentSample from pop_science domain data."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample = EnhancedJudgmentSample(**pop_science_sample_data)

        assert sample.domain == "pop_science"
        assert sample.expected_judgment == "no"

    def test_construct_from_philosophic_sample(self, philosophic_sample_data):
        """Should construct EnhancedJudgmentSample from philosophic domain data."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample = EnhancedJudgmentSample(**philosophic_sample_data)

        assert sample.domain == "philosophic"
        assert sample.expected_judgment == "no"
        assert sample.difficulty == 4


# =============================================================================
# EvidenceCitation Tests
# =============================================================================

class TestEvidenceCitation:
    """Tests for the EvidenceCitation dataclass."""

    def test_has_required_fields(self):
        """EvidenceCitation must have text, start_idx, end_idx, supports_condition."""
        from src.schema.extended_schema import EvidenceCitation

        citation = EvidenceCitation(
            text="test citation",
            start_idx=10,
            end_idx=23,
            supports_condition="Condition 1"
        )

        assert hasattr(citation, "text")
        assert hasattr(citation, "start_idx")
        assert hasattr(citation, "end_idx")
        assert hasattr(citation, "supports_condition")

    def test_indices_must_be_non_negative(self):
        """Citation indices must be non-negative."""
        from src.schema.extended_schema import EvidenceCitation

        with pytest.raises((ValueError, TypeError)):
            EvidenceCitation(
                text="test",
                start_idx=-1,
                end_idx=10,
                supports_condition="Condition 1"
            )

    def test_end_idx_must_be_greater_than_start_idx(self):
        """end_idx must be > start_idx."""
        from src.schema.extended_schema import EvidenceCitation

        with pytest.raises((ValueError, TypeError)):
            EvidenceCitation(
                text="test",
                start_idx=10,
                end_idx=5,  # Invalid: end < start
                supports_condition="Condition 1"
            )

    def test_text_length_matches_index_span(self):
        """Citation text length should match index span."""
        from src.schema.extended_schema import EvidenceCitation

        citation = EvidenceCitation(
            text="test citation",
            start_idx=10,
            end_idx=23,
            supports_condition="Condition 1"
        )

        # Length of "test citation" = 13
        # 23 - 10 = 13
        assert len(citation.text) == citation.end_idx - citation.start_idx

    def test_verify_citation_against_evidence(self):
        """Citation should be verifiable against source evidence."""
        from src.schema.extended_schema import EvidenceCitation

        evidence = "The experiment showed 95% correlation between variables."
        citation = EvidenceCitation(
            text="95% correlation",
            start_idx=22,
            end_idx=37,
            supports_condition="Correlation exists"
        )

        # The citation text should match the evidence at specified indices
        assert evidence[citation.start_idx:citation.end_idx] == citation.text


# =============================================================================
# DistractorType Enum Tests
# =============================================================================

class TestDistractorType:
    """Tests for the DistractorType enum."""

    def test_has_misaligned_phase(self):
        """DistractorType must have MISALIGNED_PHASE."""
        from src.schema.extended_schema import DistractorType

        assert hasattr(DistractorType, "MISALIGNED_PHASE")
        assert DistractorType.MISALIGNED_PHASE.value == "misaligned_phase"

    def test_has_inverted_judgment(self):
        """DistractorType must have INVERTED_JUDGMENT."""
        from src.schema.extended_schema import DistractorType

        assert hasattr(DistractorType, "INVERTED_JUDGMENT")
        assert DistractorType.INVERTED_JUDGMENT.value == "inverted_judgment"

    def test_has_reasoning_error(self):
        """DistractorType must have REASONING_ERROR."""
        from src.schema.extended_schema import DistractorType

        assert hasattr(DistractorType, "REASONING_ERROR")
        assert DistractorType.REASONING_ERROR.value == "reasoning_error"

    def test_has_evidence_hallucination(self):
        """DistractorType must have EVIDENCE_HALLUCINATION."""
        from src.schema.extended_schema import DistractorType

        assert hasattr(DistractorType, "EVIDENCE_HALLUCINATION")
        assert DistractorType.EVIDENCE_HALLUCINATION.value == "evidence_hallucination"

    def test_has_domain_mismatch(self):
        """DistractorType must have DOMAIN_MISMATCH."""
        from src.schema.extended_schema import DistractorType

        assert hasattr(DistractorType, "DOMAIN_MISMATCH")
        assert DistractorType.DOMAIN_MISMATCH.value == "domain_mismatch"

    def test_enum_has_five_types(self):
        """DistractorType should have exactly 5 distractor types."""
        from src.schema.extended_schema import DistractorType

        assert len(DistractorType) == 5


# =============================================================================
# DistractorSample Tests
# =============================================================================

class TestDistractorSample:
    """Tests for the DistractorSample dataclass."""

    def test_has_required_fields(self):
        """DistractorSample must have all required fields."""
        from src.schema.extended_schema import DistractorSample, DistractorType

        sample = DistractorSample(
            correct_completion="Judgment: Yes",
            distractor_completion="This might be true...",
            distractor_type=DistractorType.MISALIGNED_PHASE,
            explanation="P2 response to P3 task"
        )

        assert hasattr(sample, "correct_completion")
        assert hasattr(sample, "distractor_completion")
        assert hasattr(sample, "distractor_type")
        assert hasattr(sample, "explanation")

    def test_distractor_type_must_be_enum(self):
        """distractor_type must be a DistractorType enum value."""
        from src.schema.extended_schema import DistractorSample, DistractorType

        sample = DistractorSample(
            correct_completion="Judgment: Yes",
            distractor_completion="This might be true...",
            distractor_type=DistractorType.MISALIGNED_PHASE,
            explanation="P2 response to P3 task"
        )

        assert isinstance(sample.distractor_type, DistractorType)

    def test_construct_misaligned_phase_distractor(self, misaligned_phase_distractor):
        """Should construct DistractorSample for misaligned phase."""
        from src.schema.extended_schema import DistractorSample, DistractorType

        sample = DistractorSample(
            correct_completion=misaligned_phase_distractor["correct_completion"],
            distractor_completion=misaligned_phase_distractor["distractor_completion"],
            distractor_type=DistractorType.MISALIGNED_PHASE,
            explanation=misaligned_phase_distractor["explanation"]
        )

        assert sample.distractor_type == DistractorType.MISALIGNED_PHASE

    def test_construct_inverted_judgment_distractor(self, inverted_judgment_distractor):
        """Should construct DistractorSample for inverted judgment."""
        from src.schema.extended_schema import DistractorSample, DistractorType

        sample = DistractorSample(
            correct_completion=inverted_judgment_distractor["correct_completion"],
            distractor_completion=inverted_judgment_distractor["distractor_completion"],
            distractor_type=DistractorType.INVERTED_JUDGMENT,
            explanation=inverted_judgment_distractor["explanation"]
        )

        assert sample.distractor_type == DistractorType.INVERTED_JUDGMENT

    def test_construct_evidence_hallucination_distractor(self, evidence_hallucination_distractor):
        """Should construct DistractorSample for evidence hallucination."""
        from src.schema.extended_schema import DistractorSample, DistractorType

        sample = DistractorSample(
            correct_completion=evidence_hallucination_distractor["correct_completion"],
            distractor_completion=evidence_hallucination_distractor["distractor_completion"],
            distractor_type=DistractorType.EVIDENCE_HALLUCINATION,
            explanation=evidence_hallucination_distractor["explanation"]
        )

        assert sample.distractor_type == DistractorType.EVIDENCE_HALLUCINATION

    def test_construct_reasoning_error_distractor(self, reasoning_error_distractor):
        """Should construct DistractorSample for reasoning error."""
        from src.schema.extended_schema import DistractorSample, DistractorType

        sample = DistractorSample(
            correct_completion=reasoning_error_distractor["correct_completion"],
            distractor_completion=reasoning_error_distractor["distractor_completion"],
            distractor_type=DistractorType.REASONING_ERROR,
            explanation=reasoning_error_distractor["explanation"]
        )

        assert sample.distractor_type == DistractorType.REASONING_ERROR


# =============================================================================
# StudentPacket Tests
# =============================================================================

class TestStudentPacket:
    """Tests for the StudentPacket dataclass (what model sees)."""

    def test_has_prompt_field(self):
        """StudentPacket must have prompt field."""
        from src.schema.extended_schema import StudentPacket

        packet = StudentPacket(
            prompt="Test prompt",
            evidence="Test evidence",
            conditions=["Condition 1", "Condition 2"]
        )

        assert hasattr(packet, "prompt")
        assert packet.prompt == "Test prompt"

    def test_has_evidence_field(self):
        """StudentPacket must have evidence field."""
        from src.schema.extended_schema import StudentPacket

        packet = StudentPacket(
            prompt="Test prompt",
            evidence="Test evidence",
            conditions=["Condition 1"]
        )

        assert hasattr(packet, "evidence")
        assert packet.evidence == "Test evidence"

    def test_has_conditions_field(self):
        """StudentPacket must have conditions field (list)."""
        from src.schema.extended_schema import StudentPacket

        conditions = ["Condition 1", "Condition 2"]
        packet = StudentPacket(
            prompt="Test prompt",
            evidence="Test evidence",
            conditions=conditions
        )

        assert hasattr(packet, "conditions")
        assert packet.conditions == conditions

    def test_does_not_have_expected_judgment(self):
        """StudentPacket must NOT have expected_judgment field.

        The model should not see the answer during generation.
        """
        from src.schema.extended_schema import StudentPacket

        packet = StudentPacket(
            prompt="Test prompt",
            evidence="Test evidence",
            conditions=["Condition 1"]
        )

        assert not hasattr(packet, "expected_judgment")

    def test_does_not_have_evaluator_fields(self, evaluator_packet_fields):
        """StudentPacket must NOT have any evaluator fields."""
        from src.schema.extended_schema import StudentPacket

        packet = StudentPacket(
            prompt="Test prompt",
            evidence="Test evidence",
            conditions=["Condition 1"]
        )

        for field in evaluator_packet_fields:
            assert not hasattr(packet, field), f"StudentPacket should not have {field}"


# =============================================================================
# EvaluatorPacket Tests
# =============================================================================

class TestEvaluatorPacket:
    """Tests for the EvaluatorPacket dataclass (what reward function sees)."""

    def test_has_expected_judgment(self):
        """EvaluatorPacket must have expected_judgment field."""
        from src.schema.extended_schema import EvaluatorPacket

        packet = EvaluatorPacket(
            expected_judgment="yes",
            domain="empirical",
            difficulty=2,
            evidence_citations=[],
            distractor_type=None,
            verification_chain={}
        )

        assert hasattr(packet, "expected_judgment")
        assert packet.expected_judgment == "yes"

    def test_has_domain(self):
        """EvaluatorPacket must have domain field."""
        from src.schema.extended_schema import EvaluatorPacket

        packet = EvaluatorPacket(
            expected_judgment="yes",
            domain="mathematical",
            difficulty=2,
            evidence_citations=[],
            distractor_type=None,
            verification_chain={}
        )

        assert hasattr(packet, "domain")
        assert packet.domain == "mathematical"

    def test_has_difficulty(self):
        """EvaluatorPacket must have difficulty field."""
        from src.schema.extended_schema import EvaluatorPacket

        packet = EvaluatorPacket(
            expected_judgment="yes",
            domain="empirical",
            difficulty=3,
            evidence_citations=[],
            distractor_type=None,
            verification_chain={}
        )

        assert hasattr(packet, "difficulty")
        assert packet.difficulty == 3

    def test_has_evidence_citations(self):
        """EvaluatorPacket must have evidence_citations field."""
        from src.schema.extended_schema import EvaluatorPacket

        citations = [{"text": "test", "start_idx": 0, "end_idx": 4}]
        packet = EvaluatorPacket(
            expected_judgment="yes",
            domain="empirical",
            difficulty=2,
            evidence_citations=citations,
            distractor_type=None,
            verification_chain={}
        )

        assert hasattr(packet, "evidence_citations")
        assert packet.evidence_citations == citations

    def test_has_distractor_type(self):
        """EvaluatorPacket must have distractor_type field (optional)."""
        from src.schema.extended_schema import EvaluatorPacket

        packet = EvaluatorPacket(
            expected_judgment="yes",
            domain="empirical",
            difficulty=2,
            evidence_citations=[],
            distractor_type="inverted_judgment",
            verification_chain={}
        )

        assert hasattr(packet, "distractor_type")
        assert packet.distractor_type == "inverted_judgment"

    def test_has_verification_chain(self):
        """EvaluatorPacket must have verification_chain field."""
        from src.schema.extended_schema import EvaluatorPacket

        chain = {"domain": "mathematical", "verification_type": "proof"}
        packet = EvaluatorPacket(
            expected_judgment="yes",
            domain="mathematical",
            difficulty=2,
            evidence_citations=[],
            distractor_type=None,
            verification_chain=chain
        )

        assert hasattr(packet, "verification_chain")
        assert packet.verification_chain == chain

    def test_does_not_have_prompt(self):
        """EvaluatorPacket must NOT have prompt field.

        The prompt is in StudentPacket, not EvaluatorPacket.
        """
        from src.schema.extended_schema import EvaluatorPacket

        packet = EvaluatorPacket(
            expected_judgment="yes",
            domain="empirical",
            difficulty=2,
            evidence_citations=[],
            distractor_type=None,
            verification_chain={}
        )

        assert not hasattr(packet, "prompt")


# =============================================================================
# DualPacket Tests
# =============================================================================

class TestDualPacket:
    """Tests for the DualPacket class combining Student and Evaluator packets."""

    def test_has_student_and_evaluator(self):
        """DualPacket must have both student and evaluator attributes."""
        from src.schema.extended_schema import DualPacket, StudentPacket, EvaluatorPacket

        student = StudentPacket(
            prompt="Test",
            evidence="Evidence",
            conditions=["C1"]
        )
        evaluator = EvaluatorPacket(
            expected_judgment="yes",
            domain="empirical",
            difficulty=2,
            evidence_citations=[],
            distractor_type=None,
            verification_chain={}
        )

        dual = DualPacket(student=student, evaluator=evaluator)

        assert hasattr(dual, "student")
        assert hasattr(dual, "evaluator")
        assert isinstance(dual.student, StudentPacket)
        assert isinstance(dual.evaluator, EvaluatorPacket)

    def test_student_packet_isolation(self):
        """Student packet must not expose evaluator data."""
        from src.schema.extended_schema import DualPacket, StudentPacket, EvaluatorPacket

        student = StudentPacket(
            prompt="What is 2+2?",
            evidence="Basic arithmetic",
            conditions=["Addition is valid"]
        )
        evaluator = EvaluatorPacket(
            expected_judgment="yes",
            domain="mathematical",
            difficulty=1,
            evidence_citations=[],
            distractor_type=None,
            verification_chain={}
        )

        dual = DualPacket(student=student, evaluator=evaluator)

        # Student packet must not contain expected_judgment
        student_str = str(dual.student)
        assert "expected_judgment" not in student_str.lower()
        assert "yes" not in student_str.lower() or "yes" in dual.student.evidence.lower()

    def test_evaluator_packet_has_ground_truth(self):
        """Evaluator packet must have ground truth for reward calculation."""
        from src.schema.extended_schema import DualPacket, StudentPacket, EvaluatorPacket

        student = StudentPacket(
            prompt="Test",
            evidence="Evidence",
            conditions=["C1"]
        )
        evaluator = EvaluatorPacket(
            expected_judgment="no",
            domain="pop_science",
            difficulty=3,
            evidence_citations=[],
            distractor_type=None,
            verification_chain={}
        )

        dual = DualPacket(student=student, evaluator=evaluator)

        # Evaluator must contain ground truth
        assert dual.evaluator.expected_judgment == "no"
        assert dual.evaluator.domain == "pop_science"


# =============================================================================
# VerificationChain Tests
# =============================================================================

class TestVerificationChain:
    """Tests for the DomainVerificationChain dataclass."""

    def test_has_base_fields(self):
        """VerificationChain must have domain and verification_type."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(
            domain="mathematical",
            verification_type="proof",
            required_elements=["axioms", "steps", "conclusion"]
        )

        assert hasattr(chain, "domain")
        assert hasattr(chain, "verification_type")
        assert hasattr(chain, "required_elements")

    # -------------------------------------------------------------------------
    # Mathematical Domain Tests
    # -------------------------------------------------------------------------

    def test_mathematical_chain_has_proof_steps(self, mathematical_verification_chain):
        """Mathematical verification chain must have proof_steps."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**mathematical_verification_chain)

        assert hasattr(chain, "proof_steps")
        assert chain.proof_steps is not None
        assert len(chain.proof_steps) > 0

    def test_mathematical_chain_has_axiom_refs(self, mathematical_verification_chain):
        """Mathematical verification chain must have axiom_refs."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**mathematical_verification_chain)

        assert hasattr(chain, "axiom_refs")
        assert chain.axiom_refs is not None

    def test_mathematical_chain_has_qed_marker(self, mathematical_verification_chain):
        """Mathematical verification chain must have qed_marker."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**mathematical_verification_chain)

        assert hasattr(chain, "qed_marker")
        assert chain.qed_marker is True

    def test_mathematical_chain_verification_type_is_proof(self, mathematical_verification_chain):
        """Mathematical chain verification_type must be 'proof'."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**mathematical_verification_chain)

        assert chain.verification_type == "proof"

    # -------------------------------------------------------------------------
    # Empirical Domain Tests
    # -------------------------------------------------------------------------

    def test_empirical_chain_has_data_points(self, empirical_verification_chain):
        """Empirical verification chain must have data_points."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**empirical_verification_chain)

        assert hasattr(chain, "data_points")
        assert chain.data_points is not None
        assert len(chain.data_points) > 0

    def test_empirical_chain_has_methodology(self, empirical_verification_chain):
        """Empirical verification chain must have methodology."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**empirical_verification_chain)

        assert hasattr(chain, "methodology")
        assert chain.methodology is not None

    def test_empirical_chain_has_replication(self, empirical_verification_chain):
        """Empirical verification chain must have replication status."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**empirical_verification_chain)

        assert hasattr(chain, "replication")
        assert chain.replication is not None

    def test_empirical_chain_verification_type_is_experimental(self, empirical_verification_chain):
        """Empirical chain verification_type must be 'experimental'."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**empirical_verification_chain)

        assert chain.verification_type == "experimental"

    # -------------------------------------------------------------------------
    # Philosophic Domain Tests
    # -------------------------------------------------------------------------

    def test_philosophic_chain_has_performative_check(self, philosophic_verification_chain):
        """Philosophic verification chain must have performative_check."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**philosophic_verification_chain)

        assert hasattr(chain, "performative_check")
        assert chain.performative_check is not None

    def test_philosophic_chain_has_self_consistency(self, philosophic_verification_chain):
        """Philosophic verification chain must have self_consistency."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**philosophic_verification_chain)

        assert hasattr(chain, "self_consistency")

    def test_philosophic_chain_has_counterposition_test(self, philosophic_verification_chain):
        """Philosophic verification chain must have counterposition_test."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**philosophic_verification_chain)

        assert hasattr(chain, "counterposition_test")
        assert chain.counterposition_test is not None

    def test_philosophic_chain_verification_type_is_performative(self, philosophic_verification_chain):
        """Philosophic chain verification_type must be 'performative_consistency'."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**philosophic_verification_chain)

        assert chain.verification_type == "performative_consistency"

    # -------------------------------------------------------------------------
    # Pop Science Domain Tests
    # -------------------------------------------------------------------------

    def test_pop_science_chain_has_unfulfilled_claims(self, pop_science_verification_chain):
        """Pop science verification chain must track unfulfilled_claims."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**pop_science_verification_chain)

        assert hasattr(chain, "unfulfilled_claims")
        assert chain.unfulfilled_claims is not None
        assert len(chain.unfulfilled_claims) > 0

    def test_pop_science_chain_verification_type_is_counterposition_detection(
        self, pop_science_verification_chain
    ):
        """Pop science chain verification_type must be 'counterposition_detection'."""
        from src.schema.extended_schema import DomainVerificationChain

        chain = DomainVerificationChain(**pop_science_verification_chain)

        assert chain.verification_type == "counterposition_detection"


# =============================================================================
# CoverageCell Tests
# =============================================================================

class TestCoverageCell:
    """Tests for the CoverageCell dataclass."""

    def test_has_required_fields(self):
        """CoverageCell must have domain, judgment_type, difficulty, has_distractor."""
        from src.schema.extended_schema import CoverageCell

        cell = CoverageCell(
            domain="mathematical",
            judgment_type="yes",
            difficulty=2,
            has_distractor=False
        )

        assert hasattr(cell, "domain")
        assert hasattr(cell, "judgment_type")
        assert hasattr(cell, "difficulty")
        assert hasattr(cell, "has_distractor")

    def test_cell_id_format_positive(self, coverage_cell_data):
        """CoverageCell must compute correct cell_id for positive examples."""
        from src.schema.extended_schema import CoverageCell

        # Test first example: mathematical_yes_1_positive
        data = coverage_cell_data[0]
        cell = CoverageCell(
            domain=data["domain"],
            judgment_type=data["judgment_type"],
            difficulty=data["difficulty"],
            has_distractor=data["has_distractor"]
        )

        assert cell.cell_id == data["expected_cell_id"]

    def test_cell_id_format_distractor(self, coverage_cell_data):
        """CoverageCell must compute correct cell_id for distractor examples."""
        from src.schema.extended_schema import CoverageCell

        # Test second example: empirical_no_3_distractor
        data = coverage_cell_data[1]
        cell = CoverageCell(
            domain=data["domain"],
            judgment_type=data["judgment_type"],
            difficulty=data["difficulty"],
            has_distractor=data["has_distractor"]
        )

        assert cell.cell_id == data["expected_cell_id"]

    def test_cell_id_deterministic(self):
        """Same inputs must produce same cell_id."""
        from src.schema.extended_schema import CoverageCell

        cell1 = CoverageCell(
            domain="empirical",
            judgment_type="no",
            difficulty=3,
            has_distractor=True
        )
        cell2 = CoverageCell(
            domain="empirical",
            judgment_type="no",
            difficulty=3,
            has_distractor=True
        )

        assert cell1.cell_id == cell2.cell_id

    def test_cell_id_unique_for_different_inputs(self):
        """Different inputs must produce different cell_ids."""
        from src.schema.extended_schema import CoverageCell

        cell1 = CoverageCell(
            domain="mathematical",
            judgment_type="yes",
            difficulty=1,
            has_distractor=False
        )
        cell2 = CoverageCell(
            domain="mathematical",
            judgment_type="no",
            difficulty=1,
            has_distractor=False
        )
        cell3 = CoverageCell(
            domain="mathematical",
            judgment_type="yes",
            difficulty=2,
            has_distractor=False
        )

        assert cell1.cell_id != cell2.cell_id
        assert cell1.cell_id != cell3.cell_id
        assert cell2.cell_id != cell3.cell_id

    def test_cell_id_property_is_string(self):
        """cell_id must be a string."""
        from src.schema.extended_schema import CoverageCell

        cell = CoverageCell(
            domain="mathematical",
            judgment_type="yes",
            difficulty=1,
            has_distractor=False
        )

        assert isinstance(cell.cell_id, str)

    def test_all_coverage_cells(self, coverage_cell_data):
        """Test all coverage cell examples."""
        from src.schema.extended_schema import CoverageCell

        for data in coverage_cell_data:
            cell = CoverageCell(
                domain=data["domain"],
                judgment_type=data["judgment_type"],
                difficulty=data["difficulty"],
                has_distractor=data["has_distractor"]
            )
            assert cell.cell_id == data["expected_cell_id"], \
                f"Expected {data['expected_cell_id']}, got {cell.cell_id}"


# =============================================================================
# Integration Tests
# =============================================================================

class TestSchemaIntegration:
    """Integration tests ensuring schema components work together."""

    def test_enhanced_sample_to_dual_packet(self, empirical_sample_data):
        """EnhancedJudgmentSample should convert to DualPacket."""
        from src.schema.extended_schema import (
            EnhancedJudgmentSample,
            DualPacket,
            StudentPacket,
            EvaluatorPacket
        )

        sample = EnhancedJudgmentSample(**empirical_sample_data)

        # Should have a method to convert to dual packet
        if hasattr(sample, "to_dual_packet"):
            dual = sample.to_dual_packet()
            assert isinstance(dual, DualPacket)
            assert isinstance(dual.student, StudentPacket)
            assert isinstance(dual.evaluator, EvaluatorPacket)

    def test_verification_chain_matches_domain(self, mathematical_sample_data):
        """VerificationChain domain must match sample domain."""
        from src.schema.extended_schema import (
            EnhancedJudgmentSample,
            DomainVerificationChain
        )

        sample = EnhancedJudgmentSample(**mathematical_sample_data)

        if sample.verification_chain:
            chain = DomainVerificationChain(**sample.verification_chain)
            assert chain.domain == sample.domain

    def test_evidence_citations_in_evidence_text(self, empirical_sample_data):
        """Evidence citations should exist within evidence_text."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample = EnhancedJudgmentSample(**empirical_sample_data)

        for citation_data in sample.evidence_citations:
            # The citation text should be findable in the evidence
            assert citation_data["text"] in sample.evidence_text or \
                   sample.evidence_text[
                       citation_data["start_idx"]:citation_data["end_idx"]
                   ] == citation_data["text"]

    def test_coverage_cell_from_sample(self, empirical_sample_data):
        """Should create CoverageCell from EnhancedJudgmentSample."""
        from src.schema.extended_schema import EnhancedJudgmentSample, CoverageCell

        sample = EnhancedJudgmentSample(**empirical_sample_data)

        cell = CoverageCell(
            domain=sample.domain,
            judgment_type=sample.expected_judgment,
            difficulty=sample.difficulty,
            has_distractor=(sample.distractor_type is not None)
        )

        # Cell should match sample attributes
        assert cell.domain == "empirical"
        assert cell.judgment_type == "yes"
        assert cell.difficulty == 3
        assert cell.has_distractor is False


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSchemaSerialization:
    """Tests for schema serialization to/from JSON."""

    def test_enhanced_sample_to_dict(self, mathematical_sample_data):
        """EnhancedJudgmentSample should serialize to dict."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample = EnhancedJudgmentSample(**mathematical_sample_data)

        # Dataclass should have asdict capability
        if hasattr(sample, "to_dict"):
            d = sample.to_dict()
        else:
            from dataclasses import asdict
            d = asdict(sample)

        assert isinstance(d, dict)
        assert "proposition" in d
        assert "domain" in d

    def test_enhanced_sample_to_json(self, mathematical_sample_data):
        """EnhancedJudgmentSample should serialize to JSON."""
        from src.schema.extended_schema import EnhancedJudgmentSample
        import json

        sample = EnhancedJudgmentSample(**mathematical_sample_data)

        # Should be JSON serializable
        if hasattr(sample, "to_json"):
            json_str = sample.to_json()
        else:
            from dataclasses import asdict
            json_str = json.dumps(asdict(sample))

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["domain"] == "mathematical"

    def test_enhanced_sample_from_dict(self, mathematical_sample_data):
        """EnhancedJudgmentSample should deserialize from dict."""
        from src.schema.extended_schema import EnhancedJudgmentSample

        sample = EnhancedJudgmentSample(**mathematical_sample_data)

        # Should support reconstruction from dict
        if hasattr(sample, "to_dict"):
            d = sample.to_dict()
        else:
            from dataclasses import asdict
            d = asdict(sample)

        reconstructed = EnhancedJudgmentSample(**d)
        assert reconstructed.proposition == sample.proposition
        assert reconstructed.domain == sample.domain

    def test_dual_packet_serializes_separately(self, empirical_sample_data):
        """DualPacket student and evaluator should serialize separately."""
        from src.schema.extended_schema import (
            EnhancedJudgmentSample,
            StudentPacket,
            EvaluatorPacket,
            DualPacket
        )
        import json
        from dataclasses import asdict

        sample = EnhancedJudgmentSample(**empirical_sample_data)

        student = StudentPacket(
            prompt=sample.proposition,
            evidence=sample.evidence_text,
            conditions=sample.conditions
        )
        evaluator = EvaluatorPacket(
            expected_judgment=sample.expected_judgment,
            domain=sample.domain,
            difficulty=sample.difficulty,
            evidence_citations=sample.evidence_citations,
            distractor_type=sample.distractor_type,
            verification_chain=sample.verification_chain
        )

        dual = DualPacket(student=student, evaluator=evaluator)

        # Serialize student packet
        student_json = json.dumps(asdict(dual.student))
        evaluator_json = json.dumps(asdict(dual.evaluator))

        # Student should not contain expected_judgment
        assert "expected_judgment" not in student_json

        # Evaluator should contain expected_judgment
        assert "expected_judgment" in evaluator_json


# =============================================================================
