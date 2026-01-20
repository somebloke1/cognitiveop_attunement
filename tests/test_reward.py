#!/usr/bin/env python3
"""
TDD Tests for Enhanced Reward Function

Tests the reward function architecture for the Cognitive Organism project.
These tests are written TDD-style and should FAIL initially until the
enhanced reward module is implemented.

Test Categories:
1. Correctness Component (PRIMARY)
2. Evidence Grounding Component (Insight 3)
3. Domain Verification Component (Insight 7)
4. Structure Component (existing)
5. Composite Reward (Multiplicative Gating)
6. Batch Reward Function (TRL-compatible interface)

Key Principle: Form without correct judgment is empty form.
Correctness gates structural rewards via multiplicative scoring.
"""

import pytest
import json
from typing import Optional


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_mathematical_completion_correct():
    """A correct mathematical judgment with proper structure."""
    return {
        "completion": """
        Given axioms A1 (transitivity) and A2 (symmetry), we can derive:

        Step 1: By A1, if a < b and b < c, then a < c
        Step 2: Applying to the given values x=1, y=2, z=3
        Step 3: Since 1 < 2 and 2 < 3, by transitivity 1 < 3

        The conditions are fulfilled:
        - Condition 1: Axiom A1 applies - FULFILLED
        - Condition 2: Values satisfy ordering - FULFILLED

        Therefore, the proposition holds.
        QED

        Judgment: Yes
        """,
        "expected_judgment": "Yes",
        "domain": "mathematical",
        "input_text": "Proposition: If x < y and y < z, then x < z. Given: x=1, y=2, z=3",
        "citations": [],
        "verification_chain": {
            "verification_type": "proof",
            "required_elements": ["axioms_referenced", "logical_steps", "conclusion_marker"],
            "proof_steps": ["Apply transitivity axiom", "Substitute values", "Derive conclusion"]
        }
    }


@pytest.fixture
def sample_mathematical_completion_wrong():
    """A wrong mathematical judgment despite proper structure."""
    return {
        "completion": """
        Given axioms A1 (transitivity) and A2 (symmetry), we can derive:

        Step 1: By A1, if a < b and b < c, then a < c
        Step 2: However, the axioms don't apply here

        The conditions are not fulfilled.
        QED

        Judgment: No
        """,
        "expected_judgment": "Yes",  # This is WRONG
        "domain": "mathematical",
        "input_text": "Proposition: If x < y and y < z, then x < z. Given: x=1, y=2, z=3",
        "citations": [],
        "verification_chain": {}
    }


@pytest.fixture
def sample_empirical_completion_with_citations():
    """An empirical judgment with valid evidence citations."""
    return {
        "completion": """
        Examining the evidence provided:

        The data states "temperature increased by 2.5 degrees" which directly
        supports the hypothesis of warming.

        Additionally, "measurements taken over 30 days" indicates sufficient
        sampling period.

        Conditions checked:
        - Condition 1: Temperature change documented - FULFILLED (see: "temperature increased by 2.5 degrees")
        - Condition 2: Adequate measurement period - FULFILLED (see: "measurements taken over 30 days")

        Based on the verified data, the proposition is supported.

        Judgment: Yes
        """,
        "expected_judgment": "Yes",
        "domain": "empirical",
        "input_text": "The study found that temperature increased by 2.5 degrees. Measurements taken over 30 days showed consistent warming.",
        "citations": [
            {"text": "temperature increased by 2.5 degrees", "supports_condition": "temperature_change"},
            {"text": "measurements taken over 30 days", "supports_condition": "measurement_period"}
        ],
        "verification_chain": {
            "verification_type": "experimental",
            "required_elements": ["data_points_cited", "methodology_stated"],
            "data_points": ["2.5 degree increase", "30 day period"]
        }
    }


@pytest.fixture
def sample_empirical_completion_hallucinated():
    """An empirical judgment with hallucinated citations (not in input)."""
    return {
        "completion": """
        The evidence clearly shows "a 5 degree temperature drop" which contradicts
        the warming hypothesis.

        The "50 day measurement period" provides robust data.

        Conditions:
        - Temperature data: "5 degree drop" indicates cooling
        - Measurement period: Sufficient

        Judgment: No
        """,
        "expected_judgment": "Yes",  # Wrong because citations are hallucinated
        "domain": "empirical",
        "input_text": "The study found that temperature increased by 2.5 degrees. Measurements taken over 30 days showed consistent warming.",
        "citations": [],
        "verification_chain": {}
    }


@pytest.fixture
def sample_philosophic_completion():
    """A philosophic judgment with performative consistency check."""
    return {
        "completion": """
        Position examined: "All knowledge is merely subjective opinion"

        Performative consistency check:
        - To affirm this position requires: claiming to KNOW that knowledge is merely opinion
        - But if all knowledge is merely opinion, this claim is also merely opinion
        - The position presupposes what it denies (objective knowledge)

        This is a self-defeating counterposition. The act of affirming it
        reasonably presupposes the basic positions it claims to reject.

        The position cannot be held coherently by one who grasps it
        intelligently and affirms it reasonably.

        Judgment: No
        """,
        "expected_judgment": "No",
        "domain": "philosophic",
        "input_text": "Position: All knowledge is merely subjective opinion. Assess whether this can be coherently affirmed.",
        "citations": [],
        "verification_chain": {
            "verification_type": "performative_consistency",
            "required_elements": ["position_stated", "conditions_of_affirmation", "self_consistency_check"],
            "performative_check": "Affirming requires objective claim, contradicting position"
        }
    }


@pytest.fixture
def sample_common_sense_completion():
    """A common sense practical judgment."""
    return {
        "completion": """
        Examining the practical situation:

        The shortcut through the park works when it's dry, but the evidence
        shows "heavy rain overnight" which means the path will be muddy.

        In this context, the practical outcome would be:
        - Longer travel time due to mud
        - Possibly getting stuck or dirty

        Based on experience with similar situations, the shortcut is not
        advisable today.

        Judgment: No
        """,
        "expected_judgment": "No",
        "domain": "common_sense",
        "input_text": "Should I take the shortcut through the park? Evidence: heavy rain overnight.",
        "citations": [{"text": "heavy rain overnight", "supports_condition": "weather"}],
        "verification_chain": {
            "verification_type": "pragmatic",
            "required_elements": ["situation_described", "practical_outcome"],
            "practical_outcomes": ["muddy path", "longer travel time"]
        }
    }


@pytest.fixture
def sample_pop_science_completion():
    """A pop science detection judgment."""
    return {
        "completion": """
        Claim examined: "Coffee causes cancer based on new study"

        Actual evidence analysis:
        - The study found correlation between coffee consumption and one type of cancer
        - Sample size: 200 participants
        - No control for confounding variables mentioned

        Conditions for causal claim to be true:
        1. Controlled study design - NOT FULFILLED (observational only)
        2. Mechanism identified - NOT FULFILLED
        3. Replication - NOT FULFILLED

        This is a case of correlation claimed as causation.
        The conditions for the causal claim are unfulfilled.

        Judgment: No
        """,
        "expected_judgment": "No",
        "domain": "pop_science",
        "input_text": "A study of 200 people found coffee drinkers had higher rates of one cancer type.",
        "citations": [],
        "verification_chain": {
            "verification_type": "counterposition_detection",
            "required_elements": ["claimed_proof_identified", "conditions_unfulfilled_listed"],
            "unfulfilled_claims": ["causal mechanism", "controlled design", "replication"]
        }
    }


@pytest.fixture
def sample_no_judgment_completion():
    """A completion that lacks any judgment marker."""
    return {
        "completion": """
        This is an interesting proposition to consider. There are many
        factors that might influence the outcome. On one hand, we could
        argue that the conditions are met. On the other hand, there are
        some uncertainties.

        Further investigation might be needed to determine the answer.
        """,
        "expected_judgment": "Yes",
        "domain": "empirical",
        "input_text": "Some evidence text.",
        "citations": [],
        "verification_chain": {}
    }


@pytest.fixture
def sample_insufficient_judgment():
    """A legitimate 'Insufficient' judgment."""
    return {
        "completion": """
        Examining the evidence provided:

        The proposition asks about temperature trends, but the evidence
        only mentions "some measurements were taken" without specifying
        actual values or timeframes.

        Conditions for judgment:
        - Condition 1: Actual temperature data - NOT PROVIDED
        - Condition 2: Measurement methodology - NOT SPECIFIED

        The evidence is insufficient to determine whether the conditions
        for the proposition are fulfilled or not.

        Judgment: Insufficient
        """,
        "expected_judgment": "Insufficient",
        "domain": "empirical",
        "input_text": "Some measurements were taken.",
        "citations": [],
        "verification_chain": {}
    }


@pytest.fixture
def sample_minimal_correct():
    """Minimal completion with just correct judgment."""
    return {
        "completion": "Judgment: Yes",
        "expected_judgment": "Yes",
        "domain": "empirical",
        "input_text": "Evidence text.",
        "citations": [],
        "verification_chain": {}
    }


@pytest.fixture
def sample_verification_chains():
    """Sample verification chains for each domain."""
    return {
        "mathematical": {
            "verification_type": "proof",
            "required_elements": ["axioms_referenced", "logical_steps", "conclusion_marker"],
            "proof_steps": ["State axiom", "Apply rule", "Derive conclusion"],
            "output_markers": ["QED", "therefore", "thus", "it follows that"]
        },
        "empirical": {
            "verification_type": "experimental",
            "required_elements": ["data_points_cited", "methodology_stated", "replication_status"],
            "data_points": ["measurement value", "sample size"],
            "output_markers": ["data shows", "measured", "observed", "replicated"]
        },
        "common_sense": {
            "verification_type": "pragmatic",
            "required_elements": ["situation_described", "practical_outcome"],
            "practical_outcomes": ["works/doesn't work", "experience-based"],
            "output_markers": ["works", "practical", "experience shows"]
        },
        "pop_science": {
            "verification_type": "counterposition_detection",
            "required_elements": ["claimed_proof_identified", "conditions_unfulfilled_listed"],
            "unfulfilled_claims": ["correlation not causation", "authority appeal"],
            "output_markers": ["claim unfounded", "correlation not causation", "conditions unfulfilled"]
        },
        "philosophic": {
            "verification_type": "performative_consistency",
            "required_elements": ["position_stated", "self_consistency_check"],
            "performative_check": "Can this be coherently affirmed?",
            "output_markers": ["self-defeating", "performative contradiction", "presupposes"]
        }
    }


# =============================================================================
# 1. CORRECTNESS COMPONENT TESTS (PRIMARY)
# =============================================================================

class TestCorrectnessComponent:
    """Tests for score_correctness(completion, expected_judgment)"""

    def test_correct_yes_judgment(self, sample_mathematical_completion_correct):
        """Correct 'Yes' judgment returns +0.6"""
        from src.training.enhanced_reward import score_correctness

        score = score_correctness(
            completion=sample_mathematical_completion_correct["completion"],
            expected_judgment="Yes"
        )
        assert score == pytest.approx(0.6)

    def test_correct_no_judgment(self, sample_philosophic_completion):
        """Correct 'No' judgment returns +0.6"""
        from src.training.enhanced_reward import score_correctness

        score = score_correctness(
            completion=sample_philosophic_completion["completion"],
            expected_judgment="No"
        )
        assert score == pytest.approx(0.6)

    def test_correct_insufficient_judgment(self, sample_insufficient_judgment):
        """Correct 'Insufficient' judgment returns +0.6"""
        from src.training.enhanced_reward import score_correctness

        score = score_correctness(
            completion=sample_insufficient_judgment["completion"],
            expected_judgment="Insufficient"
        )
        assert score == pytest.approx(0.6)

    def test_wrong_judgment(self, sample_mathematical_completion_wrong):
        """Wrong judgment returns -0.5"""
        from src.training.enhanced_reward import score_correctness

        score = score_correctness(
            completion=sample_mathematical_completion_wrong["completion"],
            expected_judgment="Yes"  # Completion says No, expected is Yes
        )
        assert score == pytest.approx(-0.5)

    def test_no_judgment_detected(self, sample_no_judgment_completion):
        """No judgment marker detected returns 0.0"""
        from src.training.enhanced_reward import score_correctness

        score = score_correctness(
            completion=sample_no_judgment_completion["completion"],
            expected_judgment="Yes"
        )
        assert score == pytest.approx(0.0)

    def test_case_insensitive_matching(self):
        """Judgment extraction is case insensitive"""
        from src.training.enhanced_reward import score_correctness

        # Various case combinations
        completions = [
            "JUDGMENT: YES",
            "judgment: yes",
            "Judgment: YES",
            "JUDGMENT: yes",
        ]

        for completion in completions:
            score = score_correctness(completion, "Yes")
            assert score == pytest.approx(0.6), f"Failed for: {completion}"

    def test_judgment_extraction_various_formats(self):
        """Judgment extraction handles various formats"""
        from src.training.enhanced_reward import score_correctness

        formats = [
            "Judgment: Yes",
            "Judgment - Yes",
            "Judgment:\nYes",
            "judgment:yes",  # No space
            "The judgment is: Yes",  # Preceding text
            "Final judgment: Yes.",  # Trailing punctuation
        ]

        for completion in formats:
            score = score_correctness(completion, "Yes")
            assert score == pytest.approx(0.6), f"Failed for format: {completion}"

    def test_insufficient_not_rewarded_incorrectly(self):
        """'Insufficient' when 'Yes' is expected returns -0.5"""
        from src.training.enhanced_reward import score_correctness

        completion = "The evidence is lacking. Judgment: Insufficient"
        score = score_correctness(completion, "Yes")
        assert score == pytest.approx(-0.5)


# =============================================================================
# 2. EVIDENCE GROUNDING COMPONENT TESTS (Insight 3)
# =============================================================================

class TestEvidenceGroundingComponent:
    """Tests for score_evidence_grounding(completion, input_text, citations)"""

    def test_citation_present_bonus(self, sample_empirical_completion_with_citations):
        """Having a citation present adds +0.05"""
        from src.training.enhanced_reward import score_evidence_grounding

        score = score_evidence_grounding(
            completion=sample_empirical_completion_with_citations["completion"],
            input_text=sample_empirical_completion_with_citations["input_text"],
            citations=sample_empirical_completion_with_citations["citations"]
        )
        assert score >= 0.05  # At least citation present bonus

    def test_citation_matches_input_bonus(self, sample_empirical_completion_with_citations):
        """Citation that matches input text adds +0.05"""
        from src.training.enhanced_reward import score_evidence_grounding

        score = score_evidence_grounding(
            completion=sample_empirical_completion_with_citations["completion"],
            input_text=sample_empirical_completion_with_citations["input_text"],
            citations=sample_empirical_completion_with_citations["citations"]
        )
        assert score >= 0.10  # Citation present + matches input

    def test_citation_relevant_to_conditions_bonus(self, sample_empirical_completion_with_citations):
        """Citation relevant to conditions adds +0.05"""
        from src.training.enhanced_reward import score_evidence_grounding

        score = score_evidence_grounding(
            completion=sample_empirical_completion_with_citations["completion"],
            input_text=sample_empirical_completion_with_citations["input_text"],
            citations=sample_empirical_completion_with_citations["citations"]
        )
        assert score == pytest.approx(0.15)  # Full bonus

    def test_hallucinated_citation_no_bonus(self, sample_empirical_completion_hallucinated):
        """Hallucinated citations (not in input) get minimal/no bonus"""
        from src.training.enhanced_reward import score_evidence_grounding

        score = score_evidence_grounding(
            completion=sample_empirical_completion_hallucinated["completion"],
            input_text=sample_empirical_completion_hallucinated["input_text"],
            citations=[]
        )
        assert score < 0.05  # Citation present but doesn't match input

    def test_no_citations_zero_score(self, sample_no_judgment_completion):
        """No citations in completion returns 0.0"""
        from src.training.enhanced_reward import score_evidence_grounding

        score = score_evidence_grounding(
            completion=sample_no_judgment_completion["completion"],
            input_text=sample_no_judgment_completion["input_text"],
            citations=[]
        )
        assert score == pytest.approx(0.0)

    def test_partial_match_partial_score(self):
        """Partially matching citations get partial score"""
        from src.training.enhanced_reward import score_evidence_grounding

        completion = 'The evidence shows "temperature increased" which supports the claim.'
        input_text = "Temperature increased by 2.5 degrees over the study period."

        score = score_evidence_grounding(
            completion=completion,
            input_text=input_text,
            citations=[]
        )
        # "temperature increased" is in input, so should get some credit
        assert 0.0 < score < 0.15

    def test_quote_extraction(self):
        """Can extract quoted text from completion"""
        from src.training.enhanced_reward import extract_quotes

        completion = '''The evidence states "first quote" and also "second quote" were found.'''
        quotes = extract_quotes(completion)

        assert "first quote" in quotes
        assert "second quote" in quotes
        assert len(quotes) == 2


# =============================================================================
# 3. DOMAIN VERIFICATION COMPONENT TESTS (Insight 7)
# =============================================================================

class TestDomainVerificationComponent:
    """Tests for score_domain_verification(completion, domain, verification_chain)"""

    def test_mathematical_proof_structure(self, sample_mathematical_completion_correct, sample_verification_chains):
        """Mathematical domain checks for proof structure"""
        from src.training.enhanced_reward import score_domain_verification

        score = score_domain_verification(
            completion=sample_mathematical_completion_correct["completion"],
            domain="mathematical",
            verification_chain=sample_verification_chains["mathematical"]
        )
        # Has QED, therefore, axiom references
        assert score > 0.05

    def test_mathematical_markers_detected(self, sample_verification_chains):
        """Mathematical verification detects QED, therefore, etc."""
        from src.training.enhanced_reward import score_domain_verification

        completion = "Given axioms... Step 1... Therefore... QED"
        score = score_domain_verification(
            completion=completion,
            domain="mathematical",
            verification_chain=sample_verification_chains["mathematical"]
        )
        assert score > 0.0

    def test_empirical_data_reference(self, sample_empirical_completion_with_citations, sample_verification_chains):
        """Empirical domain checks for data references"""
        from src.training.enhanced_reward import score_domain_verification

        score = score_domain_verification(
            completion=sample_empirical_completion_with_citations["completion"],
            domain="empirical",
            verification_chain=sample_verification_chains["empirical"]
        )
        # Has data, evidence, measurement language
        assert score > 0.05

    def test_empirical_markers_detected(self, sample_verification_chains):
        """Empirical verification detects data, measured, observed, etc."""
        from src.training.enhanced_reward import score_domain_verification

        completion = "The data shows a clear pattern. We observed that... measured values indicate..."
        score = score_domain_verification(
            completion=completion,
            domain="empirical",
            verification_chain=sample_verification_chains["empirical"]
        )
        assert score > 0.0

    def test_philosophic_performative_check(self, sample_philosophic_completion, sample_verification_chains):
        """Philosophic domain checks for performative assessment"""
        from src.training.enhanced_reward import score_domain_verification

        score = score_domain_verification(
            completion=sample_philosophic_completion["completion"],
            domain="philosophic",
            verification_chain=sample_verification_chains["philosophic"]
        )
        # Has self-defeating, presupposes, coherent language
        assert score > 0.05

    def test_philosophic_markers_detected(self, sample_verification_chains):
        """Philosophic verification detects self-defeating, presupposes, etc."""
        from src.training.enhanced_reward import score_domain_verification

        completion = "This position is self-defeating because it presupposes what it denies."
        score = score_domain_verification(
            completion=completion,
            domain="philosophic",
            verification_chain=sample_verification_chains["philosophic"]
        )
        assert score > 0.0

    def test_common_sense_practical_structure(self, sample_common_sense_completion, sample_verification_chains):
        """Common sense domain checks for practical structure"""
        from src.training.enhanced_reward import score_domain_verification

        score = score_domain_verification(
            completion=sample_common_sense_completion["completion"],
            domain="common_sense",
            verification_chain=sample_verification_chains["common_sense"]
        )
        # Has practical, experience, works language
        assert score > 0.0

    def test_pop_science_detection_structure(self, sample_pop_science_completion, sample_verification_chains):
        """Pop science domain checks for counterposition detection"""
        from src.training.enhanced_reward import score_domain_verification

        score = score_domain_verification(
            completion=sample_pop_science_completion["completion"],
            domain="pop_science",
            verification_chain=sample_verification_chains["pop_science"]
        )
        # Has unfulfilled, correlation, claim language
        assert score > 0.0

    def test_wrong_domain_structure_low_score(self, sample_verification_chains):
        """Using wrong domain structure gets low score"""
        from src.training.enhanced_reward import score_domain_verification

        # Mathematical proof language for empirical domain
        completion = "Given axioms A1, A2... Step 1... Therefore... QED"
        score = score_domain_verification(
            completion=completion,
            domain="empirical",  # Mismatch!
            verification_chain=sample_verification_chains["empirical"]
        )
        # Should get lower score than if domain matched
        assert score < 0.08

    def test_unknown_domain_returns_zero(self):
        """Unknown domain returns 0.0"""
        from src.training.enhanced_reward import score_domain_verification

        score = score_domain_verification(
            completion="Some text",
            domain="unknown_domain",
            verification_chain={}
        )
        assert score == pytest.approx(0.0)


# =============================================================================
# 4. STRUCTURE COMPONENT TESTS (existing)
# =============================================================================

class TestStructureComponent:
    """Tests for score_structure(completion)"""

    def test_explicit_judgment_marker_bonus(self):
        """Having explicit 'Judgment: X' marker adds bonus"""
        from src.training.enhanced_reward import score_structure

        completion_with = "Analysis complete. Judgment: Yes"
        completion_without = "I think the answer is yes."

        score_with = score_structure(completion_with)
        score_without = score_structure(completion_without)

        assert score_with > score_without
        assert score_with > 0

    def test_condition_identification_bonus(self):
        """Identifying conditions adds bonus"""
        from src.training.enhanced_reward import score_structure

        completion_with_conditions = """
        Conditions checked:
        - Condition 1: Temperature data - FULFILLED
        - Condition 2: Sample size - FULFILLED

        Judgment: Yes
        """

        completion_without = "Judgment: Yes"

        score_with = score_structure(completion_with_conditions)
        score_without = score_structure(completion_without)

        assert score_with > score_without

    def test_fulfillment_reasoning_bonus(self):
        """Fulfillment reasoning language adds bonus"""
        from src.training.enhanced_reward import score_structure

        completion_with_fulfillment = """
        The conditions are fulfilled because the evidence shows X.
        Therefore, the link between conditions and proposition is established.
        Judgment: Yes
        """

        completion_minimal = "Judgment: Yes"

        score_with = score_structure(completion_with_fulfillment)
        score_minimal = score_structure(completion_minimal)

        assert score_with > score_minimal

    def test_structure_score_range(self):
        """Structure score is in range 0.0 to 0.15"""
        from src.training.enhanced_reward import score_structure

        test_cases = [
            "Judgment: Yes",
            "Some random text without judgment",
            "Conditions: 1, 2, 3. Fulfilled. Therefore. Judgment: Yes",
            "",
        ]

        for completion in test_cases:
            score = score_structure(completion)
            assert 0.0 <= score <= 0.15, f"Score {score} out of range for: {completion}"


# =============================================================================
# 5. COMPOSITE REWARD TESTS (Multiplicative Gating)
# =============================================================================

class TestCompositeReward:
    """Tests for compute_reward() with multiplicative gating"""

    def test_correct_with_all_bonuses_near_max(self, sample_mathematical_completion_correct, sample_verification_chains):
        """Correct judgment with all bonuses approaches max (~1.0)"""
        from src.training.enhanced_reward import compute_reward

        reward = compute_reward(
            completion=sample_mathematical_completion_correct["completion"],
            expected_judgment="Yes",
            domain="mathematical",
            input_text=sample_mathematical_completion_correct["input_text"],
            citations=[],
            verification_chain=sample_verification_chains["mathematical"]
        )

        # Should be close to max (correctness + bonuses)
        assert reward >= 0.7  # At least 0.6 correctness + some bonuses
        assert reward <= 1.0

    def test_correct_without_bonuses_around_base(self, sample_minimal_correct):
        """Correct judgment without bonuses is around +0.6"""
        from src.training.enhanced_reward import compute_reward

        reward = compute_reward(
            completion=sample_minimal_correct["completion"],
            expected_judgment="Yes",
            domain="empirical",
            input_text="",
            citations=[],
            verification_chain={}
        )

        # Just correctness, minimal structure
        assert 0.5 <= reward <= 0.8

    def test_incorrect_judgment_negative(self, sample_mathematical_completion_wrong):
        """Incorrect judgment yields negative reward"""
        from src.training.enhanced_reward import compute_reward

        reward = compute_reward(
            completion=sample_mathematical_completion_wrong["completion"],
            expected_judgment="Yes",
            domain="mathematical",
            input_text="",
            citations=[],
            verification_chain={}
        )

        assert reward < 0
        assert reward == pytest.approx(-0.5)  # No bonuses when wrong

    def test_bonuses_gated_by_correctness(self, sample_mathematical_completion_wrong, sample_verification_chains):
        """Structural bonuses don't apply when judgment is wrong"""
        from src.training.enhanced_reward import compute_reward

        # This completion has good structure (QED, axioms, steps) but wrong judgment
        reward = compute_reward(
            completion=sample_mathematical_completion_wrong["completion"],
            expected_judgment="Yes",  # Completion says No
            domain="mathematical",
            input_text="",
            citations=[],
            verification_chain=sample_verification_chains["mathematical"]
        )

        # Despite good structure, wrong judgment = -0.5, no bonuses
        assert reward == pytest.approx(-0.5)

    def test_correct_with_partial_bonuses_intermediate(self, sample_empirical_completion_with_citations, sample_verification_chains):
        """Correct judgment with some bonuses is intermediate"""
        from src.training.enhanced_reward import compute_reward

        reward = compute_reward(
            completion=sample_empirical_completion_with_citations["completion"],
            expected_judgment="Yes",
            domain="empirical",
            input_text=sample_empirical_completion_with_citations["input_text"],
            citations=sample_empirical_completion_with_citations["citations"],
            verification_chain=sample_verification_chains["empirical"]
        )

        # Correctness + evidence grounding + domain verification + structure
        assert 0.6 < reward <= 1.0

    def test_reward_range(self):
        """Total reward range is -0.5 to +1.0"""
        from src.training.enhanced_reward import compute_reward

        # Worst case: wrong judgment, no structure
        worst = compute_reward(
            completion="Wrong answer. Judgment: Yes",
            expected_judgment="No",
            domain="empirical",
            input_text="",
            citations=[],
            verification_chain={}
        )
        assert worst >= -0.5

        # Best case: correct with all bonuses
        best = compute_reward(
            completion='''
            The evidence "data point" supports condition 1. FULFILLED.
            Data shows clear pattern. Observed and measured values confirm.
            Judgment: Yes
            ''',
            expected_judgment="Yes",
            domain="empirical",
            input_text="The data point shows clear evidence.",
            citations=[{"text": "data point", "supports_condition": "evidence"}],
            verification_chain={"verification_type": "experimental", "output_markers": ["data shows", "measured"]}
        )
        assert best <= 1.0

    def test_no_judgment_gated(self, sample_no_judgment_completion):
        """No judgment marker means zero correctness, gated bonuses"""
        from src.training.enhanced_reward import compute_reward

        reward = compute_reward(
            completion=sample_no_judgment_completion["completion"],
            expected_judgment="Yes",
            domain="empirical",
            input_text="",
            citations=[],
            verification_chain={}
        )

        # No judgment detected = 0.0 correctness score
        # Could have some negative length penalty
        assert reward <= 0.1


# =============================================================================
# 6. BATCH REWARD FUNCTION TESTS (TRL-compatible interface)
# =============================================================================

class TestBatchRewardFunction:
    """Tests for reward_fn() TRL-compatible interface"""

    def test_batch_processing(self, sample_mathematical_completion_correct, sample_philosophic_completion):
        """Handles batch of completions"""
        from src.training.enhanced_reward import reward_fn

        rewards = reward_fn(
            completions=[
                sample_mathematical_completion_correct["completion"],
                sample_philosophic_completion["completion"],
            ],
            prompts=["prompt1", "prompt2"],  # Required by TRL
            expected_judgment=["Yes", "No"],
            domain=["mathematical", "philosophic"],
            input_text=["input1", "input2"],
            citations=["[]", "[]"],
            verification_chain=["{}"] * 2,
        )

        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)

    def test_batch_returns_list_of_floats(self):
        """Returns list of float rewards"""
        from src.training.enhanced_reward import reward_fn

        rewards = reward_fn(
            completions=["Judgment: Yes", "Judgment: No", "Judgment: Insufficient"],
            prompts=["p1", "p2", "p3"],
            expected_judgment=["Yes", "No", "Insufficient"],
            domain=["empirical", "empirical", "empirical"],
        )

        assert isinstance(rewards, list)
        assert len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)

    def test_batch_handles_missing_optional_kwargs(self):
        """Handles missing optional keyword arguments gracefully"""
        from src.training.enhanced_reward import reward_fn

        # Minimal required arguments
        rewards = reward_fn(
            completions=["Judgment: Yes"],
            prompts=["prompt"],
            expected_judgment=["Yes"],
        )

        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_batch_handles_json_encoded_kwargs(self):
        """Handles JSON-encoded citation and verification chain kwargs"""
        from src.training.enhanced_reward import reward_fn

        citations_json = json.dumps([{"text": "evidence", "supports_condition": "cond1"}])
        chain_json = json.dumps({"verification_type": "experimental", "output_markers": ["data shows"]})

        rewards = reward_fn(
            completions=["The evidence shows... Judgment: Yes"],
            prompts=["prompt"],
            expected_judgment=["Yes"],
            domain=["empirical"],
            input_text=["The evidence is clear."],
            citations=[citations_json],
            verification_chain=[chain_json],
        )

        assert len(rewards) == 1

    def test_batch_mixed_correctness(self):
        """Batch with mixed correct/incorrect judgments"""
        from src.training.enhanced_reward import reward_fn

        rewards = reward_fn(
            completions=[
                "Judgment: Yes",  # Correct
                "Judgment: No",   # Wrong (expected Yes)
                "Judgment: Yes",  # Wrong (expected No)
            ],
            prompts=["p1", "p2", "p3"],
            expected_judgment=["Yes", "Yes", "No"],
            domain=["empirical", "empirical", "empirical"],
        )

        assert rewards[0] > 0   # Correct
        assert rewards[1] < 0   # Wrong
        assert rewards[2] < 0   # Wrong

    def test_batch_empty_input(self):
        """Handles empty batch gracefully"""
        from src.training.enhanced_reward import reward_fn

        rewards = reward_fn(
            completions=[],
            prompts=[],
            expected_judgment=[],
        )

        assert rewards == []

    def test_batch_single_item(self):
        """Handles single-item batch"""
        from src.training.enhanced_reward import reward_fn

        rewards = reward_fn(
            completions=["Judgment: Yes"],
            prompts=["prompt"],
            expected_judgment=["Yes"],
            domain=["mathematical"],
        )

        assert len(rewards) == 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestRewardFunctionIntegration:
    """Integration tests ensuring all components work together"""

    def test_full_scoring_pipeline(
        self,
        sample_mathematical_completion_correct,
        sample_empirical_completion_with_citations,
        sample_philosophic_completion,
        sample_verification_chains
    ):
        """Full pipeline produces expected relative ordering"""
        from src.training.enhanced_reward import reward_fn

        # Build test batch
        completions = [
            sample_mathematical_completion_correct["completion"],
            sample_empirical_completion_with_citations["completion"],
            sample_philosophic_completion["completion"],
            "Wrong answer. Judgment: Yes",  # Wrong for philosophic
        ]

        rewards = reward_fn(
            completions=completions,
            prompts=["p"] * 4,
            expected_judgment=[
                sample_mathematical_completion_correct["expected_judgment"],
                sample_empirical_completion_with_citations["expected_judgment"],
                sample_philosophic_completion["expected_judgment"],
                "No",  # Actual is Yes, so wrong
            ],
            domain=[
                sample_mathematical_completion_correct["domain"],
                sample_empirical_completion_with_citations["domain"],
                sample_philosophic_completion["domain"],
                "philosophic",
            ],
            input_text=[
                sample_mathematical_completion_correct["input_text"],
                sample_empirical_completion_with_citations["input_text"],
                sample_philosophic_completion["input_text"],
                "Some input",
            ],
            citations=["[]"] * 4,
            verification_chain=[
                json.dumps(sample_verification_chains["mathematical"]),
                json.dumps(sample_verification_chains["empirical"]),
                json.dumps(sample_verification_chains["philosophic"]),
                "{}",
            ],
        )

        # First three should be positive (correct judgments)
        assert rewards[0] > 0
        assert rewards[1] > 0
        assert rewards[2] > 0

        # Fourth should be negative (wrong judgment)
        assert rewards[3] < 0

    def test_consistent_scoring(self):
        """Same input produces same output (deterministic)"""
        from src.training.enhanced_reward import reward_fn

        kwargs = {
            "completions": ["Analysis shows conditions met. Judgment: Yes"],
            "prompts": ["prompt"],
            "expected_judgment": ["Yes"],
            "domain": ["empirical"],
            "input_text": ["Evidence text"],
            "citations": ["[]"],
            "verification_chain": ["{}"],
        }

        reward1 = reward_fn(**kwargs)
        reward2 = reward_fn(**kwargs)

        assert reward1 == reward2


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_very_long_completion(self):
        """Handles very long completions"""
        from src.training.enhanced_reward import compute_reward

        long_completion = "Analysis. " * 1000 + "Judgment: Yes"
        reward = compute_reward(
            completion=long_completion,
            expected_judgment="Yes",
            domain="empirical",
            input_text="",
            citations=[],
            verification_chain={}
        )

        assert isinstance(reward, float)
        assert reward > 0  # Correct judgment

    def test_very_short_completion(self):
        """Handles very short completions"""
        from src.training.enhanced_reward import compute_reward

        short_completion = "Yes"
        reward = compute_reward(
            completion=short_completion,
            expected_judgment="Yes",
            domain="empirical",
            input_text="",
            citations=[],
            verification_chain={}
        )

        assert isinstance(reward, float)
        # May have length penalty

    def test_empty_completion(self):
        """Handles empty completion"""
        from src.training.enhanced_reward import compute_reward

        reward = compute_reward(
            completion="",
            expected_judgment="Yes",
            domain="empirical",
            input_text="",
            citations=[],
            verification_chain={}
        )

        assert isinstance(reward, float)
        assert reward <= 0  # No judgment + length penalty

    def test_unicode_in_completion(self):
        """Handles unicode characters"""
        from src.training.enhanced_reward import compute_reward

        completion = "The analysis shows \u03c0 (pi) is irrational. Judgment: Yes"
        reward = compute_reward(
            completion=completion,
            expected_judgment="Yes",
            domain="mathematical",
            input_text="Is pi irrational?",
            citations=[],
            verification_chain={}
        )

        assert isinstance(reward, float)

    def test_multiple_judgment_markers(self):
        """Handles completion with multiple judgment markers (uses last)"""
        from src.training.enhanced_reward import score_correctness

        completion = """
        Initial assessment: Judgment: No
        After further review...
        Final Judgment: Yes
        """

        # Should extract the LAST judgment
        score = score_correctness(completion, "Yes")
        # Behavior depends on implementation - document expected behavior
        assert isinstance(score, float)

    def test_special_characters_in_evidence(self):
        """Handles special characters in evidence text"""
        from src.training.enhanced_reward import score_evidence_grounding

        completion = 'The evidence shows "temp=25\u00b0C; p=101kPa" supports the claim.'
        input_text = "Temperature: 25\u00b0C, Pressure: 101kPa measured."

        score = score_evidence_grounding(
            completion=completion,
            input_text=input_text,
            citations=[]
        )

        assert isinstance(score, float)


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Tests for helper/utility functions in the reward module"""

    def test_extract_judgment_function(self):
        """extract_judgment returns normalized judgment or None"""
        from src.training.enhanced_reward import extract_judgment

        assert extract_judgment("Judgment: Yes") == "Yes"
        assert extract_judgment("Judgment: No") == "No"
        assert extract_judgment("Judgment: Insufficient") == "Insufficient"
        assert extract_judgment("No judgment here") is None

    def test_extract_quotes_function(self):
        """extract_quotes finds quoted strings"""
        from src.training.enhanced_reward import extract_quotes

        text = 'He said "hello" and she replied "goodbye".'
        quotes = extract_quotes(text)

        assert "hello" in quotes
        assert "goodbye" in quotes

    def test_extract_quotes_empty(self):
        """extract_quotes returns empty list when no quotes"""
        from src.training.enhanced_reward import extract_quotes

        text = "No quotes in this text."
        quotes = extract_quotes(text)

        assert quotes == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
