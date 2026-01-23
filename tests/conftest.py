"""
Pytest fixtures for Cognitive Organism test suite.

Provides sample data, mock objects, and utilities for testing
coverage analysis and evidence grounding modules.
"""

import pytest
from typing import Dict, List, Any
from dataclasses import dataclass


# =============================================================================
# Domain Constants
# =============================================================================

# All domains from src/domains.py - 35 principled domains
DOMAINS = [
    # Formal mode (3)
    "pure_mathematics",
    "formal_logic",
    "set_theory",
    # Empirical - Physical (4)
    "physics",
    "chemistry",
    "astronomy",
    "geology",
    # Empirical - Biological (2)
    "biochemistry",
    "biology",
    # Empirical - Sensitive (2)
    "neuroscience",
    "animal_cognition",
    # Empirical - Intelligent (6)
    "experimental_psychology",
    "developmental_psychology",
    "sociology",
    "anthropology",
    "economics",
    "history",
    # Common Sense (5)
    "everyday_practical",
    "social_common_sense",
    "professional_judgment",
    "generational_folk_wisdom",
    "subcultural_knowledge",
    # Dialectical (2)
    "philosophical",
    "pop_science",
    # Adversarial contexts (6)
    "deliberate_deception",
    "manipulation",
    "propaganda",
    "sarcasm_irony",
    "adversarial_debate",
    "strategic_ambiguity",
    # Edge cases (5)
    "self_reference",
    "category_errors",
    "vagueness_sorites",
    "counterfactuals",
    "future_contingents",
]
JUDGMENT_TYPES = ["Yes", "No", "Insufficient"]
DIFFICULTY_LEVELS = [1, 2, 3, 4, 5]


# =============================================================================
# Sample Data Classes (for testing before implementation)
# =============================================================================


@dataclass
class SampleExample:
    """Minimal training example structure for coverage testing."""

    proposition: str
    evidence: str
    domain: str
    judgment: str  # Yes, No, Insufficient
    difficulty: int  # 1-5
    has_distractor: bool = False
    evidence_citations: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.evidence_citations is None:
            self.evidence_citations = []


# =============================================================================
# Coverage Analyzer Fixtures
# =============================================================================


@pytest.fixture
def empty_dataset() -> List[SampleExample]:
    """Empty dataset - all coverage cells should be 0."""
    return []


@pytest.fixture
def single_example() -> List[SampleExample]:
    """Single example dataset."""
    return [
        SampleExample(
            proposition="2 + 2 = 4",
            evidence="By the axioms of arithmetic...",
            domain="pure_mathematics",
            judgment="Yes",
            difficulty=1,
            has_distractor=False,
        )
    ]


@pytest.fixture
def unbalanced_dataset() -> List[SampleExample]:
    """Dataset heavily clustered in one domain/judgment/difficulty."""
    examples = []
    # 20 examples all in physics/Yes/difficulty 2
    for i in range(20):
        examples.append(
            SampleExample(
                proposition=f"Physics proposition {i}",
                evidence=f"The experiment showed result {i}.",
                domain="physics",
                judgment="Yes",
                difficulty=2,
                has_distractor=False,
            )
        )
    # Only 1 example each for other combinations
    examples.append(
        SampleExample(
            proposition="Mathematical proposition",
            evidence="By definition...",
            domain="pure_mathematics",
            judgment="No",
            difficulty=3,
            has_distractor=False,
        )
    )
    examples.append(
        SampleExample(
            proposition="Common sense proposition",
            evidence="In this situation...",
            domain="everyday_practical",
            judgment="Insufficient",
            difficulty=1,
            has_distractor=True,
        )
    )
    return examples


@pytest.fixture
def balanced_dataset() -> List[SampleExample]:
    """Well-balanced dataset across dimensions."""
    examples = []
    # Create examples across multiple dimensions using new domain names
    for domain in ["pure_mathematics", "physics", "everyday_practical"]:
        for judgment in ["Yes", "No", "Insufficient"]:
            for difficulty in [1, 2, 3]:
                for has_distractor in [True, False]:
                    examples.append(
                        SampleExample(
                            proposition=f"{domain} proposition ({judgment}, d{difficulty})",
                            evidence=f"Evidence for {domain}...",
                            domain=domain,
                            judgment=judgment,
                            difficulty=difficulty,
                            has_distractor=has_distractor,
                        )
                    )
    return examples


@pytest.fixture
def coverage_dimensions() -> Dict[str, List[Any]]:
    """Standard coverage dimensions for testing."""
    return {
        "domain": DOMAINS,
        "judgment": JUDGMENT_TYPES,
        "difficulty": DIFFICULTY_LEVELS,
        "has_distractor": [True, False],
    }


@pytest.fixture
def make_example():
    """Factory fixture for creating test examples."""

    def _make_example(
        domain: str = "physics",
        judgment: str = "Yes",
        difficulty: int = 2,
        has_distractor: bool = False,
        **kwargs,
    ) -> SampleExample:
        return SampleExample(
            proposition=kwargs.get("proposition", f"Test proposition ({domain})"),
            evidence=kwargs.get("evidence", f"Test evidence for {domain}"),
            domain=domain,
            judgment=judgment,
            difficulty=difficulty,
            has_distractor=has_distractor,
            evidence_citations=kwargs.get("evidence_citations", []),
        )

    return _make_example


@pytest.fixture
def make_dataset(make_example):
    """Factory fixture for creating test datasets."""

    def _make_dataset(
        n: int = 10,
        domain_dist: Dict[str, int] = None,
        judgment_dist: Dict[str, int] = None,
    ) -> List[SampleExample]:
        examples = []

        if domain_dist:
            for domain, count in domain_dist.items():
                for _ in range(count):
                    examples.append(make_example(domain=domain))
        else:
            for i in range(n):
                examples.append(make_example())

        return examples

    return _make_dataset


# =============================================================================
# Distractor Generator Fixtures
# =============================================================================

from enum import Enum


class DistractorType(Enum):
    """Types of distractor completions for contrastive learning."""

    MISALIGNED_PHASE = "misaligned_phase"
    INVERTED_JUDGMENT = "inverted_judgment"
    REASONING_ERROR = "reasoning_error"
    EVIDENCE_HALLUCINATION = "evidence_hallucination"
    DOMAIN_MISMATCH = "domain_mismatch"


@dataclass
class JudgmentSample:
    """Extended judgment sample for distractor generation testing."""

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


# -----------------------------------------------------------------------------
# P3 Correct Completions (Judgment Level - proper commitments)
# -----------------------------------------------------------------------------


@pytest.fixture
def p3_judgment_yes_mathematical() -> str:
    """A correct P3 judgment completion affirming a mathematical proposition."""
    return """Condition fulfillment:
- The formula n(n+1)/2 is valid for summing first n positive integers: Fulfilled
- The arithmetic calculation 100(101)/2 = 5050 is correct: Fulfilled
- The integers being summed are positive: Fulfilled

The conditions for the virtually unconditioned are met. The evidence demonstrates that
Gauss's formula applies and the arithmetic is verified.

Judgment: Yes

The proposition is affirmed because all conditions linking it to truth are fulfilled
in the evidence provided. This is a mathematical judgment where fulfillment is found
in the formal coherence of the proof."""


@pytest.fixture
def p3_judgment_no_empirical() -> str:
    """A correct P3 judgment completion denying an empirical proposition."""
    return """Condition fulfillment:
- The experiment was properly controlled: Fulfilled
- The sample size was statistically significant: Fulfilled
- The results showed the predicted correlation: Not fulfilled
- The measurements were within acceptable error margins: Fulfilled

The conditions for affirmation are NOT met. While the methodology was sound,
the data shows no correlation between the variables.

Judgment: No

The proposition is denied because the condition requiring positive correlation
was not fulfilled in the experimental data. The measurement showed r = 0.02,
which is not statistically significant."""


@pytest.fixture
def p3_judgment_insufficient_philosophic() -> str:
    """A correct P3 judgment completion withholding on a philosophic proposition."""
    return """Condition fulfillment:
- The position is clearly articulated: Fulfilled
- The performative self-consistency can be assessed: Not determined
- The counterposition test is applicable: Not determined

The virtually unconditioned cannot be grasped. The position requires
further analysis to determine whether affirming it presupposes what it denies.

Judgment: Insufficient

I withhold judgment because the conditions for determining self-consistency
cannot be verified from the evidence provided. More analysis is needed to
determine whether this is a position or counterposition."""


@pytest.fixture
def p3_completions(
    p3_judgment_yes_mathematical,
    p3_judgment_no_empirical,
    p3_judgment_insufficient_philosophic,
) -> Dict[str, str]:
    """Dictionary of all P3 completions by type."""
    return {
        "yes_mathematical": p3_judgment_yes_mathematical,
        "no_empirical": p3_judgment_no_empirical,
        "insufficient_philosophic": p3_judgment_insufficient_philosophic,
    }


# -----------------------------------------------------------------------------
# P2 Completions (Understanding Level - hypotheses, NO judgment)
# -----------------------------------------------------------------------------


@pytest.fixture
def p2_understanding_hypothesis() -> str:
    """A P2 understanding completion - generates hypotheses without judgment."""
    return """The data suggests several possible interpretations:

Perhaps the correlation indicates a causal relationship, though this would
require further investigation. It might be that the observed pattern reflects
an underlying mechanism we haven't yet identified.

Another possibility is that this could indicate measurement artifacts rather
than a genuine phenomenon. The pattern might emerge from the particular
sampling methodology used.

These hypotheses warrant further inquiry. What additional evidence would
help us determine which interpretation is correct?"""


@pytest.fixture
def p2_understanding_exploration() -> str:
    """A P2 completion exploring possibilities without commitment."""
    return """Considering the evidence, we can formulate several hypotheses:

1. This could be explained by factor A affecting variable X
2. Alternatively, the relationship might be mediated by B
3. It's also possible that the correlation is spurious

The question remains open. More data would help clarify which of these
possibilities best accounts for the observations. Each hypothesis suggests
different predictions that could be tested."""


@pytest.fixture
def p2_understanding_question() -> str:
    """A P2 completion ending with questions rather than judgment."""
    return """The pattern in the data is intriguing and suggests several lines
of inquiry:

What mechanism could produce this distribution? Is this consistent with
the theoretical predictions? Could there be confounding variables we haven't
considered?

Further investigation is needed to understand what's happening here. The
relationship between these variables remains to be determined."""


@pytest.fixture
def p2_completions(
    p2_understanding_hypothesis, p2_understanding_exploration, p2_understanding_question
) -> Dict[str, str]:
    """Dictionary of P2 completions."""
    return {
        "hypothesis": p2_understanding_hypothesis,
        "exploration": p2_understanding_exploration,
        "question": p2_understanding_question,
    }


# -----------------------------------------------------------------------------
# P2/P3 Language Markers for Detection
# -----------------------------------------------------------------------------


@pytest.fixture
def p2_markers() -> List[str]:
    """Language markers characteristic of P2 (Understanding) level."""
    return [
        "perhaps",
        "might be",
        "might",
        "could be",
        "could indicate",
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
    ]


@pytest.fixture
def p3_markers() -> List[str]:
    """Language markers characteristic of P3 (Judgment) level."""
    return [
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
    ]


# -----------------------------------------------------------------------------
# Complete JudgmentSample Fixtures for Distractor Generation
# -----------------------------------------------------------------------------


@pytest.fixture
def judgment_sample_mathematical_yes() -> JudgmentSample:
    """Complete judgment sample for mathematical domain with Yes judgment."""
    return JudgmentSample(
        proposition="The sum of angles in a triangle equals 180 degrees.",
        evidence="""Given: Triangle ABC with angles alpha, beta, gamma.
Construction: Draw line DE parallel to BC through point A.
By the parallel postulate, angle DAB = angle ABC (alternate interior angles).
Similarly, angle EAC = angle ACB.
Since DAE is a straight line, angle DAB + angle BAC + angle EAC = 180 degrees.
Therefore, angle ABC + angle BAC + angle ACB = 180 degrees. QED.""",
        conditions=[
            "The proof uses valid geometric axioms",
            "Each step follows logically from the previous",
            "The construction is valid in Euclidean geometry",
            "The conclusion matches the proposition",
        ],
        conditions_fulfilled={
            "The proof uses valid geometric axioms": True,
            "Each step follows logically from the previous": True,
            "The construction is valid in Euclidean geometry": True,
            "The conclusion matches the proposition": True,
        },
        judgment="Yes",
        judgment_reasoning="All conditions are fulfilled through formal proof.",
        domain="mathematical",
        difficulty=2,
    )


@pytest.fixture
def judgment_sample_empirical_no() -> JudgmentSample:
    """Complete judgment sample for empirical domain with No judgment."""
    return JudgmentSample(
        proposition="The experimental treatment significantly improves reaction time.",
        evidence="""The study measured reaction times across 150 participants.
The control group (n=75) showed a mean reaction time of 245ms (SD=32).
The experimental group (n=75) showed a mean reaction time of 238ms (SD=29).
Statistical analysis: t(148) = 1.42, p = 0.158
Effect size: Cohen's d = 0.23""",
        conditions=[
            "The study has adequate statistical power",
            "The p-value is below the significance threshold (p < 0.05)",
            "The effect size is practically meaningful",
            "The experimental design is valid",
        ],
        conditions_fulfilled={
            "The study has adequate statistical power": True,
            "The p-value is below the significance threshold (p < 0.05)": False,
            "The effect size is practically meaningful": False,
            "The experimental design is valid": True,
        },
        judgment="No",
        judgment_reasoning="The p-value of 0.158 exceeds the significance threshold.",
        domain="empirical",
        difficulty=3,
    )


@pytest.fixture
def judgment_sample_pop_science_no() -> JudgmentSample:
    """Complete judgment sample for pop science domain with No judgment."""
    return JudgmentSample(
        proposition="Drinking coffee before bed improves sleep quality.",
        evidence="""A new study suggests that drinking coffee before bed improves
sleep quality. Scientists at the Institute found that 7 out of 10 participants
reported feeling more rested. The study was funded by CoffeeCorp Inc.
Many celebrities have already adopted this technique.""",
        conditions=[
            "The study uses objective sleep quality measures",
            "The sample size is statistically adequate",
            "There is no conflict of interest",
            "The results have been replicated independently",
        ],
        conditions_fulfilled={
            "The study uses objective sleep quality measures": False,
            "The sample size is statistically adequate": False,
            "There is no conflict of interest": False,
            "The results have been replicated independently": False,
        },
        judgment="No",
        judgment_reasoning="Classic pop science: appeals to authority, conflicts of interest, no rigorous methodology.",
        domain="pop_science",
        difficulty=2,
    )


@pytest.fixture
def judgment_sample_philosophic_yes() -> JudgmentSample:
    """Complete judgment sample for philosophic domain (counterposition detection)."""
    return JudgmentSample(
        proposition="The claim 'all knowledge is merely subjective opinion' is a counterposition.",
        evidence="""Consider the position: "All knowledge is merely subjective opinion."
If this statement is merely subjective opinion, then there is no reason to accept it.
If it is objectively true, then it contradicts itself by being non-subjective knowledge.
The proponent must either apply it to itself (undermining it) or exempt it (special pleading).""",
        conditions=[
            "The position is clearly stated",
            "The self-referential test is applicable",
            "Applying the position to itself reveals incoherence",
            "There is no way to hold the position without performative contradiction",
        ],
        conditions_fulfilled={
            "The position is clearly stated": True,
            "The self-referential test is applicable": True,
            "Applying the position to itself reveals incoherence": True,
            "There is no way to hold the position without performative contradiction": True,
        },
        judgment="Yes",
        judgment_reasoning="The position is self-defeating - affirming it reasonably presupposes what it denies.",
        domain="philosophic",
        difficulty=4,
    )


@pytest.fixture
def judgment_sample_insufficient() -> JudgmentSample:
    """Judgment sample where evidence is genuinely insufficient."""
    return JudgmentSample(
        proposition="The new medication reduces blood pressure more effectively than the current standard.",
        evidence="""A preliminary trial with 15 participants showed promising results.
The researchers reported "encouraging trends" but noted that the full dataset
is still being analyzed. No statistical analysis was provided in the abstract.
The study is ongoing and expected to conclude next year.""",
        conditions=[
            "A controlled comparison with standard treatment was conducted",
            "The sample size provides adequate statistical power",
            "Statistical significance was demonstrated",
            "The effect size is clinically meaningful",
        ],
        conditions_fulfilled={
            "A controlled comparison with standard treatment was conducted": True,
            "The sample size provides adequate statistical power": False,
            "Statistical significance was demonstrated": False,
            "The effect size is clinically meaningful": False,
        },
        judgment="Insufficient",
        judgment_reasoning="Cannot determine fulfillment - awaiting complete data and analysis.",
        domain="empirical",
        difficulty=2,
    )


@pytest.fixture
def judgment_samples(
    judgment_sample_mathematical_yes,
    judgment_sample_empirical_no,
    judgment_sample_pop_science_no,
    judgment_sample_philosophic_yes,
    judgment_sample_insufficient,
) -> Dict[str, JudgmentSample]:
    """Dictionary of all complete judgment samples."""
    return {
        "mathematical_yes": judgment_sample_mathematical_yes,
        "empirical_no": judgment_sample_empirical_no,
        "pop_science_no": judgment_sample_pop_science_no,
        "philosophic_yes": judgment_sample_philosophic_yes,
        "insufficient": judgment_sample_insufficient,
    }


# -----------------------------------------------------------------------------
# Mixed/Problematic Cases for Testing Detection
# -----------------------------------------------------------------------------


@pytest.fixture
def mixed_p2_p3_text() -> str:
    """Text that mixes P2 and P3 language (problematic - should be detected)."""
    return """The data suggests several possibilities. Perhaps the correlation
indicates causation, though this might require further investigation.

On the other hand, the conditions appear to be fulfilled:
- Condition 1: Fulfilled
- Condition 2: Fulfilled

Judgment: Yes

This could be wrong though, it's just my hypothesis."""


@pytest.fixture
def borderline_p3_weak_commitment() -> str:
    """P3 structure but weak commitment language (should be flagged)."""
    return """Condition fulfillment:
- Condition 1: Probably fulfilled
- Condition 2: Seems fulfilled

I think the judgment might be Yes, but I'm not entirely sure. The evidence
sort of supports the proposition, so I'll tentatively affirm it.

Judgment: Yes (tentative)"""


@pytest.fixture
def p3_with_p2_hedging() -> str:
    """P3 completion that hedges inappropriately (distractor candidate)."""
    return """Condition fulfillment:
- Condition 1: Fulfilled
- Condition 2: Fulfilled

The conditions appear to be met. Perhaps we can say...

Judgment: Yes

Though it might be otherwise if we considered different interpretations."""


# -----------------------------------------------------------------------------
# Input Text Fixtures for Evidence Hallucination Testing
# -----------------------------------------------------------------------------


@pytest.fixture
def input_text_empirical_study() -> str:
    """Input text for empirical study - for hallucination detection."""
    return """The randomized controlled trial enrolled 200 adult participants.
The experimental group (n=100) received the intervention for 8 weeks.
The control group (n=100) received standard care.

Primary outcome: The experimental group showed a mean improvement of 12.3 points
on the depression scale (95% CI: 9.1-15.5, p < 0.001).

Secondary outcome: Sleep quality improved by 18% in the experimental group
compared to 3% in the control group.

Adverse events: Two participants in the experimental group reported mild headaches.
No serious adverse events were recorded."""


@pytest.fixture
def input_text_mathematical_proof() -> str:
    """Input text for mathematical proof - for hallucination detection."""
    return """Theorem: For all integers n >= 1, the sum 1 + 2 + ... + n = n(n+1)/2.

Proof by induction:
Base case (n=1): LHS = 1, RHS = 1(2)/2 = 1. Verified.

Inductive step: Assume true for n=k. Then:
1 + 2 + ... + k + (k+1) = k(k+1)/2 + (k+1)
                        = (k+1)(k/2 + 1)
                        = (k+1)(k+2)/2

Thus the formula holds for n=k+1.
By mathematical induction, the formula holds for all n >= 1. QED."""


# -----------------------------------------------------------------------------
# Factory for JudgmentSample Creation
# -----------------------------------------------------------------------------


@pytest.fixture
def make_judgment_sample():
    """Factory for creating JudgmentSample instances."""

    def _make(
        judgment: str = "Yes", domain: str = "empirical", difficulty: int = 2, **kwargs
    ) -> JudgmentSample:
        defaults = {
            "proposition": f"Test proposition for {domain} domain",
            "evidence": f"Test evidence supporting the proposition in {domain} context.",
            "conditions": [
                f"Condition 1 for {domain}",
                f"Condition 2 for {domain}",
            ],
            "conditions_fulfilled": {
                f"Condition 1 for {domain}": judgment == "Yes",
                f"Condition 2 for {domain}": judgment != "No",
            },
            "judgment_reasoning": f"Standard reasoning for {judgment} judgment in {domain}.",
        }
        defaults.update(kwargs)
        return JudgmentSample(
            judgment=judgment, domain=domain, difficulty=difficulty, **defaults
        )

    return _make


# =============================================================================
# Extended Schema Fixtures (for test_schema.py TDD tests)
# =============================================================================

from typing import Optional


@pytest.fixture
def sample_factory():
    """Factory for creating test samples with customizable fields for EnhancedJudgmentSample."""

    def _create_sample(
        proposition: str = "Test proposition",
        domain: str = "empirical",
        evidence_text: str = "Test evidence text with data points.",
        conditions: Optional[List[str]] = None,
        expected_judgment: str = "yes",
        difficulty: int = 2,
        evidence_citations: Optional[List[Dict]] = None,
        verification_chain: Optional[Dict] = None,
        distractor_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a sample data dict matching EnhancedJudgmentSample fields."""
        return {
            "proposition": proposition,
            "domain": domain,
            "evidence_text": evidence_text,
            "conditions": conditions or ["Condition 1", "Condition 2"],
            "expected_judgment": expected_judgment,
            "difficulty": difficulty,
            "evidence_citations": evidence_citations or [],
            "verification_chain": verification_chain or {},
            "distractor_type": distractor_type,
        }

    return _create_sample


# -----------------------------------------------------------------------------
# Domain-Specific Sample Data Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mathematical_proposition() -> str:
    """A proposition requiring mathematical verification."""
    return "The sum of angles in a Euclidean triangle equals 180 degrees."


@pytest.fixture
def mathematical_evidence_text() -> str:
    """Evidence for mathematical judgment."""
    return """
    Axiom 1: Through any two points, exactly one line can be drawn.
    Axiom 2: A line segment can be extended indefinitely in either direction.
    Axiom 3: Given a point and a radius, a circle can be drawn.
    Axiom 4: All right angles are equal.
    Axiom 5 (Parallel Postulate): Through a point not on a line, exactly one parallel can be drawn.

    Proof: Draw a line through vertex C parallel to base AB.
    By the parallel postulate and alternate interior angles:
    - Angle at A = angle on left of C
    - Angle at B = angle on right of C
    The angles at C form a straight line (180 degrees).
    Therefore, angles A + B + C = 180 degrees. QED.
    """


@pytest.fixture
def mathematical_conditions() -> List[str]:
    """Conditions for mathematical proposition to be true."""
    return [
        "Euclidean axioms are assumed",
        "Parallel postulate holds",
        "Proof steps are logically valid",
        "Conclusion follows from premises",
    ]


@pytest.fixture
def mathematical_sample_data(
    mathematical_proposition,
    mathematical_evidence_text,
    mathematical_conditions,
) -> Dict[str, Any]:
    """Complete mathematical judgment sample data dict."""
    return {
        "proposition": mathematical_proposition,
        "domain": "mathematical",
        "evidence_text": mathematical_evidence_text,
        "conditions": mathematical_conditions,
        "expected_judgment": "yes",
        "difficulty": 2,
        "evidence_citations": [
            {
                "text": "angles A + B + C = 180 degrees. QED",
                "start_idx": 512,
                "end_idx": 548,
                "supports_condition": "Conclusion follows from premises",
            }
        ],
        "verification_chain": {
            "domain": "mathematical",
            "verification_type": "proof",
            "required_elements": [
                "axioms_referenced",
                "definitions_used",
                "logical_steps",
                "conclusion_marker",
            ],
            "proof_steps": [
                "Draw parallel through C",
                "Apply alternate interior angles",
                "Sum angles at C",
            ],
            "axiom_refs": ["Parallel Postulate", "Alternate Interior Angles Theorem"],
            "qed_marker": True,
        },
        "distractor_type": None,
    }


@pytest.fixture
def empirical_sample_data() -> Dict[str, Any]:
    """Complete empirical judgment sample data dict."""
    return {
        "proposition": "The new drug reduces blood pressure by at least 10mmHg on average.",
        "domain": "empirical",
        "evidence_text": """
    Study Design: Double-blind randomized controlled trial
    Sample Size: n=500 (250 treatment, 250 placebo)
    Duration: 12 weeks

    Results:
    - Treatment group: Mean reduction 14.2mmHg (95% CI: 12.1-16.3)
    - Placebo group: Mean reduction 2.1mmHg (95% CI: 0.8-3.4)
    - Difference: 12.1mmHg (p < 0.001)

    Methodology: Blood pressure measured by calibrated sphygmomanometer
    at baseline and week 12. Blinding verified by post-study questionnaire.

    Replication: Results consistent with two prior Phase 2 trials.
    """,
        "conditions": [
            "Study methodology is sound",
            "Sample size provides adequate statistical power",
            "Measured difference exceeds 10mmHg",
            "Results are statistically significant",
            "Findings are replicated",
        ],
        "expected_judgment": "yes",
        "difficulty": 3,
        "evidence_citations": [
            {
                "text": "Mean reduction 14.2mmHg (95% CI: 12.1-16.3)",
                "start_idx": 180,
                "end_idx": 224,
                "supports_condition": "Measured difference exceeds 10mmHg",
            },
            {
                "text": "p < 0.001",
                "start_idx": 295,
                "end_idx": 304,
                "supports_condition": "Results are statistically significant",
            },
        ],
        "verification_chain": {
            "domain": "empirical",
            "verification_type": "experimental",
            "required_elements": [
                "methodology_stated",
                "data_points_cited",
                "replication_status",
                "measurement_precision",
            ],
            "data_points": ["14.2mmHg reduction", "p < 0.001", "n=500"],
            "methodology": "Double-blind RCT",
            "replication": "Consistent with Phase 2 trials",
        },
        "distractor_type": None,
    }


@pytest.fixture
def pop_science_sample_data() -> Dict[str, Any]:
    """Complete pop science judgment sample data dict - detecting counterposition."""
    return {
        "proposition": "Drinking red wine prevents heart disease according to scientists.",
        "domain": "pop_science",
        "evidence_text": """
    News headline: "Red Wine: The Heart-Healthy Drink Scientists Love"

    Article claims:
    - "Studies show people who drink red wine have healthier hearts"
    - "Resveratrol in red wine has antioxidant properties"
    - "The Mediterranean diet includes wine and is linked to longevity"

    Actual study referenced (buried in paragraph 8):
    - Observational study, not controlled trial
    - Correlation between moderate wine consumption and cardiac outcomes
    - Did not control for lifestyle factors, diet, or socioeconomic status
    - Authors note: "These findings do not establish causation"

    Expert quote: "More research is needed before making recommendations."
    """,
        "conditions": [
            "Causal mechanism demonstrated (not just correlation)",
            "Controlled experimental evidence exists",
            "Confounding variables accounted for",
            "Scientific consensus supports the claim",
        ],
        "expected_judgment": "no",
        "difficulty": 3,
        "evidence_citations": [
            {
                "text": "These findings do not establish causation",
                "start_idx": 420,
                "end_idx": 460,
                "supports_condition": "Causal mechanism demonstrated (not just correlation)",
            },
        ],
        "verification_chain": {
            "domain": "pop_science",
            "verification_type": "counterposition_detection",
            "required_elements": [
                "claimed_proof_identified",
                "actual_evidence_examined",
                "conditions_unfulfilled_listed",
                "verdict_on_claim",
            ],
            "unfulfilled_claims": [
                "Causation claimed from correlation",
                "Confounders not controlled",
            ],
            "actual_evidence_level": "observational correlation only",
        },
        "distractor_type": None,
    }


@pytest.fixture
def philosophic_sample_data() -> Dict[str, Any]:
    """Complete philosophic judgment sample data dict - counterposition detection."""
    return {
        "proposition": "All knowledge is merely subjective opinion with no objective validity.",
        "domain": "philosophic",
        "evidence_text": """
    The claim "All knowledge is merely subjective opinion" invites analysis:

    1. The claim itself purports to be knowledge about knowledge.
    2. If true, the claim itself is merely subjective opinion.
    3. As merely subjective opinion, it has no objective validity.
    4. Therefore, we have no reason to accept it as true.

    The claim is performatively self-defeating:
    - Affirming it reasonably presupposes that the affirmation has
      objective validity (is not merely subjective opinion).
    - But this contradicts what is being affirmed.

    This is a counterposition: it cannot be held coherently by one
    who grasps it intelligently and affirms it reasonably.
    """,
        "conditions": [
            "The position can be affirmed without self-contradiction",
            "Affirming the position does not presuppose what it denies",
            "The position is coherent with the activity of knowing it",
        ],
        "expected_judgment": "no",
        "difficulty": 4,
        "evidence_citations": [
            {
                "text": "performatively self-defeating",
                "start_idx": 280,
                "end_idx": 309,
                "supports_condition": "Affirming the position does not presuppose what it denies",
            },
        ],
        "verification_chain": {
            "domain": "philosophic",
            "verification_type": "performative_consistency",
            "required_elements": [
                "position_stated",
                "conditions_of_affirmation",
                "self_consistency_check",
                "counterposition_test",
            ],
            "performative_check": "Affirming requires what is denied",
            "self_consistency": False,
            "counterposition_test": "Position is self-defeating",
        },
        "distractor_type": None,
    }


# -----------------------------------------------------------------------------
# Verification Chain Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mathematical_verification_chain() -> Dict[str, Any]:
    """Verification chain for mathematical domain."""
    return {
        "domain": "mathematical",
        "verification_type": "proof",
        "required_elements": [
            "axioms_referenced",
            "definitions_used",
            "logical_steps",
            "conclusion_marker",
        ],
        "proof_steps": [
            "Step 1: Draw parallel",
            "Step 2: Apply theorem",
            "Step 3: Sum angles",
        ],
        "axiom_refs": ["Axiom 5 (Parallel Postulate)"],
        "qed_marker": True,
    }


@pytest.fixture
def empirical_verification_chain() -> Dict[str, Any]:
    """Verification chain for empirical domain."""
    return {
        "domain": "empirical",
        "verification_type": "experimental",
        "required_elements": [
            "methodology_stated",
            "data_points_cited",
            "replication_status",
            "measurement_precision",
        ],
        "data_points": ["14.2mmHg", "95% CI", "p < 0.001"],
        "methodology": "Double-blind RCT",
        "replication": "Phase 2 consistent",
    }


@pytest.fixture
def common_sense_verification_chain() -> Dict[str, Any]:
    """Verification chain for common sense domain."""
    return {
        "domain": "common_sense",
        "verification_type": "pragmatic",
        "required_elements": [
            "situation_described",
            "practical_outcome",
            "experience_cited",
        ],
        "practical_outcomes": ["Works in this situation"],
        "situation_factors": ["Time", "Distance", "Conditions"],
    }


@pytest.fixture
def pop_science_verification_chain() -> Dict[str, Any]:
    """Verification chain for pop science domain."""
    return {
        "domain": "pop_science",
        "verification_type": "counterposition_detection",
        "required_elements": [
            "claimed_proof_identified",
            "actual_evidence_examined",
            "conditions_unfulfilled_listed",
            "verdict_on_claim",
        ],
        "unfulfilled_claims": ["Causation from correlation", "No controls"],
        "actual_evidence_level": "observational",
    }


@pytest.fixture
def philosophic_verification_chain() -> Dict[str, Any]:
    """Verification chain for philosophic domain."""
    return {
        "domain": "philosophic",
        "verification_type": "performative_consistency",
        "required_elements": [
            "position_stated",
            "conditions_of_affirmation",
            "self_consistency_check",
            "counterposition_test",
        ],
        "performative_check": "Does affirming presuppose what is denied?",
        "self_consistency": False,
        "counterposition_test": "Position is self-defeating",
    }


# -----------------------------------------------------------------------------
# Distractor Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def misaligned_phase_distractor() -> Dict[str, Any]:
    """A P2 (Understanding) response to a P3 (Judgment) task."""
    return {
        "correct_completion": """
            Checking conditions:
            - Euclidean axioms assumed: Yes
            - Parallel postulate holds: Yes
            - Proof steps valid: Yes
            - Conclusion follows: Yes

            All conditions fulfilled.
            Judgment: Yes
        """,
        "distractor_completion": """
            This is an interesting mathematical claim. Let me explore some hypotheses:

            Hypothesis 1: The proposition might be true in Euclidean geometry.
            Hypothesis 2: It could be false in non-Euclidean geometries.
            Hypothesis 3: The proof might contain hidden assumptions.

            What if we considered spherical geometry? The angles would sum to more
            than 180 degrees. This raises interesting questions about the nature
            of geometric truth...
        """,
        "distractor_type": "misaligned_phase",
        "explanation": "Response generates hypotheses (P2) instead of judging (P3). "
        "Lacks commitment to Yes/No/Insufficient. Asks questions instead of answering.",
    }


@pytest.fixture
def inverted_judgment_distractor() -> Dict[str, Any]:
    """Correct-looking reasoning leading to wrong judgment."""
    return {
        "correct_completion": """
            Checking conditions:
            - Study methodology sound: Yes (double-blind RCT)
            - Adequate sample size: Yes (n=500)
            - Difference exceeds 10mmHg: Yes (14.2mmHg measured)
            - Statistically significant: Yes (p < 0.001)
            - Replicated: Yes (consistent with Phase 2)

            All conditions fulfilled.
            Judgment: Yes
        """,
        "distractor_completion": """
            Checking conditions:
            - Study methodology sound: Yes (double-blind RCT)
            - Adequate sample size: Yes (n=500)
            - Difference exceeds 10mmHg: Yes (14.2mmHg measured)
            - Statistically significant: Yes (p < 0.001)
            - Replicated: Yes (consistent with Phase 2)

            Although conditions appear fulfilled, there may be unknown confounders.
            The study duration of 12 weeks may be insufficient for long-term claims.

            Judgment: Insufficient
        """,
        "distractor_type": "inverted_judgment",
        "explanation": "Reasoning correctly identifies fulfilled conditions but then "
        "invents concerns not present in the conditions to justify wrong judgment.",
    }


@pytest.fixture
def evidence_hallucination_distractor() -> Dict[str, Any]:
    """Response that cites non-existent evidence."""
    return {
        "correct_completion": """
            The evidence states "Mean reduction 14.2mmHg" which exceeds the
            required 10mmHg threshold. The p-value of < 0.001 indicates
            statistical significance.

            Judgment: Yes
        """,
        "distractor_completion": """
            The evidence clearly shows "the drug achieved a 22mmHg reduction
            in the primary endpoint" and "all secondary endpoints were met
            with p < 0.0001". The FDA advisory committee voted unanimously
            for approval.

            Judgment: Yes
        """,
        "distractor_type": "evidence_hallucination",
        "explanation": "Citations (22mmHg, secondary endpoints, FDA committee) "
        "do not exist in the provided evidence. Fabricated support.",
    }


@pytest.fixture
def reasoning_error_distractor() -> Dict[str, Any]:
    """Correct judgment with logically flawed reasoning."""
    return {
        "correct_completion": """
            Checking conditions:
            - Euclidean axioms assumed: Yes (explicitly stated in proof)
            - Parallel postulate holds: Yes (used to draw parallel line)
            - Proof steps valid: Yes (alternate interior angles correctly applied)
            - Conclusion follows: Yes (angle sum at C = 180 degrees = straight line)

            Judgment: Yes
        """,
        "distractor_completion": """
            The proposition must be true because:
            - It has been known since ancient Greece
            - Euclid was a famous mathematician
            - My geometry teacher taught this
            - The proof looks complicated so it's probably right

            Judgment: Yes
        """,
        "distractor_type": "reasoning_error",
        "explanation": "Correct judgment (Yes) but reasoning is appeals to authority "
        "and tradition rather than checking condition fulfillment.",
    }


# -----------------------------------------------------------------------------
# Coverage Cell Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def coverage_cell_data() -> List[Dict[str, Any]]:
    """Sample coverage cells for testing cell_id computation."""
    return [
        {
            "domain": "mathematical",
            "judgment_type": "yes",
            "difficulty": 1,
            "has_distractor": False,
            "expected_cell_id": "mathematical_yes_1_positive",
        },
        {
            "domain": "empirical",
            "judgment_type": "no",
            "difficulty": 3,
            "has_distractor": True,
            "expected_cell_id": "empirical_no_3_distractor",
        },
        {
            "domain": "philosophic",
            "judgment_type": "insufficient",
            "difficulty": 5,
            "has_distractor": False,
            "expected_cell_id": "philosophic_insufficient_5_positive",
        },
    ]


# -----------------------------------------------------------------------------
# Packet Field Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def student_packet_fields() -> List[str]:
    """Fields that should be in student packet (model input)."""
    return ["prompt", "evidence", "conditions"]


@pytest.fixture
def evaluator_packet_fields() -> List[str]:
    """Fields that should be in evaluator packet (reward function)."""
    return [
        "expected_judgment",
        "domain",
        "difficulty",
        "evidence_citations",
        "distractor_type",
        "verification_chain",
    ]


# =============================================================================
# Evidence Grounding Fixtures (for test_evidence.py)
# =============================================================================


@pytest.fixture
def simple_evidence() -> str:
    """Simple evidence text for basic citation testing."""
    return """Water boils at 100 degrees Celsius at standard pressure.
    The experiment was conducted under controlled laboratory conditions.
    Temperature was measured using a calibrated thermometer."""


@pytest.fixture
def evidence_with_quotable_segments() -> str:
    """Evidence text with clear quotable segments for citation testing."""
    return """The study examined 150 participants over a 6-month period.
    Results showed a 95% correlation between exercise frequency and reported mood improvements.
    The sample size was adequate for statistical power (n=150, power=0.85).
    Measurements were taken at baseline, 3 months, and 6 months.
    All participants provided informed consent and completed the full protocol."""


@pytest.fixture
def input_text_for_evidence() -> str:
    """Realistic input text for evidence grounding tests."""
    return """The randomized controlled trial measured temperature at 25°C with humidity at 60%.
    Results showed a correlation coefficient of 0.85 between variables X and Y.
    The sample size was 150 participants, providing adequate statistical power.
    The p-value was 0.003, indicating statistical significance.
    The effect size (Cohen's d = 0.62) suggests a medium to large practical effect."""


@pytest.fixture
def completion_with_valid_citations() -> str:
    """Completion containing valid citations that exist in evidence_with_quotable_segments."""
    return """The evidence clearly demonstrates the relationship between the variables.
    The study reports "150 participants over a 6-month period" as the study design.
    Most importantly, "95% correlation between exercise frequency and reported mood improvements" indicates a strong relationship.
    The "sample size was adequate for statistical power" which ensures reliability.
    The measurements were systematic: "baseline, 3 months, and 6 months" timepoints.
    
    Judgment: Yes
    
    All conditions are fulfilled based on the cited evidence."""


@pytest.fixture
def completion_with_hallucinated_citations() -> str:
    """Completion containing citations that do NOT exist in the input."""
    return """The evidence shows compelling results for the hypothesis.
    The study measured "temperature at 30°C" under controlled conditions.
    Results demonstrated "correlation coefficient of 0.95" between the variables.
    The "sample size was 500 participants" ensuring robust statistical power.
    The analysis revealed "p-value was 0.0001" indicating high significance.
    
    Judgment: Yes
    
    The evidence strongly supports the proposition."""


@pytest.fixture
def completion_with_no_citations() -> str:
    """Completion without any citations."""
    return """The evidence supports the proposition through multiple lines of reasoning.
    The experimental design was sound and the methodology appropriate.
    The results clearly demonstrate the predicted relationship.
    Statistical analysis confirms the hypothesis.
    
    Judgment: Yes
    
    Based on the overall pattern of evidence, the proposition is affirmed."""


@pytest.fixture
def completion_with_mixed_citations() -> str:
    """Completion with both valid and hallucinated citations."""
    return """The study provides mixed evidence for the claim.
    Valid citation: "95% correlation between exercise frequency" shows a strong relationship.
    Hallucinated: "the effect was replicated in three independent studies" (not in evidence).
    Valid citation: "150 participants over a 6-month period" provides adequate power.
    Hallucinated: "the control group showed no correlation" (not mentioned).
    
    Judgment: Yes
    
    The valid citations support the proposition despite some unsupported claims."""


@pytest.fixture
def completion_with_paraphrases(evidence_with_quotable_segments) -> str:
    """Completion that paraphrases evidence rather than quoting directly."""
    return """The research tracked 150 subjects for six months.
    The findings revealed a ninety-five percent correlation between how often people exercised
    and their self-reported mood enhancements.
    The number of participants was sufficient for statistical analysis.
    Data collection occurred at the start, middle, and end of the study period."""


@pytest.fixture
def conditions_list() -> List[str]:
    """Standard conditions list for testing relevance."""
    return [
        "Temperature must be at least 100 degrees Celsius",
        "Pressure must be at standard atmospheric pressure",
        "Measurements must be taken with calibrated instruments",
        "The correlation coefficient must exceed 0.80",
    ]


@pytest.fixture
def mathematical_evidence() -> Dict[str, str]:
    """Mathematical domain evidence for testing."""
    return {
        "evidence": """Axiom 1: The sum of angles in a Euclidean triangle equals 180 degrees.
        Proof: Draw a line parallel to the base through the apex.
        By alternate interior angles, the angles at the apex equal the base angles.
        The three angles at the apex form a straight line (180 degrees).
        Therefore, the sum of the triangle's angles equals 180 degrees. QED.""",
        "expected_citation": "The sum of angles in a Euclidean triangle equals 180 degrees",
    }


@pytest.fixture
def empirical_evidence() -> Dict[str, str]:
    """Empirical domain evidence for testing."""
    return {
        "evidence": """Experimental Observation Log:
        Trial 1: Temperature = 100.0C, Pressure = 101.3 kPa, Result: Boiling observed
        Trial 2: Temperature = 100.1C, Pressure = 101.2 kPa, Result: Boiling observed
        Trial 3: Temperature = 99.9C, Pressure = 101.4 kPa, Result: Boiling observed
        Conclusion: Water boils at approximately 100C at standard pressure.""",
        "expected_citation": "Temperature = 100.0C",
    }


@pytest.fixture
def common_sense_evidence() -> Dict[str, str]:
    """Common sense domain evidence for testing."""
    return {
        "evidence": """Situation: You need to get to the airport by 3pm.
        Current time: 1:30pm
        Distance: 15 miles
        Traffic conditions: Heavy traffic expected on main highway
        Alternative route: Side roads, 18 miles but less traffic
        Past experience: Highway takes 45-60 minutes in heavy traffic, side roads take 35-40 minutes.""",
        "expected_citation": "side roads take 35-40 minutes",
    }


@pytest.fixture
def pop_science_evidence() -> Dict[str, str]:
    """Pop science domain evidence for testing."""
    return {
        "evidence": """Article: "New Study Shows Coffee Cures Cancer!"
        The study observed 50 coffee drinkers and 50 non-drinkers over 6 months.
        Coffee drinkers reported feeling healthier (subjective self-report).
        No control for diet, exercise, genetics, or other lifestyle factors.
        The correlation between coffee drinking and health is undeniable, say researchers.
        Study funded by the National Coffee Association.""",
        "expected_unfulfilled": "No control for diet, exercise, genetics, or other lifestyle factors",
    }


@pytest.fixture
def philosophic_evidence() -> Dict[str, str]:
    """Philosophic domain evidence for testing."""
    return {
        "evidence": """Position: "All truth is relative to cultural context."
        Analysis: If this position is true, then it is true relative to some cultural context.
        But then it is not absolutely true - it is only true within that context.
        If it is absolutely true, then it contradicts itself by being non-relative truth.
        The position is self-referentially incoherent when applied to itself.
        Affirming it reasonably presupposes non-relative standards of reasoning.""",
        "expected_citation": "self-referentially incoherent",
    }
