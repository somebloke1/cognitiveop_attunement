# Unified Implementation Plan: Test Architecture Improvements

*Generated: 2026-01-20*

## Executive Summary

This document presents a coherent implementation plan for six validated insights from the critical reflection on test architecture improvements. These insights passed the virtually unconditioned test: their conditions for being valuable improvements were identified as fulfilled.

**Selected Insights (Judgment: YES):**
1. Insight 2: Contrastive Distractor Learning
2. Insight 3: Evidence Grounding Requirement
3. Insight 4: Graduated Difficulty Scaffolding
4. Insight 7: Domain-Specific Verification Chains
5. Insight 9: Dual-Packet Architecture
6. Insight 10: Coverage Enforcement Matrix

**Core Principle**: Form without substance is empty form. These improvements operationalize Lonergan's cognitive theory in training infrastructure, ensuring the model learns genuine operational habits rather than surface imitation.

---

## Architecture Diagram

```
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                     DATA GENERATION LAYER                        │
                    │  ┌─────────────────────────────────────────────────────────────┐ │
                    │  │               CoverageAnalyzer (Insight 10)                  │ │
                    │  │  Domain x Judgment x Difficulty x Distractor = Coverage     │ │
                    │  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │ │
                    │  │  │ Gap Finder  │→ │ Target Cells │→ │ Generation Queue │   │ │
                    │  │  └─────────────┘  └──────────────┘  └───────────────────┘   │ │
                    │  └─────────────────────────────────────────────────────────────┘ │
                    │                              ↓                                   │
                    │  ┌─────────────────────────────────────────────────────────────┐ │
                    │  │            Extended Data Generator                          │ │
                    │  │  ┌────────────────┐  ┌───────────────────────────────────┐  │ │
                    │  │  │ Scenario Gen   │→ │ DistractorGenerator (Insight 2)   │  │ │
                    │  │  │ (existing)     │  │ ┌─────────────────────────────┐   │  │ │
                    │  │  └────────────────┘  │ │ - misaligned_phase (P2↔P3) │   │  │ │
                    │  │         ↓            │ │ - inverted_judgment        │   │  │ │
                    │  │  ┌────────────────┐  │ │ - reasoning_error          │   │  │ │
                    │  │  │ Judgment Gen   │  │ │ - evidence_hallucination   │   │  │ │
                    │  │  │ (existing)     │  │ └─────────────────────────────┘   │  │ │
                    │  │  └────────────────┘  └───────────────────────────────────┘  │ │
                    │  │         ↓                         ↓                         │ │
                    │  │  ┌─────────────────────────────────────────────────────────┐│ │
                    │  │  │     DomainVerificationChain (Insight 7)                 ││ │
                    │  │  │  ┌──────────────┬──────────────┬──────────────────────┐ ││ │
                    │  │  │  │ Mathematical │ Empirical    │ Philosophic          │ ││ │
                    │  │  │  │ proof_steps  │ data_points  │ performative_check   │ ││ │
                    │  │  │  │ axiom_refs   │ methodology  │ self_consistency     │ ││ │
                    │  │  │  │ qed_marker   │ replication  │ counterposition_test │ ││ │
                    │  │  │  └──────────────┴──────────────┴──────────────────────┘ ││ │
                    │  │  └─────────────────────────────────────────────────────────┘│ │
                    │  └─────────────────────────────────────────────────────────────┘ │
                    │                              ↓                                   │
                    │  ┌─────────────────────────────────────────────────────────────┐ │
                    │  │            DualPacketFormatter (Insight 9)                   │ │
                    │  │  ┌─────────────────────────┬────────────────────────────┐   │ │
                    │  │  │     Student Packet      │      Evaluator Packet      │   │ │
                    │  │  │ ───────────────────────│ ──────────────────────────  │   │ │
                    │  │  │ - proposition          │ - expected_judgment         │   │ │
                    │  │  │ - evidence             │ - domain                    │   │ │
                    │  │  │ - conditions           │ - difficulty (Insight 4)    │   │ │
                    │  │  │                        │ - evidence_citations        │   │ │
                    │  │  │                        │ - distractor_type           │   │ │
                    │  │  │                        │ - verification_chain        │   │ │
                    │  │  └─────────────────────────┴────────────────────────────┘   │ │
                    │  └─────────────────────────────────────────────────────────────┘ │
                    └─────────────────────────────────────────────────────────────────┘
                                                   ↓
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                     TRAINING LAYER                              │
                    │  ┌─────────────────────────────────────────────────────────────┐ │
                    │  │            EnhancedRewardFunction                           │ │
                    │  │  ┌──────────────────────────────────────────────────────┐   │ │
                    │  │  │ CORRECTNESS (0.6 / -0.5) - PRIMARY                   │   │ │
                    │  │  │ ────────────────────────────────────────────────────  │   │ │
                    │  │  │ model_judgment == expected_judgment (from evaluator) │   │ │
                    │  │  └──────────────────────────────────────────────────────┘   │ │
                    │  │         ↓ (gates following bonuses)                         │ │
                    │  │  ┌──────────────────────────────────────────────────────┐   │ │
                    │  │  │ EVIDENCE GROUNDING (0.15) - Insight 3                │   │ │
                    │  │  │ ────────────────────────────────────────────────────  │   │ │
                    │  │  │ - Citation present?                                  │   │ │
                    │  │  │ - Citation matches input text?                       │   │ │
                    │  │  │ - Citation relevant to conditions?                   │   │ │
                    │  │  └──────────────────────────────────────────────────────┘   │ │
                    │  │  ┌──────────────────────────────────────────────────────┐   │ │
                    │  │  │ DOMAIN VERIFICATION (0.1) - Insight 7                │   │ │
                    │  │  │ ────────────────────────────────────────────────────  │   │ │
                    │  │  │ - Has domain-appropriate verification structure?     │   │ │
                    │  │  │ - Matches verification chain template?               │   │ │
                    │  │  └──────────────────────────────────────────────────────┘   │ │
                    │  │  ┌──────────────────────────────────────────────────────┐   │ │
                    │  │  │ STRUCTURE (0.15) - existing                          │   │ │
                    │  │  └──────────────────────────────────────────────────────┘   │ │
                    │  └─────────────────────────────────────────────────────────────┘ │
                    │                              ↓                                   │
                    │  ┌─────────────────────────────────────────────────────────────┐ │
                    │  │            CurriculumScheduler (Insight 4)                  │ │
                    │  │  epoch 0-20:  difficulty <= 2                               │ │
                    │  │  epoch 20-50: difficulty <= 4                               │ │
                    │  │  epoch 50+:   all difficulties                              │ │
                    │  └─────────────────────────────────────────────────────────────┘ │
                    └─────────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### 1. Extended Data Schema (`src/schema/extended_schema.py`)

Extends `src/data_schema.py` with new fields for all insights.

```python
@dataclass
class ExtendedJudgmentSample:
    """Extended judgment sample with all insight requirements."""

    # Core fields (existing)
    proposition: str
    evidence: str
    conditions: List[str]
    conditions_fulfilled: Dict[str, bool]
    judgment: Literal["Yes", "No", "Insufficient"]
    judgment_reasoning: str

    # Insight 4: Graduated Difficulty
    difficulty: Literal[1, 2, 3, 4, 5]
    difficulty_rationale: str  # Why this difficulty level

    # Insight 3: Evidence Grounding
    evidence_citations: List[EvidenceCitation]  # Specific text spans

    # Insight 7: Domain Verification
    domain: Literal["mathematical", "empirical", "common_sense", "pop_science", "philosophic"]
    verification_chain: DomainVerificationChain

    # Insight 2: Distractor (for negative examples)
    distractor_type: Optional[DistractorType] = None
    distractor_error: Optional[str] = None  # What makes this wrong

    # Insight 9: Packet separation
    # (Implicit: student fields vs evaluator fields)


@dataclass
class EvidenceCitation:
    """A specific citation from input evidence."""
    text: str           # Quoted text
    start_idx: int      # Character offset in evidence
    end_idx: int
    supports_condition: str  # Which condition this supports


class DistractorType(Enum):
    """Types of distractor completions for contrastive learning."""
    MISALIGNED_PHASE = "misaligned_phase"        # P2 output for P3 task (or vice versa)
    INVERTED_JUDGMENT = "inverted_judgment"      # Wrong Yes/No/Insufficient
    REASONING_ERROR = "reasoning_error"          # Correct judgment, flawed reasoning
    EVIDENCE_HALLUCINATION = "evidence_hallucination"  # Cites non-existent evidence
    DOMAIN_MISMATCH = "domain_mismatch"          # Wrong verification structure for domain


@dataclass
class DomainVerificationChain:
    """Domain-specific verification structure."""
    domain: str
    verification_type: str
    required_elements: List[str]
    fulfilled_elements: List[str]

    # Domain-specific fields
    proof_steps: Optional[List[str]] = None          # Mathematical
    data_points: Optional[List[str]] = None          # Empirical
    practical_outcomes: Optional[List[str]] = None   # Common sense
    unfulfilled_claims: Optional[List[str]] = None   # Pop science detection
    performative_check: Optional[str] = None         # Philosophic


@dataclass
class DualPacket:
    """Separates what model sees from what reward function sees."""

    # Student packet: what the model receives during generation
    student: StudentPacket

    # Evaluator packet: metadata for reward function
    evaluator: EvaluatorPacket


@dataclass
class StudentPacket:
    """Data the model sees during generation."""
    system_prompt: str
    user_content: str  # proposition + evidence + conditions


@dataclass
class EvaluatorPacket:
    """Data for reward function evaluation."""
    expected_judgment: Literal["Yes", "No", "Insufficient"]
    domain: str
    difficulty: int
    evidence_citations: List[EvidenceCitation]
    verification_chain: DomainVerificationChain
    distractor_type: Optional[DistractorType] = None
    is_positive_example: bool = True
```

**Dependencies**: None (foundation module)

---

### 2. Coverage Analyzer (`src/coverage/analyzer.py`)

Implements Insight 10: Coverage Enforcement Matrix.

```python
class CoverageAnalyzer:
    """
    Tracks and analyzes coverage across multiple dimensions.

    Dimensions:
    - Domain: 5 levels (mathematical, empirical, common_sense, pop_science, philosophic)
    - Judgment: 3 levels (Yes, No, Insufficient)
    - Difficulty: 5 levels (1-5)
    - Distractor: 2 levels (positive, distractor)

    Total cells: 5 x 3 x 5 x 2 = 150
    """

    def __init__(self, dimensions: List[CoverageDimension]):
        self.dimensions = dimensions
        self.matrix = {}

    def analyze(self, dataset_path: Path) -> CoverageReport:
        """Analyze current dataset coverage."""
        pass

    def identify_gaps(self, min_per_cell: int = 5) -> List[CoverageGap]:
        """Find cells with insufficient examples."""
        pass

    def generate_targets(self, gaps: List[CoverageGap]) -> GenerationQueue:
        """Create prioritized generation targets from gaps."""
        pass

    def visualize(self) -> str:
        """ASCII visualization of coverage matrix."""
        pass


@dataclass
class CoverageGap:
    """A cell in the coverage matrix that needs more examples."""
    domain: str
    judgment: str
    difficulty: int
    has_distractor: bool
    current_count: int
    target_count: int
    priority: float  # Based on how critical this combination is
```

**Dependencies**: Extended Schema

---

### 3. Distractor Generator (`src/generation/distractor_generator.py`)

Implements Insight 2: Contrastive Distractor Learning.

```python
class DistractorGenerator:
    """
    Generates typed distractor completions for contrastive learning.

    Key principle: Learn P2/P3 boundary by seeing what P3 is NOT.
    """

    def __init__(self, oracle_model: str = "gemini/gemini-2.0-flash"):
        self.oracle = dspy.LM(oracle_model)

    def generate_misaligned_phase(
        self,
        sample: ExtendedJudgmentSample
    ) -> DistractorSample:
        """
        Generate P2 output for P3 task.

        P2 characteristics:
        - Generates hypotheses, not judgments
        - Offers multiple possibilities
        - No commitment to yes/no/insufficient
        - Asks questions instead of answering
        """
        pass

    def generate_inverted_judgment(
        self,
        sample: ExtendedJudgmentSample
    ) -> DistractorSample:
        """
        Generate opposite judgment with plausible-sounding reasoning.

        This teaches the model that reasoning must actually support judgment.
        """
        pass

    def generate_reasoning_error(
        self,
        sample: ExtendedJudgmentSample
    ) -> DistractorSample:
        """
        Generate correct judgment with flawed reasoning.

        Types of reasoning errors:
        - Condition not actually checked
        - Evidence misrepresented
        - Logical non-sequitur
        """
        pass

    def generate_evidence_hallucination(
        self,
        sample: ExtendedJudgmentSample
    ) -> DistractorSample:
        """
        Generate response that cites non-existent evidence.

        Key for Insight 3 (evidence grounding).
        """
        pass


@dataclass
class DistractorSample:
    """A distractor completion with typed error."""
    original_sample: ExtendedJudgmentSample
    distractor_completion: str
    distractor_type: DistractorType
    error_explanation: str  # For human review / debugging
```

**Dependencies**: Extended Schema, Oracle (existing)

---

### 4. Domain Verification Chains (`src/generation/verification_chains.py`)

Implements Insight 7: Domain-Specific Verification Chains.

```python
# Verification chain templates by domain

VERIFICATION_TEMPLATES = {
    "mathematical": {
        "verification_type": "proof",
        "required_elements": [
            "axioms_referenced",
            "definitions_used",
            "logical_steps",
            "conclusion_marker"
        ],
        "template": """
        AXIOMS/DEFINITIONS: {axioms}

        PROOF:
        Step 1: {step_1}
        Step 2: {step_2}
        ...

        CONCLUSION: {conclusion}
        QED
        """,
        "output_markers": ["QED", "thus", "therefore", "it follows that"],
    },

    "empirical": {
        "verification_type": "experimental",
        "required_elements": [
            "methodology_stated",
            "data_points_cited",
            "replication_status",
            "measurement_precision"
        ],
        "template": """
        METHODOLOGY: {methodology}

        DATA:
        - Measurement 1: {data_1}
        - Measurement 2: {data_2}

        REPLICATION: {replication_status}

        CONCLUSION: {conclusion} ({confidence_interval})
        """,
        "output_markers": ["data shows", "measured", "observed", "replicated"],
    },

    "common_sense": {
        "verification_type": "pragmatic",
        "required_elements": [
            "situation_described",
            "practical_outcome",
            "experience_cited"
        ],
        "template": """
        SITUATION: {situation}

        PRACTICAL CHECK: {practical_outcome}

        EXPERIENCE: {experience_basis}

        ASSESSMENT: {works_or_not}
        """,
        "output_markers": ["works", "practical", "experience shows", "in this situation"],
    },

    "pop_science": {
        "verification_type": "counterposition_detection",
        "required_elements": [
            "claimed_proof_identified",
            "actual_evidence_examined",
            "conditions_unfulfilled_listed",
            "verdict_on_claim"
        ],
        "template": """
        CLAIM: {claimed_proof}

        ACTUAL EVIDENCE: {actual_evidence}

        CONDITIONS FOR CLAIM TO BE TRUE:
        1. {condition_1}: {fulfilled_1}
        2. {condition_2}: {fulfilled_2}
        ...

        UNFULFILLED CONDITIONS: {unfulfilled_list}

        VERDICT ON CLAIM: {verdict}
        """,
        "output_markers": ["claim unfounded", "correlation not causation", "conditions unfulfilled"],
    },

    "philosophic": {
        "verification_type": "performative_consistency",
        "required_elements": [
            "position_stated",
            "conditions_of_affirmation",
            "self_consistency_check",
            "counterposition_test"
        ],
        "template": """
        POSITION: {position}

        CONDITIONS FOR COHERENT AFFIRMATION:
        - To affirm this intelligently requires: {intelligent_affirmation}
        - To affirm this reasonably requires: {reasonable_affirmation}

        SELF-CONSISTENCY CHECK:
        - Does affirming this presuppose what it denies? {presupposition_check}
        - Can this be held coherently by one who grasps it? {coherence_check}

        VERDICT: {position_or_counterposition}
        """,
        "output_markers": ["self-defeating", "performative contradiction", "coherent", "presupposes"],
    },
}


class VerificationChainGenerator:
    """Generates domain-appropriate verification structures."""

    def generate_chain(
        self,
        domain: str,
        sample: ExtendedJudgmentSample
    ) -> DomainVerificationChain:
        """Generate verification chain for domain."""
        pass

    def validate_chain(
        self,
        chain: DomainVerificationChain,
        completion: str
    ) -> ValidationResult:
        """Check if completion follows verification chain structure."""
        pass
```

**Dependencies**: Extended Schema

---

### 5. Enhanced Reward Function (`src/training/enhanced_reward.py`)

Updates `scripts/run_trl_grpo.py` reward function with Insights 3 and 7.

```python
def enhanced_judgment_reward_fn(
    completions: list[str],
    expected_judgment: list[str],
    domain: list[str],
    evidence: list[str],           # NEW: for citation checking
    evidence_citations: list[str], # NEW: expected citations (JSON)
    verification_chain: list[str], # NEW: expected structure (JSON)
    difficulty: list[int],         # NEW: for curriculum weighting
    **kwargs
) -> list[float]:
    """
    Enhanced reward function incorporating all insights.

    Scoring (max 1.0):
        - Correctness: 0.5 (correct) or -0.5 (wrong) -- PRIMARY
        - Evidence Grounding: 0.2 (Insight 3)
        - Domain Verification: 0.15 (Insight 7)
        - Structure: 0.1
        - Reasoning: 0.05

    Multiplicative: Bonuses only apply if judgment correct.
    """
    rewards = []

    for i, completion in enumerate(completions):
        # === CORRECTNESS (primary signal) ===
        model_judgment = extract_judgment(completion)
        is_correct = (model_judgment == expected_judgment[i])

        correctness_score = 0.5 if is_correct else -0.5
        correctness_multiplier = 1.0 if is_correct else 0.0

        # === EVIDENCE GROUNDING (Insight 3) ===
        evidence_score = compute_evidence_grounding_score(
            completion=completion,
            input_evidence=evidence[i],
            expected_citations=json.loads(evidence_citations[i]) if evidence_citations else [],
        )

        # === DOMAIN VERIFICATION (Insight 7) ===
        domain_score = compute_domain_verification_score(
            completion=completion,
            domain=domain[i],
            verification_chain=json.loads(verification_chain[i]) if verification_chain else None,
        )

        # === EXISTING: Structure and Reasoning ===
        structure_score = 0.1 if model_judgment is not None else 0.0
        reasoning_score = compute_reasoning_score(completion)

        # === FINAL SCORE ===
        structural_bonus = (
            evidence_score * 0.2 +
            domain_score * 0.15 +
            structure_score +
            reasoning_score * 0.05
        ) * correctness_multiplier

        reward = correctness_score + structural_bonus
        rewards.append(reward)

    return rewards


def compute_evidence_grounding_score(
    completion: str,
    input_evidence: str,
    expected_citations: List[dict],
) -> float:
    """
    Score evidence grounding (Insight 3).

    Checks:
    1. Does completion contain quotation markers?
    2. Do quoted segments exist in input evidence?
    3. Are quotes relevant to conditions being checked?
    """
    score = 0.0

    # Extract quoted segments from completion
    quotes = extract_quotes(completion)

    if not quotes:
        return 0.0  # No citations = no bonus

    # Check if quotes exist in input
    valid_quotes = [q for q in quotes if q in input_evidence]
    if valid_quotes:
        score += 0.5  # Quotes are real

    # Check if quotes are relevant (fuzzy match to expected citations)
    if expected_citations:
        relevance = compute_citation_relevance(valid_quotes, expected_citations)
        score += 0.5 * relevance
    else:
        score += 0.25  # Partial credit for valid quotes without expected citations

    return min(1.0, score)


def compute_domain_verification_score(
    completion: str,
    domain: str,
    verification_chain: Optional[dict],
) -> float:
    """
    Score domain verification structure (Insight 7).

    Checks if completion follows domain-appropriate verification pattern.
    """
    template = VERIFICATION_TEMPLATES.get(domain)
    if not template:
        return 0.0

    score = 0.0

    # Check for domain-specific output markers
    markers = template.get("output_markers", [])
    completion_lower = completion.lower()
    marker_hits = sum(1 for m in markers if m.lower() in completion_lower)
    score += 0.5 * (marker_hits / max(len(markers), 1))

    # Check for required structural elements
    required = template.get("required_elements", [])
    # ... structural analysis ...

    return min(1.0, score)
```

**Dependencies**: Verification Chains, Extended Schema

---

### 6. Curriculum Scheduler (`src/training/curriculum.py`)

Implements Insight 4: Graduated Difficulty Scaffolding.

```python
class CurriculumScheduler:
    """
    Schedules training examples by difficulty.

    Difficulty levels:
    1. Introductory: Clear evidence, obvious fulfillment
    2. Standard: Normal evidence, straightforward checking
    3. Advanced: Nuanced evidence, requires careful analysis
    4. Expert: Edge cases, subtle distinctions
    5. Adversarial: Intentionally tricky, tests boundaries
    """

    def __init__(self, schedule: Dict[int, int]):
        """
        Args:
            schedule: Dict mapping epoch -> max_difficulty
                Example: {0: 2, 20: 4, 50: 5}
        """
        self.schedule = schedule

    def filter_dataset(
        self,
        dataset: Dataset,
        epoch: int
    ) -> Dataset:
        """Filter dataset to include only appropriate difficulties."""
        max_difficulty = self.get_max_difficulty(epoch)
        return dataset.filter(lambda x: x['difficulty'] <= max_difficulty)

    def get_max_difficulty(self, epoch: int) -> int:
        """Get maximum difficulty for given epoch."""
        applicable = [e for e in self.schedule.keys() if e <= epoch]
        if not applicable:
            return 1
        return self.schedule[max(applicable)]


DIFFICULTY_CRITERIA = {
    1: {
        "name": "Introductory",
        "evidence_clarity": "unambiguous",
        "conditions": "2-3, clearly stated",
        "judgment": "obvious from evidence",
        "distractors": "none",
    },
    2: {
        "name": "Standard",
        "evidence_clarity": "clear",
        "conditions": "3-4, well-defined",
        "judgment": "straightforward checking",
        "distractors": "basic",
    },
    3: {
        "name": "Advanced",
        "evidence_clarity": "requires interpretation",
        "conditions": "4-5, some implicit",
        "judgment": "requires careful analysis",
        "distractors": "typed",
    },
    4: {
        "name": "Expert",
        "evidence_clarity": "nuanced",
        "conditions": "multiple interdependent",
        "judgment": "edge cases, close calls",
        "distractors": "sophisticated",
    },
    5: {
        "name": "Adversarial",
        "evidence_clarity": "deliberately ambiguous",
        "conditions": "complex, potentially misleading",
        "judgment": "tests operational boundaries",
        "distractors": "adversarial",
    },
}
```

**Dependencies**: Extended Schema

---

### 7. Dual Packet Formatter (`src/formatting/dual_packet.py`)

Implements Insight 9: Dual-Packet Architecture.

```python
class DualPacketFormatter:
    """
    Separates student packet (model input) from evaluator packet (reward function).

    Principle: Model should not see expected_judgment during generation,
    but reward function needs it for evaluation.
    """

    def format_for_training(
        self,
        sample: ExtendedJudgmentSample
    ) -> Dict[str, Any]:
        """
        Format sample into dual packet structure.

        Returns dict with:
        - 'prompt': String for model input (student packet)
        - 'expected_judgment': For reward function
        - 'domain': For reward function
        - 'difficulty': For curriculum
        - 'evidence': For citation checking
        - 'evidence_citations': JSON string for reward function
        - 'verification_chain': JSON string for reward function
        - 'distractor_type': If this is a negative example
        """
        student = self.create_student_packet(sample)
        evaluator = self.create_evaluator_packet(sample)

        return {
            "prompt": student.to_string(),
            **evaluator.to_dict(),
        }

    def create_student_packet(
        self,
        sample: ExtendedJudgmentSample
    ) -> StudentPacket:
        """Create what model sees during generation."""
        system_prompt = self.get_system_prompt(sample.domain)
        user_content = self.format_user_content(
            proposition=sample.proposition,
            evidence=sample.evidence,
            conditions=sample.conditions,
        )
        return StudentPacket(system_prompt=system_prompt, user_content=user_content)

    def create_evaluator_packet(
        self,
        sample: ExtendedJudgmentSample
    ) -> EvaluatorPacket:
        """Create metadata for reward function."""
        return EvaluatorPacket(
            expected_judgment=sample.judgment,
            domain=sample.domain,
            difficulty=sample.difficulty,
            evidence_citations=sample.evidence_citations,
            verification_chain=sample.verification_chain,
            distractor_type=sample.distractor_type,
            is_positive_example=(sample.distractor_type is None),
        )
```

**Dependencies**: Extended Schema

---

## Module Dependency Graph

```
                    ┌───────────────────────────┐
                    │   Extended Schema         │
                    │   (src/schema/)           │
                    └───────────┬───────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
┌───────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│ Coverage Analyzer │ │ Verification    │ │ Dual Packet         │
│ (src/coverage/)   │ │ Chains          │ │ Formatter           │
└─────────┬─────────┘ │ (src/generation)│ │ (src/formatting/)   │
          │           └────────┬────────┘ └──────────┬──────────┘
          │                    │                     │
          │           ┌────────┴─────────┐           │
          │           ▼                  │           │
          │  ┌─────────────────┐         │           │
          │  │ Distractor      │         │           │
          │  │ Generator       │         │           │
          │  │ (src/generation)│         │           │
          │  └────────┬────────┘         │           │
          │           │                  │           │
          │           ▼                  ▼           │
          │  ┌────────────────────────────────────┐  │
          │  │      Data Generation Pipeline      │  │
          │  │      (src/dspy_cognitive/)        │  │
          │  │      - Extended oracle pipeline    │  │
          │  │      - Coverage-guided generation  │  │
          │  └────────────────┬───────────────────┘  │
          │                   │                      │
          └───────────────────┼──────────────────────┘
                              │
                              ▼
          ┌────────────────────────────────────────┐
          │      Enhanced Reward Function          │
          │      (src/training/)                   │
          │      - Correctness check               │
          │      - Evidence grounding (Insight 3)  │
          │      - Domain verification (Insight 7) │
          └────────────────┬───────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────────────┐
          │      Curriculum Scheduler              │
          │      (src/training/)                   │
          │      - Difficulty-based filtering      │
          │      - Progressive training            │
          └────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation (Days 1-2)

**Goal**: Establish data schema and coverage analysis (Quick Wins)

| Task | Module | Depends On | Priority |
|------|--------|------------|----------|
| 1.1 | Extended Schema | None | Critical |
| 1.2 | Coverage Analyzer | 1.1 | Critical |
| 1.3 | Analyze current dataset | 1.2 | Critical |
| 1.4 | Difficulty tagging | 1.1 | High |

**Deliverables**:
- `src/schema/extended_schema.py`
- `src/coverage/analyzer.py`
- Coverage report for existing 450 examples
- Difficulty annotations for existing examples

**Tests** (TDD):
- Schema validation tests
- Coverage calculation tests
- Gap identification tests

---

### Phase 2: Generation Enhancement (Days 3-5)

**Goal**: Implement distractor generation and verification chains

| Task | Module | Depends On | Priority |
|------|--------|------------|----------|
| 2.1 | Verification Chain Templates | 1.1 | High |
| 2.2 | Distractor Generator | 1.1, 2.1 | High |
| 2.3 | Dual Packet Formatter | 1.1 | Medium |
| 2.4 | Update oracle pipeline | 2.1, 2.2, 2.3 | High |

**Deliverables**:
- `src/generation/verification_chains.py`
- `src/generation/distractor_generator.py`
- `src/formatting/dual_packet.py`
- Updated `src/dspy_cognitive/data_generation.py`

**Tests** (TDD):
- Distractor type classification tests
- Verification chain structure tests
- P2/P3 misalignment detection tests
- Dual packet separation tests

---

### Phase 3: Training Enhancement (Days 6-8)

**Goal**: Implement enhanced reward function and curriculum

| Task | Module | Depends On | Priority |
|------|--------|------------|----------|
| 3.1 | Evidence grounding scorer | 2.3 | High |
| 3.2 | Domain verification scorer | 2.1 | High |
| 3.3 | Enhanced reward function | 3.1, 3.2 | High |
| 3.4 | Curriculum scheduler | 1.4 | Medium |
| 3.5 | Update training script | 3.3, 3.4 | High |

**Deliverables**:
- `src/training/enhanced_reward.py`
- `src/training/curriculum.py`
- Updated `scripts/run_trl_grpo.py`

**Tests** (TDD):
- Evidence citation extraction tests
- Citation-to-evidence matching tests
- Domain verification scoring tests
- Curriculum filtering tests
- Reward function integration tests

---

### Phase 4: Data Generation (Days 9-12)

**Goal**: Generate enhanced dataset with full coverage

| Task | Module | Depends On | Priority |
|------|--------|------------|----------|
| 4.1 | Generate difficulty-balanced examples | All Phase 2 | High |
| 4.2 | Generate distractors | 2.2 | High |
| 4.3 | Fill coverage gaps | 1.2 | High |
| 4.4 | Validate enhanced dataset | All | High |

**Deliverables**:
- `data/enhanced/judgment_train.jsonl` (800+ examples)
- `data/enhanced/judgment_val.jsonl`
- Coverage report showing >90% cell fill
- Distractor examples for each type

---

### Phase 5: Integration Testing (Days 13-14)

**Goal**: End-to-end validation

| Task | Module | Depends On | Priority |
|------|--------|------------|----------|
| 5.1 | Training run with enhanced data | All | Critical |
| 5.2 | Compare baseline vs enhanced | 5.1 | Critical |
| 5.3 | Ablation studies | 5.1 | Medium |

**Deliverables**:
- Trained adapter with enhanced pipeline
- Comparison metrics (baseline vs enhanced)
- Documentation of improvements

---

## Test Categories

### Category 1: Schema Validation Tests (`tests/schema/`)

```python
# tests/schema/test_extended_schema.py

def test_extended_judgment_sample_validates_difficulty():
    """Difficulty must be 1-5."""
    with pytest.raises(ValidationError):
        ExtendedJudgmentSample(difficulty=6, ...)

def test_evidence_citation_indices_valid():
    """Citation indices must be within evidence bounds."""
    evidence = "The study found correlation."
    citation = EvidenceCitation(text="correlation", start_idx=16, end_idx=27, ...)
    assert evidence[citation.start_idx:citation.end_idx] == citation.text

def test_distractor_type_enum_values():
    """All distractor types defined."""
    assert DistractorType.MISALIGNED_PHASE.value == "misaligned_phase"
    assert len(DistractorType) == 5

def test_dual_packet_separation():
    """Student packet must not contain evaluator fields."""
    sample = create_test_sample()
    dual = DualPacketFormatter().format_for_training(sample)
    assert "expected_judgment" not in dual["prompt"]
    assert "expected_judgment" in dual
```

### Category 2: Distractor Generation Tests (`tests/generation/`)

```python
# tests/generation/test_distractor_generator.py

def test_misaligned_phase_generates_p2_output():
    """Misaligned phase distractor should look like Understanding output."""
    sample = create_judgment_sample()
    distractor = DistractorGenerator().generate_misaligned_phase(sample)

    # P2 characteristics
    assert "hypothesis" in distractor.distractor_completion.lower() or \
           "might be" in distractor.distractor_completion.lower() or \
           "could be" in distractor.distractor_completion.lower()

    # Should NOT have firm judgment
    assert not re.search(r"judgment:\s*(yes|no)", distractor.distractor_completion.lower())

def test_inverted_judgment_has_opposite():
    """Inverted judgment should be opposite of correct."""
    sample = create_judgment_sample(judgment="Yes")
    distractor = DistractorGenerator().generate_inverted_judgment(sample)

    extracted = extract_judgment(distractor.distractor_completion)
    assert extracted in ["No", "Insufficient"]

def test_evidence_hallucination_cites_nonexistent():
    """Hallucination distractor should cite text not in evidence."""
    sample = create_judgment_sample(evidence="The study measured temperature.")
    distractor = DistractorGenerator().generate_evidence_hallucination(sample)

    citations = extract_quotes(distractor.distractor_completion)
    for citation in citations:
        assert citation not in sample.evidence

def test_distractor_preserves_original():
    """Distractor should link to original sample."""
    sample = create_judgment_sample()
    distractor = DistractorGenerator().generate_reasoning_error(sample)

    assert distractor.original_sample == sample
    assert distractor.error_explanation != ""
```

### Category 3: Evidence Grounding Tests (`tests/training/`)

```python
# tests/training/test_evidence_grounding.py

def test_extract_quotes_finds_quoted_text():
    """Should extract text within quotes."""
    completion = 'The evidence states "correlation was found" which supports...'
    quotes = extract_quotes(completion)
    assert "correlation was found" in quotes

def test_citation_matches_evidence():
    """Citations should exist in input evidence."""
    evidence = "The experiment showed 95% correlation between X and Y."
    completion = 'The evidence shows "95% correlation" which fulfills...'

    score = compute_evidence_grounding_score(
        completion=completion,
        input_evidence=evidence,
        expected_citations=[],
    )
    assert score > 0.4  # Valid quote exists

def test_hallucinated_citation_penalized():
    """Citations not in evidence should not score."""
    evidence = "The study measured temperature."
    completion = 'The evidence clearly shows "pressure readings" indicating...'

    score = compute_evidence_grounding_score(
        completion=completion,
        input_evidence=evidence,
        expected_citations=[],
    )
    assert score < 0.1  # Invalid quote

def test_no_citation_no_bonus():
    """Completion without citations gets no grounding bonus."""
    completion = "The conditions are fulfilled. Judgment: Yes."

    score = compute_evidence_grounding_score(
        completion=completion,
        input_evidence="Some evidence text.",
        expected_citations=[],
    )
    assert score == 0.0
```

### Category 4: Coverage Analysis Tests (`tests/coverage/`)

```python
# tests/coverage/test_analyzer.py

def test_coverage_matrix_dimensions():
    """Matrix should have correct dimensions."""
    analyzer = CoverageAnalyzer(dimensions=[
        CoverageDimension("domain", ["mathematical", "empirical"]),
        CoverageDimension("judgment", ["Yes", "No"]),
    ])

    assert analyzer.total_cells == 4  # 2 x 2

def test_gap_identification():
    """Should identify cells below threshold."""
    dataset = create_test_dataset_with_gaps()
    analyzer = CoverageAnalyzer(standard_dimensions())

    gaps = analyzer.identify_gaps(min_per_cell=5)

    assert len(gaps) > 0
    assert all(gap.current_count < 5 for gap in gaps)

def test_coverage_report_format():
    """Coverage report should be readable."""
    analyzer = CoverageAnalyzer(standard_dimensions())
    analyzer.analyze(Path("test_data.jsonl"))

    report = analyzer.visualize()
    assert "Domain" in report
    assert "Yes" in report

def test_targeted_generation_queue():
    """Should prioritize largest gaps."""
    gaps = [
        CoverageGap(domain="mathematical", judgment="Yes", difficulty=1,
                   has_distractor=False, current_count=0, target_count=5, priority=1.0),
        CoverageGap(domain="empirical", judgment="No", difficulty=2,
                   has_distractor=False, current_count=3, target_count=5, priority=0.5),
    ]

    queue = CoverageAnalyzer.generate_targets(gaps)
    assert queue[0].current_count == 0  # Largest gap first
```

### Category 5: Domain Verification Tests (`tests/generation/`)

```python
# tests/generation/test_verification_chains.py

def test_mathematical_chain_has_proof_structure():
    """Mathematical verification should include proof elements."""
    sample = create_mathematical_sample()
    chain = VerificationChainGenerator().generate_chain("mathematical", sample)

    assert chain.verification_type == "proof"
    assert "axioms_referenced" in chain.required_elements
    assert chain.proof_steps is not None

def test_empirical_chain_has_data_structure():
    """Empirical verification should include data elements."""
    sample = create_empirical_sample()
    chain = VerificationChainGenerator().generate_chain("empirical", sample)

    assert chain.verification_type == "experimental"
    assert "data_points_cited" in chain.required_elements
    assert chain.data_points is not None

def test_pop_science_chain_detects_unfulfilled():
    """Pop science verification should identify unfulfilled claims."""
    sample = create_pop_science_sample()
    chain = VerificationChainGenerator().generate_chain("pop_science", sample)

    assert chain.verification_type == "counterposition_detection"
    assert chain.unfulfilled_claims is not None
    assert len(chain.unfulfilled_claims) > 0

def test_philosophic_chain_has_performative_check():
    """Philosophic verification should check self-consistency."""
    sample = create_philosophic_sample()
    chain = VerificationChainGenerator().generate_chain("philosophic", sample)

    assert chain.verification_type == "performative_consistency"
    assert chain.performative_check is not None

def test_verification_validation():
    """Validator should check completion against chain."""
    chain = create_mathematical_chain()
    good_completion = "Given axioms A1, A2. Step 1: ... Step 2: ... QED"
    bad_completion = "It seems like this might be true."

    good_result = VerificationChainGenerator().validate_chain(chain, good_completion)
    bad_result = VerificationChainGenerator().validate_chain(chain, bad_completion)

    assert good_result.score > bad_result.score
```

### Category 6: Reward Function Integration Tests (`tests/training/`)

```python
# tests/training/test_enhanced_reward.py

def test_correct_judgment_positive_reward():
    """Correct judgment should yield positive reward."""
    reward = enhanced_judgment_reward_fn(
        completions=["Judgment: Yes. Because conditions fulfilled."],
        expected_judgment=["Yes"],
        domain=["empirical"],
        evidence=["Data shows X."],
        evidence_citations=["[]"],
        verification_chain=["{}"],
        difficulty=[2],
    )
    assert reward[0] > 0

def test_wrong_judgment_negative_reward():
    """Wrong judgment should yield negative reward."""
    reward = enhanced_judgment_reward_fn(
        completions=["Judgment: No. Because I think so."],
        expected_judgment=["Yes"],
        domain=["empirical"],
        evidence=["Data shows X."],
        evidence_citations=["[]"],
        verification_chain=["{}"],
        difficulty=[2],
    )
    assert reward[0] < 0

def test_correct_with_citation_higher_than_without():
    """Correct judgment with valid citation should score higher."""
    evidence = "The measurement showed 95% accuracy."

    with_citation = enhanced_judgment_reward_fn(
        completions=['The evidence states "95% accuracy". Judgment: Yes.'],
        expected_judgment=["Yes"],
        domain=["empirical"],
        evidence=[evidence],
        evidence_citations=["[]"],
        verification_chain=["{}"],
        difficulty=[2],
    )

    without_citation = enhanced_judgment_reward_fn(
        completions=["The conditions are met. Judgment: Yes."],
        expected_judgment=["Yes"],
        domain=["empirical"],
        evidence=[evidence],
        evidence_citations=["[]"],
        verification_chain=["{}"],
        difficulty=[2],
    )

    assert with_citation[0] > without_citation[0]

def test_correct_with_domain_structure_higher():
    """Correct judgment with domain verification structure should score higher."""
    with_structure = enhanced_judgment_reward_fn(
        completions=["Given axioms, Step 1: ... Therefore ... QED. Judgment: Yes."],
        expected_judgment=["Yes"],
        domain=["mathematical"],
        evidence=["Proof: ..."],
        evidence_citations=["[]"],
        verification_chain=["{}"],
        difficulty=[2],
    )

    without_structure = enhanced_judgment_reward_fn(
        completions=["Looks correct to me. Judgment: Yes."],
        expected_judgment=["Yes"],
        domain=["mathematical"],
        evidence=["Proof: ..."],
        evidence_citations=["[]"],
        verification_chain=["{}"],
        difficulty=[2],
    )

    assert with_structure[0] > without_structure[0]

def test_bonuses_gated_by_correctness():
    """Structural bonuses should not apply when judgment wrong."""
    wrong_with_everything = enhanced_judgment_reward_fn(
        completions=['Evidence "95% accuracy" proves via QED. Judgment: No.'],
        expected_judgment=["Yes"],  # Wrong!
        domain=["mathematical"],
        evidence=["95% accuracy measured."],
        evidence_citations=["[]"],
        verification_chain=["{}"],
        difficulty=[2],
    )

    # Despite having citations and QED, wrong answer = negative
    assert wrong_with_everything[0] < 0
```

---

## Directory Structure

```
cognitiveop_attunement/
├── src/
│   ├── schema/                          # NEW: Extended data schema
│   │   ├── __init__.py
│   │   ├── extended_schema.py           # ExtendedJudgmentSample, etc.
│   │   ├── difficulty.py                # DifficultyLevel, DIFFICULTY_CRITERIA
│   │   └── distractor.py                # DistractorType, DistractorSample
│   │
│   ├── coverage/                        # NEW: Coverage analysis
│   │   ├── __init__.py
│   │   ├── analyzer.py                  # CoverageAnalyzer
│   │   ├── dimensions.py                # CoverageDimension definitions
│   │   └── visualization.py             # ASCII coverage matrices
│   │
│   ├── generation/                      # NEW: Enhanced generation
│   │   ├── __init__.py
│   │   ├── verification_chains.py       # Domain-specific verification templates
│   │   ├── distractor_generator.py      # Typed distractor generation
│   │   └── chain_validator.py           # Validation against chains
│   │
│   ├── formatting/                      # NEW: Packet formatting
│   │   ├── __init__.py
│   │   └── dual_packet.py               # Student/Evaluator packet separation
│   │
│   ├── training/                        # NEW: Training enhancements
│   │   ├── __init__.py
│   │   ├── enhanced_reward.py           # Enhanced reward function
│   │   ├── evidence_grounding.py        # Citation extraction and scoring
│   │   ├── domain_verification.py       # Domain structure scoring
│   │   └── curriculum.py                # CurriculumScheduler
│   │
│   ├── dspy_cognitive/                  # EXISTING: Update oracle pipeline
│   │   ├── data_generation.py           # Updated with new features
│   │   └── ...
│   │
│   ├── data_schema.py                   # EXISTING: Keep for compatibility
│   └── ...
│
├── tests/                               # NEW: Test framework
│   ├── __init__.py
│   ├── conftest.py                      # Pytest fixtures
│   ├── fixtures/                        # Test data fixtures
│   │   ├── samples.py                   # create_test_sample(), etc.
│   │   └── datasets/                    # Small test datasets
│   │
│   ├── schema/                          # Schema tests
│   │   ├── __init__.py
│   │   ├── test_extended_schema.py
│   │   ├── test_difficulty.py
│   │   └── test_distractor.py
│   │
│   ├── coverage/                        # Coverage tests
│   │   ├── __init__.py
│   │   ├── test_analyzer.py
│   │   └── test_visualization.py
│   │
│   ├── generation/                      # Generation tests
│   │   ├── __init__.py
│   │   ├── test_distractor_generator.py
│   │   ├── test_verification_chains.py
│   │   └── test_p2_p3_distinction.py    # Critical: operational boundary
│   │
│   ├── formatting/                      # Formatting tests
│   │   ├── __init__.py
│   │   └── test_dual_packet.py
│   │
│   └── training/                        # Training tests
│       ├── __init__.py
│       ├── test_evidence_grounding.py
│       ├── test_domain_verification.py
│       ├── test_curriculum.py
│       └── test_enhanced_reward.py      # Integration tests
│
├── scripts/
│   ├── run_trl_grpo.py                  # EXISTING: Update with enhanced reward
│   ├── analyze_coverage.py              # NEW: Coverage analysis script
│   ├── generate_enhanced_dataset.py     # NEW: Enhanced generation script
│   └── ...
│
├── data/
│   ├── oracle_generated/                # EXISTING: Current data
│   │   └── ...
│   └── enhanced/                        # NEW: Enhanced dataset
│       ├── judgment_train.jsonl
│       ├── judgment_val.jsonl
│       ├── coverage_report.json
│       └── distractor_examples/
│           ├── misaligned_phase.jsonl
│           ├── inverted_judgment.jsonl
│           └── ...
│
└── docs/
    ├── implementation_plan.md           # THIS DOCUMENT
    ├── insights_critical_reflection.md  # Source insights
    └── ...
```

---

## Success Metrics

### Data Quality Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Coverage (cells filled) | ~60% | >90% |
| Difficulty distribution | Uneven | Balanced across 1-5 |
| Distractor presence | 0% | 30% of training |
| Evidence citations | ~10% | 80%+ |
| Domain verification structure | 0% | 100% |

### Training Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Judgment accuracy (val) | TBD | +10% vs baseline |
| P2/P3 conflation rate | TBD | -50% |
| Evidence hallucination rate | TBD | -70% |
| Domain structure compliance | TBD | >80% |

### Operational Metrics

| Metric | Description |
|--------|-------------|
| Test coverage | >90% on new modules |
| Documentation | All public functions documented |
| Backward compatibility | Existing data still loadable |

---

## Risk Mitigation

### Risk 1: Distractor Quality

**Risk**: Generated distractors may be ambiguous or mislabeled.

**Mitigation**:
- Oracle verification of all distractors
- Human spot-check sample (10%)
- Clear error_explanation field for debugging

### Risk 2: Coverage Matrix Explosion

**Risk**: Full matrix (150 cells) may be too sparse to fill.

**Mitigation**:
- Start with reduced matrix (Domain x Judgment = 15 cells)
- Add dimensions incrementally
- Accept 80% coverage for difficult cells

### Risk 3: Reward Function Complexity

**Risk**: Multi-component reward may be harder to tune.

**Mitigation**:
- Ablation studies: test each component separately
- Start with original reward, add components incrementally
- Keep component weights configurable

### Risk 4: Backward Compatibility

**Risk**: Enhanced schema may break existing pipeline.

**Mitigation**:
- Extended schema inherits from existing
- Existing data loadable with defaults for new fields
- Migration script for existing data

---

## Conclusion

This implementation plan organizes six validated insights into a coherent architecture with clear module boundaries, explicit dependencies, and comprehensive test coverage. The phased approach prioritizes quick wins (coverage analysis, schema extension) before more complex features (distractor generation, curriculum scheduling).

**Key Principles Maintained**:
1. **Correctness-dominant**: Form without correct judgment remains empty form
2. **Operationally grounded**: Each insight operationalizes Lonergan's cognitive theory
3. **TDD-driven**: Tests specified before implementation
4. **Incrementally deployable**: Each phase delivers working improvements

The virtually unconditioned for this plan: If (1) the modules are implemented as specified, and (2) the tests pass, and (3) the data generation produces quality examples, then the training improvements will operationalize the selected insights.
