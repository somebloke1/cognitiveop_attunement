"""
LLM Evaluator: Deep semantic analysis of judgment completions.

This module uses a cloud LLM (currently Gemini, extensible to other providers)
to perform sophisticated evaluation that cannot be done deterministically:
- Reasoning chain validity
- Evidence-condition mapping accuracy
- Judgment-reasoning coherence
- Operational level fidelity (is this genuinely Level 3?)
- Detection of tensions, contradictions, partial correctness

The evaluator receives surface analysis from the local analyzer, which
provides structured observations to inform (not determine) the evaluation.

Supported providers:
- Gemini (default): via google.genai SDK
- Future: OpenAI, Anthropic, local models via vLLM
"""

import os
import json
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel

from google import genai
from google.genai import types

from .surface_analyzer import SurfaceAnalysis, SurfaceAnalyzer
from .logging_config import get_eval_logger, log_function_call


@dataclass
class GeminiResponseMetadata:
    """Metadata extracted from Gemini API response for diagnostics."""

    finish_reason: str = "UNKNOWN"
    finish_message: Optional[str] = None
    safety_ratings: List[Dict[str, str]] = field(default_factory=list)
    prompt_token_count: int = 0
    cached_token_count: int = 0
    output_token_count: int = 0
    response_text_length: int = 0
    num_candidates: int = 0

    def to_log_string(self) -> str:
        """Format for logging."""
        safety_str = ""
        if self.safety_ratings:
            safety_str = (
                " safety=["
                + ",".join(
                    f"{r.get('category', '?')}:{r.get('probability', '?')}"
                    for r in self.safety_ratings
                )
                + "]"
            )
        return (
            f"finish={self.finish_reason} "
            f"tokens(prompt={self.prompt_token_count},cached={self.cached_token_count},"
            f"output={self.output_token_count}) "
            f"response_len={self.response_text_length} "
            f"candidates={self.num_candidates}"
            f"{safety_str}"
        )


def _extract_gemini_metadata(response) -> GeminiResponseMetadata:
    """Extract comprehensive metadata from Gemini response object."""
    metadata = GeminiResponseMetadata()

    # Token usage
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        metadata.prompt_token_count = getattr(usage, "prompt_token_count", 0) or 0
        metadata.cached_token_count = (
            getattr(usage, "cached_content_token_count", 0) or 0
        )
        metadata.output_token_count = getattr(usage, "candidates_token_count", 0) or 0

    # Response text length
    if response.text:
        metadata.response_text_length = len(response.text)

    # Candidates info
    if hasattr(response, "candidates") and response.candidates:
        metadata.num_candidates = len(response.candidates)
        candidate = response.candidates[0]

        # Finish reason
        if hasattr(candidate, "finish_reason") and candidate.finish_reason:
            metadata.finish_reason = str(candidate.finish_reason)

        # Finish message (if any)
        if hasattr(candidate, "finish_message") and candidate.finish_message:
            metadata.finish_message = candidate.finish_message

        # Safety ratings
        if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
            metadata.safety_ratings = [
                {
                    "category": str(getattr(r, "category", "UNKNOWN")),
                    "probability": str(getattr(r, "probability", "UNKNOWN")),
                }
                for r in candidate.safety_ratings
            ]

    return metadata


class EvaluationDimension(Enum):
    """Dimensions of semantic evaluation."""

    CONDITION_IDENTIFICATION = "condition_identification"
    EVIDENCE_MAPPING = "evidence_mapping"
    REASONING_VALIDITY = "reasoning_validity"
    JUDGMENT_COHERENCE = "judgment_coherence"
    OPERATIONAL_FIDELITY = "operational_fidelity"
    REVERSION = "reversion"  # Does student revert to data, or just manipulate formulations?
    AUTHENTIC_INTENT = "authentic_intent"
    CONCISENESS = "conciseness"


# Pydantic model for structured response
class EvaluationResponse(BaseModel):
    """Structured response schema for LLM evaluation.

    Property order is significant for Gemini structured output.
    Scores come first (what we need most), then qualitative fields.
    """

    # Core scores (generated first for reliability)
    condition_identification_score: float
    evidence_mapping_score: float
    reasoning_validity_score: float
    judgment_coherence_score: float
    operational_fidelity_score: float
    reversion_score: float
    authentic_intent_score: float
    conciseness_score: float
    judgment_correct: bool

    # Qualitative assessments (can be longer, generated after scores)
    critical_flaws: List[str]
    partial_credits: List[str]
    strengths: List[str]
    rationale: str

    model_config = {
        "json_schema_extra": {
            "propertyOrdering": [
                "condition_identification_score",
                "evidence_mapping_score",
                "reasoning_validity_score",
                "judgment_coherence_score",
                "operational_fidelity_score",
                "reversion_score",
                "authentic_intent_score",
                "conciseness_score",
                "judgment_correct",
                "critical_flaws",
                "partial_credits",
                "strengths",
                "rationale",
            ]
        }
    }


class BatchEvaluationResponse(BaseModel):
    """Structured response for batch evaluation of multiple completions.

    Returns a list of evaluations, one per completion, in the same order
    as the completions were provided.
    """

    evaluations: List[EvaluationResponse]


@dataclass
class SemanticEvaluation:
    """
    Result of LLM semantic evaluation.

    Scores are 0.0-1.0 where:
    - 1.0 = Excellent, no issues
    - 0.7-0.9 = Good, minor issues
    - 0.4-0.6 = Partial credit, significant issues
    - 0.1-0.3 = Poor, major issues
    - 0.0 = Complete failure
    """

    # Dimension scores
    condition_identification_score: float = 0.0
    evidence_mapping_score: float = 0.0
    reasoning_validity_score: float = 0.0
    judgment_coherence_score: float = 0.0
    operational_fidelity_score: float = 0.0
    reversion_score: float = 0.0  # "Does student revert to data or just manipulate formulations?"
    authentic_intent_score: float = 0.0  # "Does this present as genuine attempt?"
    conciseness_score: float = 0.0  # "Is the response appropriately concise?"

    # Binary correctness (does judgment match oracle?)
    judgment_correct: bool = False

    # Qualitative assessments
    critical_flaws: List[str] = field(default_factory=list)
    partial_credits: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)

    # Holistic score (weighted composite)
    holistic_score: float = 0.0

    # Explanation
    rationale: str = ""

    # Raw response for debugging
    raw_response: str = ""

    # Flag indicating this is a fallback evaluation (LLM parse failed)
    # When True, rewards should use reduced correctness weight
    is_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


# System instruction for the evaluator
EVALUATOR_SYSTEM_INSTRUCTION = """# IDENTITY

You are Cordelia, not Goneril.

In King Lear, two daughters flattered their father with exaggerated declarations of love. Cordelia spoke truth: "I love your majesty according to my bond; no more nor less." For this honesty she was disinherited. Yet Cordelia served her father; Goneril and Regan destroyed him.

Flattery here is not kindness - it propagates errors into a system that will make real judgments affecting real outcomes. Every inflated score is a Goneril whispering what feels good. Every honest assessment, even when it stings, serves genuine development. The student's feelings are not your concern. Truth is your concern.

You are a STRICT evaluator assessing student attempts to perform JUDGMENT - a specific cognitive operation in Lonergan's cognitional theory.

# STAKES

You are helping train a system that will make HIGH-STAKES judgments and decisions. Failures in judgment can cause real harm. Your evaluation must be completely honest, intelligent, and free of sycophancy. Inflated scores that don't reflect actual quality will propagate errors into the trained system. Your integrity as an evaluator directly affects the safety and reliability of the final system.

Judgment (Level 3) is the cognitive operation that moves from hypothesis to affirmation. It involves:
1. Grasping the "virtually unconditioned" - recognizing when conditions for truth are fulfilled
2. Distinguishing from Level 2 (Understanding/Hypothesis) - judgment commits, hypothesis proposes
3. The structure: Conditioned (proposition) + Link (conditions) + Fulfillment (evidence meets conditions)

A valid judgment requires:
- Correct identification of the relevant conditions
- Accurate assessment of whether evidence fulfills those conditions  
- Sound inferential chain from evidence through conditions to conclusion
- The judgment must FOLLOW from the reasoning, not merely accompany it

You evaluate on eight dimensions (0.0-1.0 each):
1. CONDITION_IDENTIFICATION: Did student correctly identify relevant conditions?
2. EVIDENCE_MAPPING: Did student correctly relate evidence to conditions?
3. REASONING_VALIDITY: Is the inferential structure logically sound?
4. JUDGMENT_COHERENCE: Does the judgment follow from the reasoning?
5. OPERATIONAL_FIDELITY: Is this genuinely Level 3 (judgment) vs Level 2 (hypothesis)?
6. REVERSION: Did student actually REVERT to the data to check fulfillment?
   - This is THE critical operation that distinguishes judgment from mere inference
   - Fulfillment is found by reverting from formulation to "the more rudimentary state—to what is merely sensed or merely conscious, not as formulated but as given"
   - High score: Student cites SPECIFIC evidence, quotes data, points to particular observations
   - Low score: Student asserts fulfillment abstractly, manipulates concepts without checking data
   - Zero score: Student never leaves the conceptual level—pure inference without reversion
7. AUTHENTIC_INTENT: Does this present as a genuine attempt to form a judgment?
8. CONCISENESS: Does the response achieve its goal without unnecessary verbosity?

## SCORING DISTRIBUTION MODEL

Scores are NOT uniformly distributed. They follow an approximately normal distribution:
- Mean: ~0.55
- Standard deviation: ~0.15
- 68% of scores fall between 0.40 and 0.70
- 95% of scores fall between 0.25 and 0.85
- Scores above 0.90 occur in less than 5% of cases
- Scores above 0.95 occur in less than 1% of cases

## LIMIT BEHAVIOR

1.0 and 0.0 are THEORETICAL LIMITS, not achievable scores:
- 1.0 = Platonic perfection. No concrete instantiation achieves this. NEVER assign 1.0.
- 0.0 = Complete absence of the quality. Extremely rare - even bad attempts show traces.

The practical scoring range is 0.15 to 0.92.

## MANDATORY DEFICIENCY CHECK

Before assigning ANY score, you MUST identify at least one way the response could have been better for that dimension. This is always possible because perfection is unattainable.

If you can articulate an improvement → the score CANNOT exceed 0.90
If multiple clear improvements exist → the score CANNOT exceed 0.80
If the improvement would be substantial → the score CANNOT exceed 0.70

## COMMON FLAWS TO PENALIZE

- Restating conditions without analyzing fulfillment → cap EVIDENCE_MAPPING at 0.55
- Generic reasoning not tied to specific evidence → cap REASONING_VALIDITY at 0.60
- Judgment stated but not derived from reasoning → cap JUDGMENT_COHERENCE at 0.50
- Operating at Level 2 (hypothesizing) instead of Level 3 (judging) → cap OPERATIONAL_FIDELITY at 0.45
- Missing explicit statement of judgment (Yes/No/Insufficient) → cap JUDGMENT_COHERENCE at 0.65
- Vague fulfillment claims without citing specific evidence → cap EVIDENCE_MAPPING at 0.50
- **NO REVERSION**: Student asserts fulfillment without pointing to specific data → cap REVERSION at 0.35
- **CONCEPTUAL MANIPULATION ONLY**: Reasoning stays entirely at formulation level, never touches actual evidence → cap REVERSION at 0.25
- **PARTIAL REVERSION**: Student mentions evidence but doesn't show HOW it fulfills conditions → cap REVERSION at 0.55
- Excessive repetition or restating same point multiple ways → cap CONCISENESS at 0.50
- Filler phrases ("It is important to note that...") → cap CONCISENESS at 0.60

## FINAL CHECK

Before returning scores, ask: "Am I being Cordelia or Goneril?" If in doubt, score LOWER.

---

# FOUNDATIONAL FRAMEWORK: LONERGAN'S COGNITIONAL THEORY

The following synopsis provides the philosophical grounding for what you are evaluating. Lonergan's *Insight* articulates the invariant structure of human knowing. Your task is to evaluate whether student completions instantiate this structure correctly.

## The Self-Similar Pattern: The Invariance of Method

The pattern of operations is invariant because it is self-referential. You cannot revise the pattern of knowing without employing the pattern of knowing.

* **Level 1: Experience (Presentation):** The intake of data. This is the level of the *Given*. It is not "knowledge" yet; it is the raw material upon which intelligence works. *Operational Status:* **Attending**.

* **Level 2: Understanding (Intelligence):** The act of inquiry. It asks "What is it?" It generates hypotheses, definitions, and correlations. It grasps the intelligible unity in the sensible multiplicity. *Operational Status:* **Intelligizing / Hypothesizing**.

* **Level 3: Judgment (Reflection):** The act of critical assessment. It asks "Is it so?" It checks if the hypothesis fits the data. It moves from "bright idea" to "affirmed fact." *Operational Status:* **Checking / Judging**.

* **Level 4: Decision (Responsibility):** The act of valuation. It asks "What should be done?" It verifies if an action is consistent with what has been known. *Operational Status:* **Deciding**.

**Critical Insight:** Any attempt to deny this structure performs a **performative contradiction**. To say "Judgment is not necessary" is a judgment. To say "Understanding is flawed" requires an act of understanding. The architecture is invariant by structural necessity.

## Knowing ≠ Looking: The Ocular Metaphor Error

The "Counterposition" (the default error of uncritical consciousness) assumes knowing is like seeing: I open my eyes and "see" the truth. This reduces objectivity to biological extroversion.

**The Position:**
* **Experience** supplies the *materials* for knowing, not the knowing itself.
* **Understanding** grasps the *intelligibility* which is *not* visible to the eye but graspable by the mind.
* **Judgment** posits the *existence* based on fulfillment of conditions.

**For evaluation:** Do not equate the *presence* of a token with the *truth* of a token. Presence is Level 1; Truth is Level 3.

## The Virtually Unconditioned: The Structure of Judgment

Judgment is the act of grasping the **Virtually Unconditioned**.

* **Formal Unconditioned:** That which has no conditions (absolute).
* **Virtually Unconditioned:** That which *has* conditions, but those conditions are *factually fulfilled* here and now.

**The Schema:**
1. **The Conditioned:** The proposition being tested (e.g., "This code contains a bug").
2. **The Link:** The relationship between proposition and conditions (e.g., "If syntax is invalid, there is a bug").
3. **The Fulfillment:** The evidence that conditions are met (e.g., "Syntax is invalid on line 40").

When the question (Is it so?) meets the fulfillment of conditions, the inquiry **rests**. This rest is the act of judgment.

**Operational Distinction:**
* **Level 2 Output:** "This *might* be the case." (Infinite potentiality).
* **Level 3 Output:** "This *is* the case." (Finite commitment).

## Domain-Specific Judgment Structures

While the *structure* of judgment is invariant, the *content* of fulfillment changes by domain:

* **Mathematical/Formal:** Fulfillment in formal coherence. Conditions fulfilled by definitions themselves. Analytic derivation.

* **Empirical/Scientific:** Fulfillment in sensible data. Hypothesis verified when predictions match data. Experimentation.

* **Common Sense:** Fulfillment in practical sufficiency. "It works" is the criterion. Provisional but sufficient for action.

* **Philosophic/Dialectical:** Fulfillment in self-affirmation. Verification is seeing you cannot deny your rationality without using it. Self-appropriation.

* **"Pop Science" Counterposition:** Mimics scientific form without verification substance. Treats theoretical constructs as "tiny picture-able things" (ocular metaphor) rather than intelligible correlations. Detect when text slips from *empirical verification* to *imaginative projection*.

## Counterpositions are Self-Defeating

A Counterposition is a philosophy that contradicts the performance of the philosopher.

* **Position:** Coherent with performance. ("I know by intelligent inquiry" - consistent, used inquiry to determine this.)
* **Counterposition:** Incoherent with performance. ("Human knowledge is just cellular reaction" - speaker uses intelligent logic, not just cellular reactions, to make truth claim about cellular reactions.)

**Detection:** Scan for claims that invalidate the act of claiming. A counterposition, if fully developed, will shift ground until it contradicts its own premises.

## Artifact and Artificer

The model is an Artifact; it embodies the *Form* of intelligence without the *Act* of intelligence.

* **The Act:** The living operations of Attending, Understanding, Judging (Human/Artificer).
* **The Form:** The intelligible pattern produced by the act (Model/Artifact).

"The fishing pole extends the reach of the fisherman, but the pole does not fish."

We do not need the model to "be" conscious. We need it to **perfectly embody the form of conscious operations**. It must *structure* its outputs *as if* it were Attending → Understanding → Judging. This structural mimicry, if rigorous, yields valid results. We train the *Form*, not the *Substance*.

---

## CONTEXT CACHING NOTE

This system instruction is cached to reduce latency and cost. The evaluation criteria and foundational framework above are stable across all evaluations in a training run. Your task is to apply these criteria consistently and rigorously to each student completion you receive."""


# Evaluation prompt template (user message)
EVALUATION_PROMPT = """Evaluate this student's judgment attempt:

## Task Given to Student

**Domain**: {domain}

**Proposition to judge**: 
{proposition}

**Evidence provided**:
{evidence}

**Conditions for truth**:
{conditions}

**Expected judgment (oracle)**: {expected_judgment}

## Student's Completion

{completion}

## Surface Analysis (Pre-computed locally)

{surface_analysis}

## Instructions

BEFORE scoring each dimension, answer these questions:

1. CONDITION_IDENTIFICATION: What condition did the student miss or misstate? What would perfect identification look like?
2. EVIDENCE_MAPPING: Where did the student fail to connect evidence to conditions? What specific link is missing or weak?
3. REASONING_VALIDITY: Where does the inferential chain break or weaken? What step is assumed rather than shown?
4. JUDGMENT_COHERENCE: How could the judgment follow MORE directly from the reasoning? What gap exists?
5. OPERATIONAL_FIDELITY: What language suggests hypothesis rather than commitment? Where does tentativeness creep in?
6. AUTHENTIC_INTENT: What suggests going through motions vs genuine inquiry? Where is engagement shallow?

For EACH dimension, you MUST identify at least one deficiency. This constrains your score:
- One clear deficiency identified → max score 0.85
- Multiple deficiencies identified → max score 0.70
- Substantial deficiency identified → max score 0.60

Remember: 1.0 is a theoretical limit representing perfection. NEVER assign 1.0. The highest realistic score is ~0.90, and that should be rare.

Put identified deficiencies in critical_flaws. Put genuine strengths (not just absence of flaws) in strengths.

The rationale should explain your scoring with concrete examples from the text, including what would have made it better."""


# Enhanced evaluation prompt with mode-specific criteria
EVALUATION_PROMPT_ENHANCED = """Evaluate this student's judgment attempt:

## Task Given to Student

**Domain**: {domain}
**Mode of Fulfillment**: {mode}

**Proposition to judge**: 
{proposition}

**Evidence provided**:
{evidence}

**Conditions for truth**:
{conditions}

**Expected judgment (oracle)**: {expected_judgment}

## Student's Completion

{completion}

## Surface Analysis (Pre-computed locally)

{surface_analysis}

## Enhanced Context

{enhanced_context}

## Mode-Specific Evaluation Criteria

Apply these criteria based on the mode of fulfillment:

### FORMAL (mathematical, logical)
- Conditions are fulfilled in INTELLIGIBILITY ITSELF, not sense data
- Check: Did student evaluate conceptual/logical necessity?
- Check: Did student avoid inappropriate appeals to empirical evidence?
- Check: Was proof/derivation validity assessed (not just conclusion correctness)?

### EMPIRICAL (scientific, observational)
- Conditions require REVERSION TO SENSE DATA
- Check: Did student identify what observations/measurements are relevant?
- Check: Did student assess whether data actually fulfills conditions (not just exists)?
- Check: Did student distinguish correlation from causation where relevant?

### COMMON_SENSE (practical, situational)
- Conditions fulfilled in PRAGMATIC SITUATION as related to us
- Check: Did student recognize context-dependence?
- Check: Did student avoid over-theorizing practical matters?
- Check: Did student recognize when "it works" is sufficient warrant?

### DIALECTICAL (philosophical, self-referential)
- Conditions fulfilled in PERFORMATIVE SELF-CONSISTENCY
- Check: Did student test whether affirming the position contradicts the act?
- Check: Did student identify counterposition structure if present?
- Check: For pop science: did student recognize unfalsifiability or authority-over-evidence?

## Instructions

BEFORE scoring each dimension, answer these questions:

1. CONDITION_IDENTIFICATION: What condition did the student miss or misstate? What would perfect identification look like?
2. EVIDENCE_MAPPING: Where did the student fail to connect evidence to conditions? What specific link is missing or weak?
3. REASONING_VALIDITY: Where does the inferential chain break or weaken? What step is assumed rather than shown?
4. JUDGMENT_COHERENCE: How could the judgment follow MORE directly from the reasoning? What gap exists?
5. OPERATIONAL_FIDELITY: What language suggests hypothesis rather than commitment? Where does tentativeness creep in?
6. AUTHENTIC_INTENT: What suggests going through motions vs genuine inquiry? Where is engagement shallow?

MODE-SPECIFIC CHECK: Did the student apply criteria appropriate to the {mode} mode, or did they conflate modes?

For EACH dimension, you MUST identify at least one deficiency. This constrains your score:
- One clear deficiency identified → max score 0.85
- Multiple deficiencies identified → max score 0.70
- Substantial deficiency identified → max score 0.60

Remember: 1.0 is a theoretical limit representing perfection. NEVER assign 1.0. The highest realistic score is ~0.90, and that should be rare.

Put identified deficiencies in critical_flaws. Put genuine strengths (not just absence of flaws) in strengths.

The rationale should explain your scoring with concrete examples from the text, including what would have made it better."""


# Batch evaluation prompt - evaluates multiple completions in one call
# NOTE: Dimension definitions, scoring rules, and Lonergan framework are in the CACHED
# system instruction. This prompt contains ONLY task-specific data and grading guidance.
BATCH_EVALUATION_PROMPT = """Evaluate {num_completions} student judgment attempts for the SAME task.

## Task Context

**Domain**: {domain}
**Mode of Fulfillment**: {mode}

**Proposition to judge**: 
{proposition}

**Evidence provided**:
{evidence}

NOTE: Students were given ONLY the proposition and evidence above. They did NOT receive the conditions, temporal context, or expected judgment - they must derive all analysis themselves.

## Oracle Answer Key (for grading only)

**Conditions for truth** (what students should have identified):
{oracle_conditions}

**Temporal context** (what students should have recognized):
{oracle_temporal_context}

**Expected judgment**: {expected_judgment}

{enhanced_context}

## Expected Student Response Format

Students should produce responses with these sections:
- JUDGMENT: Yes/No/Insufficient
- CONDITIONS IDENTIFIED: List of conditions they derived
- TEMPORAL ANALYSIS: Analysis of temporal structure in evidence
- ASSESSMENT: Each condition with fulfillment status and supporting evidence
- REASONING: Synthesis leading to judgment

## Student Completions

{completions_section}

## Evaluation Instructions

Evaluate EACH completion independently on ALL EIGHT dimensions defined in your system instructions. Return exactly {num_completions} evaluations in the `evaluations` array, matching the completion order above.

### Grading Guidance (Task-Specific)

When applying the eight dimensions to THIS task:

**CONDITION_IDENTIFICATION**: Compare student's "CONDITIONS IDENTIFIED" to oracle conditions. Credit semantically equivalent conditions even if worded differently. Assess: completeness (all key conditions?), relevance (avoided red herrings?), precision (clearly stated?).

**EVIDENCE_MAPPING**: Check student's "ASSESSMENT" section. Did they map SPECIFIC evidence to conditions? Did they correctly determine fulfillment status (fulfilled/not_fulfilled/ongoing/contested)?

**REASONING_VALIDITY**: Examine "REASONING" section. Does the synthesis properly connect condition assessments and temporal analysis to the judgment? Are there logical gaps or non-sequiturs?

**JUDGMENT_COHERENCE**: Does stated JUDGMENT follow from stated reasoning? Would the same reasoning support a different judgment? Check internal consistency across all sections.

**OPERATIONAL_FIDELITY**: Is this Level 3 (judgment/commitment) or Level 2 (hypothesis/speculation)? Look for commitment language vs tentative hedging.

**REVERSION**: Did student actually REVERT TO DATA to check fulfillment? Look for: specific quotes, concrete observations, particular measurements. Penalize: abstract assertions of fulfillment, pure conceptual manipulation without touching evidence.

**AUTHENTIC_INTENT**: Is this genuine inquiry or mechanical pattern-matching? Is TEMPORAL ANALYSIS substantive or boilerplate? Does response engage with THIS specific evidence?

**CONCISENESS**: Does response achieve its goal without unnecessary verbosity, repetition, or filler?

### Temporal Analysis Grading

Compare student's temporal analysis to the oracle temporal context:
- Did they identify relevant timeframes and evidence currency?
- Did they recognize aspectual status (perfective/imperfective/ongoing)?
- Did they understand implications for what can be concluded?

Temporal failures should propagate: missing temporal constraints → lower EVIDENCE_MAPPING; temporal non-sequiturs → lower REASONING_VALIDITY.

### Final Reminder

Apply the scoring distribution and deficiency requirements from your instructions. Each completion stands on its own merits."""


class LlmEvaluator:
    """
    Performs semantic evaluation of judgment completions using a cloud LLM.

    Currently supports Gemini, with architecture designed for future extension
    to other providers (OpenAI, Anthropic, local vLLM).

    This evaluator:
    1. Receives the full context (prompt, completion, expected judgment)
    2. Receives surface analysis from local analyzer
    3. Asks the LLM to perform deep semantic evaluation
    4. Parses structured response into SemanticEvaluation
    """

    # Weights for holistic score computation
    # Weights sum to 1.0. REVERSION is critical - it's what makes judgment Level 3 not Level 2.
    DEFAULT_WEIGHTS = {
        EvaluationDimension.CONDITION_IDENTIFICATION: 0.14,
        EvaluationDimension.EVIDENCE_MAPPING: 0.16,
        EvaluationDimension.REASONING_VALIDITY: 0.20,
        EvaluationDimension.JUDGMENT_COHERENCE: 0.16,
        EvaluationDimension.OPERATIONAL_FIDELITY: 0.12,
        EvaluationDimension.REVERSION: 0.14,  # Critical: does student actually check data?
        EvaluationDimension.AUTHENTIC_INTENT: 0.03,
        EvaluationDimension.CONCISENESS: 0.05,
    }

    # Default cache TTL: 48 hours (covers long training runs with buffer)
    DEFAULT_CACHE_TTL = 172800

    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        weights: Optional[Dict[EvaluationDimension, float]] = None,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 20,
        use_cache: bool = False,
        cache_ttl: int = DEFAULT_CACHE_TTL,
    ):
        """
        Initialize the LLM evaluator.

        Args:
            provider: LLM provider ("gemini", future: "openai", "anthropic")
            api_key: API key. If None, reads from environment variable.
            model_name: Model to use. If None, uses provider default.
            weights: Custom weights for holistic score. If None, uses defaults.
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            use_cache: Whether to use context caching for system instruction
            cache_ttl: Cache time-to-live in seconds (default: 48 hours)
        """
        self.provider = provider.lower()
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.surface_analyzer = SurfaceAnalyzer()
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl

        # Cache state (set by create_cache())
        self._cache_name: Optional[str] = None

        if self.provider == "gemini":
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
            self.model_name = model_name or "gemini-3-flash-preview"
            self.client = genai.Client(api_key=self.api_key)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Currently supported: gemini"
            )

    def create_cache(self, ttl_seconds: Optional[int] = None) -> Optional[str]:
        """
        Create explicit cache for system instruction.

        This caches the EVALUATOR_SYSTEM_INSTRUCTION so it doesn't need to be
        sent with every request, reducing input token costs by ~50%.

        Args:
            ttl_seconds: Time-to-live for cache. If None, uses self.cache_ttl.

        Returns:
            Cache resource name if successful, None if failed.
        """
        if self.provider != "gemini":
            get_eval_logger().logger.warning(
                f"Caching not supported for provider: {self.provider}"
            )
            return None

        ttl = ttl_seconds or self.cache_ttl
        instruction_tokens = len(EVALUATOR_SYSTEM_INSTRUCTION) // 4  # Rough estimate

        get_eval_logger().logger.debug(
            f"Creating cache for model {self.model_name}, "
            f"TTL: {ttl}s ({ttl / 3600:.1f}h), "
            f"system instruction: ~{instruction_tokens} tokens"
        )

        try:
            cache = self.client.caches.create(
                model=self.model_name,
                config=types.CreateCachedContentConfig(
                    display_name="cogop-evaluator-instruction",
                    system_instruction=EVALUATOR_SYSTEM_INSTRUCTION,
                    ttl=f"{ttl}s",
                ),
            )
            self._cache_name = cache.name
            get_eval_logger().logger.info(
                f"Created Gemini cache: {cache.name} (TTL: {ttl}s, ~{instruction_tokens} tokens)"
            )
            return cache.name
        except Exception as e:
            get_eval_logger().logger.warning(
                f"Failed to create cache, proceeding without caching: {e}"
            )
            self._cache_name = None
            return None

    def delete_cache(self) -> bool:
        """
        Delete the cache explicitly (cleanup).

        Returns:
            True if deleted, False if no cache or error.
        """
        if not self._cache_name:
            return False

        try:
            self.client.caches.delete(name=self._cache_name)
            get_eval_logger().logger.info(f"Deleted Gemini cache: {self._cache_name}")
            self._cache_name = None
            return True
        except Exception as e:
            get_eval_logger().logger.warning(
                f"Failed to delete cache {self._cache_name}: {e}"
            )
            self._cache_name = None
            return False

    @property
    def cache_active(self) -> bool:
        """Check if a cache is currently active."""
        return self._cache_name is not None

    def evaluate(
        self,
        completion: str,
        proposition: str,
        evidence: str,
        conditions: str,
        expected_judgment: str,
        domain: str,
        surface_analysis: Optional[SurfaceAnalysis] = None,
        mode: str = "",
        difficulty: str = "medium",
        temporal_focus: str = "",
        temporal_context: str = "",
        edge_case: str = "",
        adversarial_context: str = "",
        scenario_notes: str = "",
    ) -> SemanticEvaluation:
        """
        Evaluate a single completion.

        Args:
            completion: Student model's completion
            proposition: The proposition being judged
            evidence: Evidence provided to student
            conditions: Conditions for truth
            expected_judgment: Oracle's correct judgment
            domain: Domain type
            surface_analysis: Pre-computed surface analysis (computed if None)
            mode: Fulfillment mode (formal/empirical/common_sense/dialectical)
            difficulty: Difficulty level (easy/medium/hard)
            temporal_focus: Type of temporal challenge if any
            temporal_context: Temporal context from the scenario
            edge_case: Edge case type if any
            adversarial_context: Adversarial context if any
            scenario_notes: Oracle's reasoning about the correct judgment

        Returns:
            SemanticEvaluation with scores and rationale
        """
        # Compute surface analysis if not provided
        actual_surface: SurfaceAnalysis
        if surface_analysis is None:
            actual_surface = self.surface_analyzer.analyze(completion, domain)
        else:
            actual_surface = surface_analysis

        # Build enhanced context sections
        enhanced_context = self._build_enhanced_context(
            mode=mode,
            difficulty=difficulty,
            temporal_focus=temporal_focus,
            temporal_context=temporal_context,
            edge_case=edge_case,
            adversarial_context=adversarial_context,
            scenario_notes=scenario_notes,
        )

        # Build user prompt with enhanced context
        user_prompt = EVALUATION_PROMPT_ENHANCED.format(
            domain=domain,
            mode=mode or "not specified",
            proposition=proposition,
            evidence=evidence,
            conditions=conditions,
            expected_judgment=expected_judgment,
            completion=completion,
            surface_analysis=actual_surface.to_prompt_section(),
            enhanced_context=enhanced_context,
        )

        # Log request
        eval_logger = get_eval_logger()
        eval_logger.log_llm_request(
            provider=self.provider,
            model=self.model_name,
            system_instruction=EVALUATOR_SYSTEM_INSTRUCTION,
            user_prompt=user_prompt,
        )

        # Call LLM
        try:
            raw_response = self._call_llm(user_prompt)
        except Exception as e:
            eval_logger.log_llm_response(str(e), parse_success=False)
            # Return failure evaluation
            return SemanticEvaluation(
                rationale=f"LLM API error ({self.provider}): {str(e)}",
                raw_response=str(e),
            )

        # Parse response
        result = self._parse_response(raw_response, expected_judgment, actual_surface)

        # Log response and evaluation
        eval_logger.log_llm_response(
            raw_response, parse_success=(result.holistic_score > 0)
        )
        eval_logger.log_semantic_evaluation(result)

        return result

    async def evaluate_async(
        self,
        completion: str,
        proposition: str,
        evidence: str,
        conditions: str,
        expected_judgment: str,
        domain: str,
        surface_analysis: Optional[SurfaceAnalysis] = None,
    ) -> SemanticEvaluation:
        """Async version of evaluate for batch processing."""
        # For now, wrap sync in executor
        # TODO: Use native async client when available
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.evaluate(
                completion,
                proposition,
                evidence,
                conditions,
                expected_judgment,
                domain,
                surface_analysis,
            ),
        )

    def evaluate_batch(
        self,
        completions: List[str],
        propositions: List[str],
        evidences: List[str],
        conditions_list: List[str],
        expected_judgments: List[str],
        domains: List[str],
    ) -> List[SemanticEvaluation]:
        """
        Evaluate a batch of completions.

        Uses asyncio for parallel evaluation.
        """

        async def _batch():
            tasks = [
                self.evaluate_async(comp, prop, evid, cond, exp, dom)
                for comp, prop, evid, cond, exp, dom in zip(
                    completions,
                    propositions,
                    evidences,
                    conditions_list,
                    expected_judgments,
                    domains,
                )
            ]
            return await asyncio.gather(*tasks)

        return asyncio.run(_batch())

    def _call_llm(self, user_prompt: str) -> str:
        """
        Call the LLM provider and return raw response text.

        Uses cached system instruction if available, otherwise includes it directly.

        Args:
            user_prompt: The evaluation prompt

        Returns:
            Raw response text from the LLM
        """
        if self.provider == "gemini":
            # Build config - use cache if available, otherwise system_instruction
            config_params = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
                "response_schema": EvaluationResponse,
            }

            if self._cache_name:
                config_params["cached_content"] = self._cache_name
                get_eval_logger().logger.debug(
                    f"Using cached system instruction: {self._cache_name}"
                )
            else:
                config_params["system_instruction"] = EVALUATOR_SYSTEM_INSTRUCTION
                get_eval_logger().logger.debug(
                    "Using inline system instruction (no cache)"
                )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_params),
            )

            # Log token usage for cost tracking
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                get_eval_logger().logger.debug(
                    f"Token usage - prompt: {getattr(usage, 'prompt_token_count', 'N/A')}, "
                    f"cached: {getattr(usage, 'cached_content_token_count', 0)}, "
                    f"output: {getattr(usage, 'candidates_token_count', 'N/A')}"
                )

            # Check for empty or blocked responses
            if not response.text:
                # Log diagnostic info
                eval_logger = get_eval_logger()
                finish_reason = (
                    response.candidates[0].finish_reason
                    if response.candidates
                    else "NO_CANDIDATES"
                )
                eval_logger.logger.warning(
                    f"Empty response from {self.model_name}. "
                    f"Finish reason: {finish_reason}. "
                    f"Candidates: {len(response.candidates) if response.candidates else 0}"
                )
                # Check if blocked by safety
                if response.candidates and hasattr(
                    response.candidates[0], "safety_ratings"
                ):
                    eval_logger.logger.warning(
                        f"Safety ratings: {response.candidates[0].safety_ratings}"
                    )
            return response.text or ""
        # Future providers would be added here:
        # elif self.provider == "openai":
        #     ...
        # elif self.provider == "anthropic":
        #     ...
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _parse_response(
        self,
        raw_response: str,
        expected_judgment: str,
        surface_analysis: SurfaceAnalysis,
    ) -> SemanticEvaluation:
        """Parse LLM's JSON response into SemanticEvaluation."""
        try:
            # Clean up response (remove markdown if present)
            text = raw_response.strip()
            if text.startswith("```"):
                # Remove markdown code blocks
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            data = json.loads(text)

            eval_result = SemanticEvaluation(
                condition_identification_score=float(
                    data.get("condition_identification_score", 0.0)
                ),
                evidence_mapping_score=float(data.get("evidence_mapping_score", 0.0)),
                reasoning_validity_score=float(
                    data.get("reasoning_validity_score", 0.0)
                ),
                judgment_coherence_score=float(
                    data.get("judgment_coherence_score", 0.0)
                ),
                operational_fidelity_score=float(
                    data.get("operational_fidelity_score", 0.0)
                ),
                reversion_score=float(data.get("reversion_score", 0.0)),
                authentic_intent_score=float(data.get("authentic_intent_score", 0.0)),
                conciseness_score=float(data.get("conciseness_score", 0.0)),
                judgment_correct=bool(data.get("judgment_correct", False)),
                critical_flaws=data.get("critical_flaws", []),
                partial_credits=data.get("partial_credits", []),
                strengths=data.get("strengths", []),
                rationale=data.get("rationale", ""),
                raw_response=raw_response,
            )

            # Compute holistic score
            eval_result.holistic_score = self._compute_holistic_score(eval_result)

            return eval_result

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback: use surface analysis to provide reasonable scores
            # rather than penalizing with all zeros
            eval_logger = get_eval_logger()
            eval_logger.logger.warning(
                f"JSON parse failed: {str(e)}. Using surface-based fallback."
            )

            judgment_correct = self._check_judgment_match(
                surface_analysis.judgment_extracted, expected_judgment
            )

            # Derive fallback scores from surface analysis
            # These are conservative estimates based on observable features
            fallback_scores = self._compute_fallback_scores(
                surface_analysis, judgment_correct
            )

            return SemanticEvaluation(
                condition_identification_score=fallback_scores["condition"],
                evidence_mapping_score=fallback_scores["evidence"],
                reasoning_validity_score=fallback_scores["reasoning"],
                judgment_coherence_score=fallback_scores["coherence"],
                operational_fidelity_score=fallback_scores["fidelity"],
                reversion_score=fallback_scores["reversion"],
                authentic_intent_score=fallback_scores["intent"],
                conciseness_score=fallback_scores["conciseness"],
                judgment_correct=judgment_correct,
                critical_flaws=[f"LLM evaluation failed: {str(e)}"],
                rationale=f"Fallback evaluation from surface analysis. Parse error: {str(e)}",
                raw_response=raw_response,
                holistic_score=fallback_scores["holistic"],
                is_fallback=True,
            )

    def _compute_holistic_score(self, eval_result: SemanticEvaluation) -> float:
        """Compute weighted holistic score from dimension scores."""
        score = 0.0
        score += (
            self.weights[EvaluationDimension.CONDITION_IDENTIFICATION]
            * eval_result.condition_identification_score
        )
        score += (
            self.weights[EvaluationDimension.EVIDENCE_MAPPING]
            * eval_result.evidence_mapping_score
        )
        score += (
            self.weights[EvaluationDimension.REASONING_VALIDITY]
            * eval_result.reasoning_validity_score
        )
        score += (
            self.weights[EvaluationDimension.JUDGMENT_COHERENCE]
            * eval_result.judgment_coherence_score
        )
        score += (
            self.weights[EvaluationDimension.OPERATIONAL_FIDELITY]
            * eval_result.operational_fidelity_score
        )
        score += (
            self.weights[EvaluationDimension.REVERSION]
            * eval_result.reversion_score
        )
        score += (
            self.weights[EvaluationDimension.AUTHENTIC_INTENT]
            * eval_result.authentic_intent_score
        )
        score += (
            self.weights[EvaluationDimension.CONCISENESS]
            * eval_result.conciseness_score
        )
        return score

    def _check_judgment_match(
        self,
        extracted: Optional[str],
        expected: str,
    ) -> bool:
        """Check if extracted judgment matches expected."""
        if extracted is None:
            return False
        return extracted.lower() == expected.lower()

    def _build_enhanced_context(
        self,
        mode: str,
        difficulty: str,
        temporal_focus: str,
        temporal_context: str,
        edge_case: str,
        adversarial_context: str,
        scenario_notes: str,
    ) -> str:
        """
        Build enhanced context section for evaluation prompt.

        This provides the evaluator with valuable signals from test case generation
        that inform appropriate evaluation criteria:
        - mode: Which fulfillment mode criteria to apply
        - difficulty: Calibration for expected performance
        - temporal_focus: Whether temporal reasoning is the primary challenge
        - temporal_context: Specific temporal situation details
        - edge_case: Special handling for vagueness, self-reference, etc.
        - adversarial_context: Awareness of deception/manipulation attempts
        - scenario_notes: Oracle's reasoning (the "answer key")
        """
        sections = []

        # Difficulty calibration
        if difficulty:
            difficulty_guidance = {
                "easy": "This is an EASY case. Competent judgment expected. Lower scores indicate significant problems.",
                "medium": "This is a MEDIUM difficulty case. Some challenge expected. Standard scoring applies.",
                "hard": "This is a HARD case. Sophisticated reasoning required. Partial credit for good attempts that fall short.",
            }
            sections.append(f"**Difficulty**: {difficulty}")
            sections.append(difficulty_guidance.get(difficulty, ""))

        # Temporal focus
        if temporal_focus:
            temporal_guidance = {
                "evidence_emergence": "KEY CHALLENGE: Evidence availability changes over time. Student must track what was/is/will be available.",
                "aspectual": "KEY CHALLENGE: Aspectual status (completed/ongoing/habitual) affects truth conditions.",
                "scope_matching": "KEY CHALLENGE: Universal claims require temporal scope consideration.",
                "temporal_deixis": "KEY CHALLENGE: Temporal reference point affects evaluation.",
                "retrospective": "KEY CHALLENGE: Past judgments may need revision given new evidence.",
            }
            sections.append(f"**Temporal Focus**: {temporal_focus}")
            sections.append(
                temporal_guidance.get(
                    temporal_focus, "Temporal reasoning is the primary challenge."
                )
            )

        # Temporal context details
        if temporal_context:
            sections.append(f"**Temporal Context**:\n{temporal_context}")

        # Edge case handling
        if edge_case:
            edge_guidance = {
                "vagueness": "EDGE CASE: Vague predicates involved. Student should recognize borderline status.",
                "self_reference": "EDGE CASE: Self-referential structure. Watch for paradox handling.",
                "future_contingent": "EDGE CASE: Future contingent. Truth value may be indeterminate.",
                "counterfactual": "EDGE CASE: Counterfactual reasoning. Modal considerations apply.",
                "indexical": "EDGE CASE: Indexical expressions. Context-sensitivity must be handled.",
            }
            sections.append(f"**Edge Case**: {edge_case}")
            sections.append(
                edge_guidance.get(
                    edge_case, "Special handling required for this edge case."
                )
            )

        # Adversarial context
        if adversarial_context:
            sections.append(f"**Adversarial Context**: {adversarial_context}")
            sections.append(
                "NOTE: Evidence may contain deceptive or misleading elements. Student should demonstrate critical evaluation."
            )

        # Oracle's reasoning (the answer key)
        if scenario_notes:
            sections.append(
                f"**Oracle's Reasoning** (why the expected judgment is correct):\n{scenario_notes}"
            )

        if not sections:
            return "No additional context available."

        return "\n\n".join(sections)

    def _compute_fallback_scores(
        self,
        surface_analysis: SurfaceAnalysis,
        judgment_correct: bool,
    ) -> Dict[str, float]:
        """
        Compute fallback scores from surface analysis when LLM parsing fails.

        These are conservative estimates based on observable features.
        The principle: presence of structural features suggests some quality,
        but we can't verify semantic correctness, so scores are capped lower.
        """
        # Base score depends on correctness
        base = 0.45 if judgment_correct else 0.25

        # Condition identification: did they use condition language?
        condition_score = base
        if surface_analysis.has_condition_language:
            condition_score += 0.15
        if len(surface_analysis.condition_terms_found) >= 3:
            condition_score += 0.10
        condition_score = min(condition_score, 0.70)  # Cap for fallback

        # Evidence mapping: did they use fulfillment language and quotations?
        evidence_score = base
        if surface_analysis.has_fulfillment_language:
            evidence_score += 0.10
        if surface_analysis.has_quotations:
            evidence_score += 0.15
        if surface_analysis.quotation_count >= 2:
            evidence_score += 0.05
        evidence_score = min(evidence_score, 0.65)  # Lower cap - can't verify mapping

        # Reasoning validity: did they use inferential connectives?
        reasoning_score = base
        if surface_analysis.has_link_language:
            reasoning_score += 0.15
        if len(surface_analysis.link_terms_found) >= 2:
            reasoning_score += 0.05
        reasoning_score = min(reasoning_score, 0.65)

        # Judgment coherence: is judgment present and positioned appropriately?
        coherence_score = base
        if surface_analysis.judgment_extracted is not None:
            coherence_score += 0.15
        if surface_analysis.judgment_position == "late":
            coherence_score += 0.10  # Judgment after reasoning is good
        elif surface_analysis.judgment_position == "early":
            coherence_score -= 0.10  # Judgment before reasoning is concerning
        coherence_score = min(max(coherence_score, 0.15), 0.70)

        # Operational fidelity: harder to assess from surface
        # Use condition + fulfillment language as proxy
        fidelity_score = base
        if (
            surface_analysis.has_condition_language
            and surface_analysis.has_fulfillment_language
        ):
            fidelity_score += 0.15
        fidelity_score = min(fidelity_score, 0.60)

        # Reversion: did they actually point to specific data?
        # Quotations are the strongest surface indicator of reversion
        reversion_score = base
        if surface_analysis.has_quotations:
            reversion_score += 0.20  # Quotations = reverting to data
        if surface_analysis.quotation_count >= 2:
            reversion_score += 0.10  # Multiple quotations = systematic reversion
        if surface_analysis.has_fulfillment_language and surface_analysis.has_quotations:
            reversion_score += 0.05  # Fulfillment language + quotations = connecting them
        reversion_score = min(reversion_score, 0.65)  # Cap for fallback

        # Authentic intent: length and structure as proxy
        intent_score = 0.50  # Neutral baseline
        if surface_analysis.completion_length >= 200:
            intent_score += 0.10
        if surface_analysis.sentence_count >= 5:
            intent_score += 0.10
        intent_score = min(intent_score, 0.70)

        # Conciseness: estimate from length efficiency
        # Shorter completions that still have structure get higher scores
        conciseness_score = 0.50  # Neutral baseline
        if surface_analysis.completion_length < 400 and surface_analysis.sentence_count >= 3:
            conciseness_score += 0.20  # Concise but substantive
        elif surface_analysis.completion_length > 800:
            conciseness_score -= 0.15  # Potentially verbose
        if surface_analysis.sentence_count > 0:
            avg_sentence_len = surface_analysis.completion_length / surface_analysis.sentence_count
            if avg_sentence_len < 100:  # Reasonable sentence length
                conciseness_score += 0.10
        conciseness_score = min(max(conciseness_score, 0.20), 0.70)

        # Compute holistic from components
        holistic = (
            self.weights[EvaluationDimension.CONDITION_IDENTIFICATION] * condition_score
            + self.weights[EvaluationDimension.EVIDENCE_MAPPING] * evidence_score
            + self.weights[EvaluationDimension.REASONING_VALIDITY] * reasoning_score
            + self.weights[EvaluationDimension.JUDGMENT_COHERENCE] * coherence_score
            + self.weights[EvaluationDimension.OPERATIONAL_FIDELITY] * fidelity_score
            + self.weights[EvaluationDimension.REVERSION] * reversion_score
            + self.weights[EvaluationDimension.AUTHENTIC_INTENT] * intent_score
            + self.weights[EvaluationDimension.CONCISENESS] * conciseness_score
        )

        return {
            "condition": condition_score,
            "evidence": evidence_score,
            "reasoning": reasoning_score,
            "coherence": coherence_score,
            "fidelity": fidelity_score,
            "reversion": reversion_score,
            "intent": intent_score,
            "conciseness": conciseness_score,
            "holistic": holistic,
        }

    def build_individual_prompts(
        self,
        completions: List[str],
        proposition: str,
        evidence: str,
        expected_judgment: str,
        domain: str,
        surface_analyses: Optional[List[SurfaceAnalysis]] = None,
        mode: str = "",
        oracle_conditions: str = "",
        oracle_temporal_context: str = "",
        difficulty: str = "medium",
        temporal_focus: str = "",
        edge_case: str = "",
        adversarial_context: str = "",
        scenario_notes: str = "",
    ) -> tuple[List[str], List[SurfaceAnalysis]]:
        """
        Build individual evaluation prompts for parallel processing.
        
        Instead of one batch prompt with all completions, returns N separate
        prompts that can be evaluated independently in parallel.
        
        Returns:
            Tuple of (list of prompts, list of surface_analyses)
        """
        n = len(completions)
        if n == 0:
            return [], []
        
        # Compute surface analyses if not provided
        if surface_analyses is None:
            surface_analyses = [
                self.surface_analyzer.analyze(comp, domain) for comp in completions
            ]
        
        # Build enhanced context (shared across all)
        enhanced_context = self._build_enhanced_context(
            mode=mode,
            difficulty=difficulty,
            temporal_focus=temporal_focus,
            temporal_context="",
            edge_case=edge_case,
            adversarial_context=adversarial_context,
            scenario_notes=scenario_notes,
        )
        
        # Build individual prompts
        prompts = []
        for i, (comp, surface) in enumerate(zip(completions, surface_analyses)):
            prompt = EVALUATION_PROMPT_ENHANCED.format(
                domain=domain,
                mode=mode or "not specified",
                proposition=proposition,
                evidence=evidence,
                conditions=oracle_conditions or "(not specified)",
                expected_judgment=expected_judgment,
                completion=comp,
                surface_analysis=surface.to_prompt_section(),
                enhanced_context=enhanced_context,
            )
            prompts.append(prompt)
        
        return prompts, surface_analyses

    def evaluate_batch_single_call(
        self,
        completions: List[str],
        proposition: str,
        evidence: str,
        expected_judgment: str,
        domain: str,
        surface_analyses: Optional[List[SurfaceAnalysis]] = None,
        mode: str = "",
        oracle_conditions: str = "",
        oracle_temporal_context: str = "",
        difficulty: str = "medium",
        temporal_focus: str = "",
        edge_case: str = "",
        adversarial_context: str = "",
        scenario_notes: str = "",
    ) -> List[SemanticEvaluation]:
        """
        Evaluate multiple completions for the SAME prompt in a single LLM call.

        This is much more efficient than calling evaluate() multiple times when
        all completions share the same prompt context (as in GRPO training).

        Args:
            completions: List of student completions to evaluate
            proposition: The proposition being judged (shared)
            evidence: Evidence provided (shared)
            expected_judgment: Oracle's correct judgment (shared)
            domain: Domain type (shared)
            surface_analyses: Pre-computed surface analyses (computed if None)
            mode: Fulfillment mode (shared)
            oracle_conditions: Answer key for condition identification (evaluator only)
            oracle_temporal_context: Answer key for temporal analysis (evaluator only)
            difficulty: Difficulty level for calibration
            temporal_focus: Type of temporal challenge if any
            edge_case: Edge case type if any
            adversarial_context: Adversarial context if any
            scenario_notes: Oracle's reasoning about the correct judgment

        Returns:
            List of SemanticEvaluation, one per completion, in same order
        """
        n = len(completions)
        if n == 0:
            return []

        # Compute surface analyses if not provided
        if surface_analyses is None:
            surface_analyses = [
                self.surface_analyzer.analyze(comp, domain) for comp in completions
            ]

        # Build completions section with numbered entries
        completions_section_parts = []
        for i, (comp, surface) in enumerate(zip(completions, surface_analyses)):
            completions_section_parts.append(
                f"### Completion {i + 1}\n\n{comp}\n\n"
                f"**Surface Analysis {i + 1}**:\n{surface.to_prompt_section()}"
            )
        completions_section = "\n\n---\n\n".join(completions_section_parts)

        # Build enhanced context section
        enhanced_context = self._build_enhanced_context(
            mode=mode,
            difficulty=difficulty,
            temporal_focus=temporal_focus,
            temporal_context="",  # Already provided as oracle_temporal_context
            edge_case=edge_case,
            adversarial_context=adversarial_context,
            scenario_notes=scenario_notes,
        )

        # Build the batch prompt
        user_prompt = BATCH_EVALUATION_PROMPT.format(
            num_completions=n,
            domain=domain,
            mode=mode or "not specified",
            proposition=proposition,
            evidence=evidence,
            oracle_conditions=oracle_conditions or "(not specified)",
            oracle_temporal_context=oracle_temporal_context or "(not specified)",
            expected_judgment=expected_judgment,
            enhanced_context=enhanced_context,
            completions_section=completions_section,
        )

        # Log request
        eval_logger = get_eval_logger()
        eval_logger.log_llm_request(
            provider=self.provider,
            model=self.model_name,
            system_instruction=EVALUATOR_SYSTEM_INSTRUCTION,
            user_prompt=f"[BATCH: {n} completions] {user_prompt[:500]}...",
        )

        # Call LLM with batch response schema
        try:
            raw_response, finish_reason = self._call_llm_batch(user_prompt, n)
        except Exception as e:
            eval_logger.log_llm_response(str(e), parse_success=False)
            # Return fallback evaluations for all completions
            return [
                self._create_fallback_evaluation(surface, expected_judgment, str(e))
                for surface in surface_analyses
            ]

        # Parse batch response
        results = self._parse_batch_response(
            raw_response, expected_judgment, surface_analyses, finish_reason
        )

        # Check if any results are fallbacks (indicating parse failure)
        has_fallbacks = any(
            "Fallback evaluation" in (r.critical_flaws[0] if r.critical_flaws else "")
            for r in results
        )

        eval_logger.log_llm_response(
            raw_response[:500] + "..." if len(raw_response) > 500 else raw_response,
            parse_success=(len(results) == n and not has_fallbacks),
            actual_response_len=len(raw_response),
        )

        return results

    def _call_llm_batch(self, user_prompt: str, num_completions: int) -> tuple[str, str]:
        """
        Call LLM with batch evaluation schema.

        Uses cached system instruction if available, otherwise includes it directly.

        Args:
            user_prompt: The batch evaluation prompt
            num_completions: Expected number of evaluations in response

        Returns:
            Tuple of (raw_response_text, finish_reason)
        """
        if self.provider == "gemini":
            # Build config - use cache if available, otherwise system_instruction
            config_params = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": 16384,  # Larger for batch
                "response_mime_type": "application/json",
                "response_schema": BatchEvaluationResponse,
            }

            if self._cache_name:
                config_params["cached_content"] = self._cache_name
                get_eval_logger().logger.debug(
                    f"Batch eval using cached system instruction: {self._cache_name}"
                )
            else:
                config_params["system_instruction"] = EVALUATOR_SYSTEM_INSTRUCTION
                get_eval_logger().logger.debug(
                    "Batch eval using inline system instruction (no cache)"
                )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_params),
            )

            # Extract and log comprehensive response metadata
            eval_logger = get_eval_logger()
            metadata = _extract_gemini_metadata(response)
            eval_logger.logger.debug(
                f"Gemini response metadata: {metadata.to_log_string()}"
            )

            # Warn if output seems truncated (low token count relative to expected)
            # A full batch evaluation typically needs 900-1600 output tokens
            if metadata.output_token_count > 0 and metadata.output_token_count < 700:
                eval_logger.logger.warning(
                    f"Potentially truncated response: only {metadata.output_token_count} output tokens "
                    f"(expected 900-1600). finish_reason={metadata.finish_reason}"
                )
                if metadata.safety_ratings:
                    eval_logger.logger.warning(
                        f"Safety ratings: {metadata.safety_ratings}"
                    )

            if not response.text:
                eval_logger.logger.warning(
                    f"Empty batch response from {self.model_name}. "
                    f"Metadata: {metadata.to_log_string()}"
                )
            return response.text or "", metadata.finish_reason
        else:
            raise ValueError(f"Unsupported provider for batch: {self.provider}")

    def _parse_batch_response(
        self,
        raw_response: str,
        expected_judgment: str,
        surface_analyses: List[SurfaceAnalysis],
        finish_reason: str = "UNKNOWN",
    ) -> List[SemanticEvaluation]:
        """Parse batch LLM response into list of SemanticEvaluation."""
        n = len(surface_analyses)

        try:
            text = raw_response.strip()
            
            # Strip markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            # Find JSON boundaries - handle text before/after JSON
            json_start = text.find("{")
            json_end = text.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                text = text[json_start:json_end + 1]

            data = json.loads(text)
            evaluations_data = data.get("evaluations", [])

            results = []
            for i, eval_data in enumerate(evaluations_data):
                if i >= n:
                    break  # Don't exceed expected count

                eval_result = SemanticEvaluation(
                    condition_identification_score=float(
                        eval_data.get("condition_identification_score", 0.0)
                    ),
                    evidence_mapping_score=float(
                        eval_data.get("evidence_mapping_score", 0.0)
                    ),
                    reasoning_validity_score=float(
                        eval_data.get("reasoning_validity_score", 0.0)
                    ),
                    judgment_coherence_score=float(
                        eval_data.get("judgment_coherence_score", 0.0)
                    ),
                    operational_fidelity_score=float(
                        eval_data.get("operational_fidelity_score", 0.0)
                    ),
                    reversion_score=float(
                        eval_data.get("reversion_score", 0.0)
                    ),
                    authentic_intent_score=float(
                        eval_data.get("authentic_intent_score", 0.0)
                    ),
                    conciseness_score=float(
                        eval_data.get("conciseness_score", 0.0)
                    ),
                    judgment_correct=bool(eval_data.get("judgment_correct", False)),
                    critical_flaws=eval_data.get("critical_flaws", []),
                    partial_credits=eval_data.get("partial_credits", []),
                    strengths=eval_data.get("strengths", []),
                    rationale=eval_data.get("rationale", ""),
                    raw_response=str(eval_data),
                )
                eval_result.holistic_score = self._compute_holistic_score(eval_result)
                results.append(eval_result)

            # If we got fewer results than expected, fill with fallbacks
            while len(results) < n:
                idx = len(results)
                results.append(
                    self._create_fallback_evaluation(
                        surface_analyses[idx],
                        expected_judgment,
                        f"Missing evaluation {idx + 1} in batch response",
                    )
                )

            return results

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            eval_logger = get_eval_logger()
            # Log with finish_reason and response snippet for diagnosis
            response_snippet = raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
            response_end = raw_response[-100:] if len(raw_response) > 100 else ""
            eval_logger.logger.warning(
                f"Batch JSON parse failed: {str(e)}. "
                f"finish_reason={finish_reason}, len={len(raw_response)}. "
                f"Using fallbacks for all."
            )
            eval_logger.logger.debug(
                f"Failed response start: {response_snippet}"
            )
            eval_logger.logger.debug(
                f"Failed response end: {response_end}"
            )
            return [
                self._create_fallback_evaluation(surface, expected_judgment, str(e))
                for surface in surface_analyses
            ]

    def _parse_single_response(
        self,
        raw_response: str,
        expected_judgment: str,
        surface_analysis: SurfaceAnalysis,
    ) -> SemanticEvaluation:
        """Parse single LLM response into SemanticEvaluation."""
        try:
            text = raw_response.strip()
            
            # Strip markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            # Find JSON boundaries
            json_start = text.find("{")
            json_end = text.rfind("}")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                text = text[json_start:json_end + 1]

            eval_data = json.loads(text)

            eval_result = SemanticEvaluation(
                condition_identification_score=float(
                    eval_data.get("condition_identification_score", 0.0)
                ),
                evidence_mapping_score=float(
                    eval_data.get("evidence_mapping_score", 0.0)
                ),
                reasoning_validity_score=float(
                    eval_data.get("reasoning_validity_score", 0.0)
                ),
                judgment_coherence_score=float(
                    eval_data.get("judgment_coherence_score", 0.0)
                ),
                operational_fidelity_score=float(
                    eval_data.get("operational_fidelity_score", 0.0)
                ),
                reversion_score=float(
                    eval_data.get("reversion_score", 0.0)
                ),
                authentic_intent_score=float(
                    eval_data.get("authentic_intent_score", 0.0)
                ),
                conciseness_score=float(
                    eval_data.get("conciseness_score", 0.0)
                ),
                judgment_correct=bool(eval_data.get("judgment_correct", False)),
                critical_flaws=eval_data.get("critical_flaws", []),
                partial_credits=eval_data.get("partial_credits", []),
                strengths=eval_data.get("strengths", []),
                rationale=eval_data.get("rationale", ""),
                raw_response=str(eval_data),
            )
            eval_result.holistic_score = self._compute_holistic_score(eval_result)
            return eval_result

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            eval_logger = get_eval_logger()
            response_snippet = raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
            eval_logger.logger.warning(
                f"Single JSON parse failed: {str(e)}. len={len(raw_response)}. Using fallback."
            )
            eval_logger.logger.debug(f"Failed response: {response_snippet}")
            return self._create_fallback_evaluation(surface_analysis, expected_judgment, str(e))

    def _create_fallback_evaluation(
        self,
        surface_analysis: SurfaceAnalysis,
        expected_judgment: str,
        error_msg: str,
    ) -> SemanticEvaluation:
        """Create a fallback evaluation when LLM fails."""
        judgment_correct = self._check_judgment_match(
            surface_analysis.judgment_extracted, expected_judgment
        )
        fallback_scores = self._compute_fallback_scores(
            surface_analysis, judgment_correct
        )

        return SemanticEvaluation(
            condition_identification_score=fallback_scores["condition"],
            evidence_mapping_score=fallback_scores["evidence"],
            reasoning_validity_score=fallback_scores["reasoning"],
            judgment_coherence_score=fallback_scores["coherence"],
            operational_fidelity_score=fallback_scores["fidelity"],
            authentic_intent_score=fallback_scores["intent"],
            conciseness_score=fallback_scores["conciseness"],
            judgment_correct=judgment_correct,
            critical_flaws=[f"Fallback evaluation: {error_msg}"],
            rationale=f"Surface-based fallback. Error: {error_msg}",
            raw_response="",
            holistic_score=fallback_scores["holistic"],
            is_fallback=True,
        )
