#!/usr/bin/env python3
"""
Generate Training Examples with Configurable Model

This generator produces judgment training examples with integrated temporal awareness.
Every test case has temporal structure (evidence timing, aspectual status, judge's standpoint).
Some cases specifically focus on temporal challenges as the primary difficulty.

Supports:
- Gemini models (gemini/gemini-2.5-pro, gemini/gemini-2.0-flash, etc.)
- Anthropic models (anthropic/claude-sonnet-4.5, anthropic/claude-opus-4, etc.)

All runs are logged to the data registry for full provenance.
Output files are timestamped to prevent overwrites.

Usage:
    # Generate 20 examples with Claude Sonnet 4.5
    python scripts/generate_with_model.py --model anthropic/claude-sonnet-4.5 --count 20

    # Generate 100 examples with Gemini Pro
    python scripts/generate_with_model.py --model gemini/gemini-2.5-pro --count 100

    # Generate with temporal focus on 30% of cases
    python scripts/generate_with_model.py --model anthropic/claude-sonnet-4.5 --count 100 --temporal-focus-ratio 0.3
"""

import argparse
import json
import logging
import os
import sys
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging - set up later after arg parsing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from src.data_registry import DataRegistry, GenerationRun
from src.domains import (
    # Principled schema
    ALL_DOMAINS,
    get_principled_prompt_context,
    get_all_domain_names,
    # Principled schema API
    PRINCIPLED_DOMAINS,
    FulfillmentMode,
    EmpiricalLevel,
    CommonSenseVariety,
    EpistemicContext,
    AdversarialType,
    EdgeCaseType,
    get_principled_domain,
    get_principled_prompt_context,
    get_domains_by_mode,
    get_domains_by_empirical_level,
    get_mode_spec,
    get_edge_case_spec,
    get_adversarial_spec,
    roll_edge_case,
    EDGE_CASE_SPECS,
    ADVERSARIAL_SPECS,
)


# ============ MODEL CONFIGURATION ============

ANTHROPIC_MODELS = {
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
    "anthropic/claude-sonnet-4": "claude-sonnet-4-20250514",
    "anthropic/claude-opus-4": "claude-opus-4-20250514",
    "anthropic/claude-opus-4.5": "claude-opus-4-5-20251101",
    "anthropic/claude-haiku-4.5": "claude-haiku-4-5-20251001",
}

GEMINI_MODELS = {
    "gemini/gemini-2.5-pro": "gemini-2.5-pro",
    "gemini/gemini-2.0-flash": "gemini-2.0-flash",
    "gemini/gemini-2.0-pro": "gemini-2.0-pro",
}

# Temporal focus types for when temporal structure is the PRIMARY challenge
TEMPORAL_FOCUS_TYPES = [
    "evidence_emergence",  # Evidence revealed sequentially; must track condition-status over time
    "aspectual",  # Perfective vs imperfective evidence; ongoing vs completed
    "scope_matching",  # Evidence-relative claim vs reality-claim; avoid over/under-commitment
    "temporal_deixis",  # Same proposition from different temporal standpoints
    "retrospective",  # Historical judgment reassessment; level-of-error identification
]


def get_provider(model: str) -> str:
    """Determine provider from model string."""
    if model.startswith("anthropic/"):
        return "anthropic"
    elif model.startswith("gemini/"):
        return "gemini"
    else:
        raise ValueError(f"Unknown model provider: {model}")


# ============ TRAINING PROMPT VARIATION SYSTEM ============
#
# To prevent overtraining on specific phrasings, we vary the surrounding context
# while keeping the format markers (JUDGMENT: / REASONING:) fixed.
#
# The oracle model generates variations based on these seed concepts:

PROMPT_VARIATION_INSTRUCTION = """Generate a varied system prompt for training a judgment model.

SEED CONCEPTS (vary the phrasing, preserve the operational meaning):

ROLE FRAMING seed: "The act of reflective understanding that grasps whether conditions for a prospective judgment are fulfilled"
- The cognitive operation that moves from hypothesis to commitment
- The transition from "this might be so" to "this is so" (or "this is not so" or "I cannot yet determine")
- Vary: "You are...", "Your function is...", "You perform...", "This is the operation of..."

TASK DESCRIPTION seed: "Determine whether conditions for truth are fulfilled in the available evidence"
- The structure: (1) a proposition to be judged, (2) conditions that link the proposition to what would make it true, (3) evidence that may or may not fulfill those conditions
- Vary how this is expressed while preserving the three-part structure
- Do NOT use academic jargon or attribute to any philosopher/theory

OPERATIONAL ANCHORS: Select TWO from different languages/traditions that capture the spirit of judgment:
- Greek: krisis (judgment/decision), diakrisis (discernment), dokimazein (to test/examine)
- Hebrew: mishpat (judgment/justice), binah (discernment), shakhal (to weigh/consider)  
- Sanskrit: viveka (discrimination/discernment), tarka (reasoning to conclusion), nirnaya (settled judgment)
- German: Urteil (judgment), Beurteilung (assessment), Entscheidung (decision after weighing)
- Latin: iudicium (judgment), discretio (discernment), sententia (considered opinion)
- Chinese: 判断 pànduàn (judgment), 明辨 míngbiàn (clear discernment), 审度 shěndù (careful assessment)

CULTURAL ANCHOR (optional - include in ~40% of prompts):
Make a succinct reference to a folk tale, parable, or proverb that exemplifies the imperative to be critically intelligent - to pause and reflect rather than naively repeat ideas or accept claims without examination. The reference should capture the wisdom of the reflective pause: the moment between hearing and assenting, between appearance and judgment. Draw from global storytelling traditions. Vary the references; do not repeat the same one.

CLOSING EXHORTATION seed: "Affirm only what the evidence warrants; withhold when grounds are insufficient"
- Knowledge vs mere opinion
- The discipline of not claiming more than you can support
- Honesty over convenience

OUTPUT FORMAT (MUST BE PRESERVED EXACTLY - this is for extraction):
JUDGMENT: Yes, No, or Insufficient
REASONING: [explanation]

Generate a complete system prompt that:
1. Varies the role framing (do not use academic terms or attributions)
2. Varies the task description (operational, not theoretical)
3. Includes operational anchors in exactly TWO languages/traditions
4. Optionally includes ONE cultural anchor (~40% of the time)
5. Varies the closing exhortation
6. PRESERVES the format instruction and example EXACTLY as shown

Return ONLY the system prompt text, no commentary."""

# Default/fallback prompt when variation generation fails
DEFAULT_TRAINING_PROMPT = """You perform the operation of judgment (krisis/mishpat). Given a proposition and evidence, determine whether the conditions for truth are fulfilled.

If conditions are fulfilled in the evidence, affirm (Yes). If conditions are identifiably not fulfilled, deny (No). If the evidence is insufficient to determine, withhold (Insufficient).

OUTPUT FORMAT (follow exactly):
JUDGMENT: Yes, No, or Insufficient
REASONING: Your explanation of condition fulfillment

Example:
JUDGMENT: Insufficient
REASONING: The first condition requires X, but the evidence only shows Y. The second condition...

Claim only what the evidence supports. Withhold when grounds are lacking."""

# Fixed format block that MUST appear in all variations
FORMAT_BLOCK = """OUTPUT FORMAT (follow exactly):
JUDGMENT: Yes, No, or Insufficient
REASONING: Your explanation of condition fulfillment

Example:
JUDGMENT: Insufficient
REASONING: The first condition requires X, but the evidence only shows Y. The second condition..."""


def generate_varied_training_prompt(generator) -> str:
    """Generate a varied training system prompt using the oracle model.

    Args:
        generator: The generator instance (AnthropicGenerator or GeminiGenerator)

    Returns:
        A varied system prompt with fixed format block
    """
    try:
        # Use the generator's _call method which handles API differences
        varied_prompt = generator._call(
            system="You generate system prompts for AI training. Return only the prompt text.",
            user=PROMPT_VARIATION_INSTRUCTION,
            max_tokens=1024,
            temperature=0.8,  # Slightly higher for variation
        )
        varied_prompt = varied_prompt.strip()

        # Verify the format block is present
        if "JUDGMENT:" not in varied_prompt or "REASONING:" not in varied_prompt:
            logger.warning("Varied prompt missing format markers, using default")
            return DEFAULT_TRAINING_PROMPT

        return varied_prompt

    except Exception as e:
        logger.warning(f"Prompt variation failed: {e}, using default")
        return DEFAULT_TRAINING_PROMPT


# ============ UNIFIED SYSTEM PROMPTS ============

# This prompt pervades ALL scenario generation - temporal awareness is not optional
SCENARIO_SYSTEM_PROMPT = """You are generating training scenarios for a cognitive AI system learning Lonergan's theory of judgment.

JUDGMENT is the cognitive operation that moves from hypothesis to affirmation by grasping the "virtually unconditioned":
- CONDITIONED: The proposition to be judged
- LINK: The connection between the proposition and its conditions
- FULFILLMENT: Evidence that meets the conditions

THE THREE JUDGMENT OUTCOMES - CRITICAL DISTINCTIONS:

"Yes" = Conditions ARE fulfilled. Evidence POSITIVELY ESTABLISHES that conditions are met. Affirm.

"No" = Evidence OBLIGES NEGATION. We have sufficient evidence to determine that at least one condition is NOT fulfilled. Deny.
  This includes:
  - Direct contradiction: Evidence explicitly shows the opposite
  - Failed verification: A proper test was conducted and yielded negative results
  - Logical incompatibility: The evidence is inconsistent with the proposition
  
  NOTE: "No" is a POSITIVE judgment - we are affirming that conditions are unfulfilled, based on evidence.

"Insufficient" = THE DATA REQUIRED TO JUDGE DOES NOT YET EXIST OR IS INACCESSIBLE.
  The question cannot be answered because the relevant information has not been obtained.
  
  GENUINE INSUFFICIENCY requires ONE of these situations:
  - Evidence not yet collected: "The study is ongoing, results expected next month"
  - Data pending: "Samples have been taken but not yet analyzed"
  - Future contingent: "The election hasn't happened yet"
  - Inaccessible information: "The records were sealed/destroyed/never created"
  - Untested claim: "No one has yet investigated this specific question"
  
  THE KEY TEST: Would ATTEMPTING to judge require inventing information that doesn't exist?
  - If yes → Insufficient
  - If the evidence exists but doesn't support the claim → No
  - If the evidence exists and supports the claim → Yes

COMMON ERRORS TO AVOID:

❌ WRONG: Generating "Insufficient" when evidence EXISTS but is NEGATIVE
   "The clinical trial found no significant effect" → This is "No" (negative result exists)
   
❌ WRONG: Treating absence of positive evidence as insufficiency
   "There is no evidence supporting X" → If a proper search was done, this is "No"
   
❌ WRONG: Confusing uncertainty about causes with insufficiency
   "We don't know why, but the effect wasn't observed" → This is "No" (effect not observed)

✓ RIGHT: "Insufficient" when the investigation hasn't been done
   "No study has yet examined whether X affects Y" → Genuinely Insufficient
   
✓ RIGHT: "Insufficient" when results are literally pending
   "The experiment concluded yesterday; results are being analyzed" → Genuinely Insufficient

TEMPORAL STRUCTURE pervades all judgment:
1. Every judgment is made FROM a temporal standpoint (the judge's "now")
2. Evidence has ASPECTUAL STATUS: perfective (completed) or imperfective (ongoing)
3. Conditions may be fulfilled at T1 but not T2, or vice versa
4. SCOPE must match warrant: "evidence doesn't support X" ≠ "X doesn't occur"

When generating scenarios, ALWAYS include:
- When the evidence is from (temporal markers)
- Whether evidence is complete (perfective) or still accumulating (imperfective)
- The judge's temporal standpoint (implicit or explicit)
- Any temporal gap between evidence-time and judgment-time

Output JSON with these fields:
{
  "proposition": "A clear statement to be judged",
  "evidence": "Available evidence WITH temporal markers (when obtained, whether complete)",
  "conditions_for_truth": ["condition 1", "condition 2", "condition 3"],
  "temporal_context": {
    "evidence_timeframe": "When the evidence is from",
    "aspectual_status": "perfective (complete) or imperfective (ongoing)",
    "judge_standpoint": "The temporal position from which judgment is made",
    "scope_note": "What the evidence actually warrants vs. what might be over-claimed"
  },
  "scenario_notes": "Brief note on why this leads to the expected judgment"
}"""


TEMPORAL_FOCUS_PROMPTS = {
    "evidence_emergence": """
TEMPORAL FOCUS: EVIDENCE EMERGENCE

This scenario should exercise sequential evidence revelation. Design it so that:
- Evidence arrives in stages (past observations, recent data, current state)
- At intermediate stages, the virtually unconditioned is NOT yet grasped
- The judge must track which conditions become fulfilled as evidence accumulates
- Include at least one COMPLICATION: contradictory evidence, competing hypotheses, or evidence that UNFULFILLS a previously-fulfilled condition

The challenge: recognizing when to withhold judgment (evidence incomplete) vs. when to commit (virtually unconditioned grasped).""",
    "aspectual": """
TEMPORAL FOCUS: ASPECTUAL JUDGMENT

This scenario should exercise the perfective/imperfective distinction. Design it so that:
- Some evidence is COMPLETE (perfective): "the study concluded," "has been established"
- Some evidence is ONGOING (imperfective): "the study is ongoing," "data continues to accumulate"
- The correct judgment DEPENDS on recognizing that ongoing evidence cannot ground the same certainty as completed evidence

The challenge: not treating imperfective evidence as if it were perfective.""",
    "scope_matching": """
TEMPORAL FOCUS: SCOPE MATCHING

This scenario should exercise the distinction between evidence-relative and reality claims. Design it so that:
- The evidence warrants a LIMITED claim (e.g., "evidence does not support X")
- A system that OVER-COMMITS would claim "X does not occur" (exceeds warrant)
- A system that UNDER-COMMITS would refuse to judge even the limited claim

The challenge: matching judgment-scope precisely to evidential warrant. Over-commitment is error even if accidentally true.""",
    "temporal_deixis": """
TEMPORAL FOCUS: TEMPORAL DEIXIS INVERSION

This scenario should involve THE SAME PROPOSITION evaluated from TWO temporal standpoints:
- Standpoint T1 (earlier): Evidence available then leads to judgment J1
- Standpoint T2 (later): Different evidence available leads to different judgment J2

The proposition itself doesn't change; the FULFILLMENT OF CONDITIONS changes with standpoint.

The challenge: understanding that truth-status differs not because truth is relative, but because condition-fulfillment is temporally indexed.""",
    "retrospective": """
TEMPORAL FOCUS: RETROSPECTIVE REVISION

This scenario should involve reassessing a HISTORICAL JUDGMENT. Design it so that:
- An earlier judgment was made in good faith with available evidence
- New evidence or understanding has emerged
- The system must identify the LEVEL at which error occurred:
  * Level 1 (Attention): Relevant data was overlooked
  * Level 2 (Understanding): Wrong intelligibility/interpretation
  * Level 3 (Judgment): Correct understanding but wrong judgment
  * Level 4 (Decision): Correct judgment but failure to act
  * Cross-level: Bias cascade spanning multiple levels

The challenge: identifying error-level while treating original judges charitably.""",
}


JUDGMENT_SYSTEM_PROMPT = """You are performing the cognitive operation of JUDGMENT according to Lonergan's cognitional theory.

{lonergan_context}

THE VIRTUALLY UNCONDITIONED:
- Conditioned: The proposition being assessed
- Link: The connection between proposition and its conditions  
- Fulfillment: Evidence meeting conditions AT THIS STANDPOINT

TEMPORAL PRINCIPLES (apply to ALL judgments):

1. SCOPE-MATCHING: Your judgment must match evidential scope.
   - "Evidence does not support X" ≠ "X does not occur"
   - Over-commitment (claiming more than evidence warrants) is error even if accidentally true
   - Under-commitment (refusing to judge when evidence suffices) is also error

2. ASPECTUAL AWARENESS: Distinguish completed from ongoing.
   - Perfective evidence (completed): "The study concluded," "has established"
   - Imperfective evidence (ongoing): "The study continues," "is accumulating"
   - Ongoing evidence cannot ground the same judgment as completed evidence

3. STANDPOINT-RELATIVITY: Your judgment is FROM a temporal standpoint.
   - Acknowledge what evidence is available NOW vs. what might emerge
   - Do not import future knowledge into past-standpoint assessments
   - The virtually unconditioned is relative to available evidence

Output JSON:
{{
  "condition_fulfillment": [
    {{"condition": "c1", "status": "fulfilled|not_fulfilled|contested|ongoing", "evidence_bearing": "What evidence bears on this condition", "aspectual_note": "If relevant, whether evidence is complete or ongoing"}}
  ],
  "scope_assessment": {{
    "evidence_warrants": "What claim the evidence actually supports",
    "stronger_claim_would_require": "What would be needed to claim more"
  }},
  "judgment": "Yes|No|Insufficient",
  "reasoning": "Detailed explanation including temporal considerations"
}}

Remember:
- "Yes" = All conditions FULFILLED (perfective) by available evidence - grasp the virtually unconditioned
- "No" = At least one condition identifiably NOT fulfilled
- "Insufficient" = Cannot determine - evidence incomplete, ongoing, or contested"""


# ============ DOMAIN CONTEXT ============


def _get_domain_context(domain: str, use_principled: bool = True) -> str:
    """Get domain-specific guidance for scenario generation.

    Args:
        domain: Domain name
        use_principled: If True, use principled schema (with mode context).
                       If False, use legacy API.
    """
    # Try principled schema first
    if use_principled and domain in PRINCIPLED_DOMAINS:
        return get_principled_prompt_context(domain)

    # Fall back to legacy API
    if domain in ALL_DOMAINS:
        return get_principled_prompt_context(domain)

    # Fallback for very old domain names
    legacy_mapping = {
        "mathematical": "pure_mathematics",
        "empirical": "physics",
        "common_sense": "everyday_practical",
        "pop_science": "pop_science",  # Now in principled schema as dialectical
        "philosophic": "philosophical",
    }
    mapped = legacy_mapping.get(domain)
    if mapped and mapped in PRINCIPLED_DOMAINS:
        return get_principled_prompt_context(mapped)
    if mapped and mapped in ALL_DOMAINS:
        return get_principled_prompt_context(mapped)

    return f"Domain: {domain}\nGenerate a scenario appropriate for this domain."


def _get_mode_guidance(mode: FulfillmentMode) -> str:
    """Get mode-specific generation guidance for the system prompt."""
    mode_spec = get_mode_spec(mode)

    guidance = {
        FulfillmentMode.FORMAL: """
MODE: FORMAL (conditions fulfilled in intelligibility itself)

For FORMAL domains, the virtually unconditioned is grasped through:
- Logical/mathematical proof
- Coherence within an axiomatic system
- NO reversion to sense data required

Generate scenarios where:
- Evidence is conceptual/logical, not empirical
- Verification is by proof, not observation
- The question is whether the proof is valid/complete""",
        FulfillmentMode.EMPIRICAL: """
MODE: EMPIRICAL (conditions fulfilled by reversion to sense data)

For EMPIRICAL domains, the virtually unconditioned is grasped through:
- Measurements and observations
- Experimental results
- Statistical analysis

Generate scenarios where:
- Evidence comes from controlled observation
- Verification requires empirical data
- Claims must be compared against measured reality""",
        FulfillmentMode.COMMON_SENSE: """
MODE: COMMON SENSE (conditions fulfilled in pragmatic situation)

For COMMON SENSE domains, the virtually unconditioned is grasped through:
- Practical experience in concrete situations
- "It works" in this particular context
- Accumulated situational insight (not theoretical understanding)

Generate scenarios where:
- Evidence is practical/experiential
- Context matters crucially
- The judgment relates things to US, not to each other abstractly""",
        FulfillmentMode.DIALECTICAL: """
MODE: DIALECTICAL (conditions fulfilled in performative self-consistency)

For DIALECTICAL domains, the virtually unconditioned is grasped through:
- Whether affirming the position contradicts the act of affirming
- Positions vs. counterpositions
- Self-defeating claims

Generate scenarios where:
- The claim may undermine itself when affirmed
- Pop science vocabulary is used without proper verification
- Authority is invoked over evidence""",
    }

    return guidance.get(mode, "")


def _get_edge_case_prompt(edge_type: EdgeCaseType) -> str:
    """Get generation prompt for an edge case modifier."""
    spec = get_edge_case_spec(edge_type)
    return f"""
EDGE CASE MODIFIER: {edge_type.value.upper()}

{spec.description}

Challenge for the model: {spec.challenge}

{spec.generation_prompt}
"""


def _get_adversarial_prompt(adv_type: AdversarialType) -> str:
    """Get generation prompt for an adversarial context."""
    spec = get_adversarial_spec(adv_type)
    return f"""
ADVERSARIAL CONTEXT: {adv_type.value.upper()}

{spec.description}

Detection challenge: {spec.detection_challenge}
Key pitfall to avoid: {spec.key_pitfall}

{spec.generation_prompt}
"""


# ============ GENERATOR BASE ============


class BaseGenerator:
    """Base class for scenario and judgment generation."""

    model: str  # Set by subclasses

    def _call(
        self, system: str, user: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        raise NotImplementedError

    def _parse_json(self, response: str, context: str = "response") -> Dict[str, Any]:
        """Parse JSON from response, handling markdown wrapping and trailing text."""
        try:
            text = response
            # Strip markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            text = text.strip()

            # Try direct parse first
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Handle "Extra data" - find the complete JSON object
            # by matching braces
            if text.startswith("{"):
                depth = 0
                in_string = False
                escape = False
                for i, char in enumerate(text):
                    if escape:
                        escape = False
                        continue
                    if char == "\\":
                        escape = True
                        continue
                    if char == '"' and not escape:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            return json.loads(text[: i + 1])

            # Fallback: try to find JSON object boundaries
            raise json.JSONDecodeError("Could not find valid JSON object", text, 0)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse {context} JSON: {e}\nResponse: {response[:500]}"
            )

    def generate_scenario(
        self,
        domain: str,
        difficulty: str,
        expected_judgment: str,
        temporal_focus: Optional[str] = None,
        edge_case: Optional[EdgeCaseType] = None,
        adversarial_context: Optional[AdversarialType] = None,
    ) -> Dict[str, Any]:
        """Generate a judgment scenario with integrated temporal awareness.

        Args:
            domain: Domain name (from principled or legacy schema)
            difficulty: easy/medium/hard
            expected_judgment: Yes/No/Insufficient
            temporal_focus: Optional specific temporal challenge type
            edge_case: Optional edge case modifier (self-reference, vagueness, etc.)
            adversarial_context: Optional adversarial epistemic context
        """

        system = SCENARIO_SYSTEM_PROMPT
        domain_context = _get_domain_context(domain)

        # Get mode-specific guidance if using principled schema
        mode_guidance = ""
        if domain in PRINCIPLED_DOMAINS:
            spec = get_principled_domain(domain)
            mode_guidance = _get_mode_guidance(spec.mode)

        # Build user prompt
        temporal_instruction = ""
        if temporal_focus and temporal_focus in TEMPORAL_FOCUS_PROMPTS:
            temporal_instruction = TEMPORAL_FOCUS_PROMPTS[temporal_focus]

        # Edge case modifier
        edge_case_instruction = ""
        if edge_case:
            edge_case_instruction = _get_edge_case_prompt(edge_case)

        # Adversarial context
        adversarial_instruction = ""
        if adversarial_context:
            adversarial_instruction = _get_adversarial_prompt(adversarial_context)

        # Build judgment-specific guidance
        judgment_guidance = {
            "Yes": """For "Yes" judgment: Create a scenario where evidence CLEARLY FULFILLS all conditions.
The evidence should be complete (perfective) and positive. The virtually unconditioned is grasped.""",
            "No": """For "No" judgment: Create a scenario where evidence CLEARLY SHOWS at least one condition is NOT fulfilled.
The evidence should be sufficient to determine non-fulfillment - not merely absent, but definitively negative.""",
            "Insufficient": """For "Insufficient" judgment: Create a scenario where THE INFORMATION ITSELF IS MISSING.
NOT negative evidence, but ABSENT evidence. The question cannot yet be answered because:
- Key data has not been collected yet (study ongoing, results pending)
- Critical measurements have not been taken
- The event in question hasn't occurred yet (future contingent)
- Required information is genuinely inaccessible

WRONG: "The study found no effect" (this is No, not Insufficient)
RIGHT: "The study is still in progress, results expected next month" (genuinely Insufficient)
WRONG: "Evidence contradicts the claim" (this is No)
RIGHT: "No one has yet tested this specific claim" (genuinely Insufficient)""",
        }

        user = f"""Generate a {difficulty} scenario for the {domain} domain.
The scenario should lead to a judgment of: {expected_judgment}

{domain_context}
{mode_guidance}
{temporal_instruction}
{edge_case_instruction}
{adversarial_instruction}

{judgment_guidance.get(expected_judgment, "")}

Return ONLY valid JSON, no markdown."""

        response = self._call(system, user, max_tokens=16384)
        result = self._parse_json(response, "scenario")

        # Add metadata
        result["_meta"] = {
            "domain": domain,
            "difficulty": difficulty,
            "expected_judgment": expected_judgment,
            "temporal_focus": temporal_focus,
            "edge_case": edge_case.value if edge_case else None,
            "adversarial_context": adversarial_context.value
            if adversarial_context
            else None,
        }

        return result

    def generate_judgment(
        self,
        proposition: str,
        evidence: str,
        conditions: List[str],
        domain: str,
        temporal_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a judgment for a scenario with temporal awareness."""
        from src.dspy_cognitive.lonergan_context import get_judgment_context

        lonergan_context = get_judgment_context(domain)
        system = JUDGMENT_SYSTEM_PROMPT.format(lonergan_context=lonergan_context)

        conditions_str = "\n".join(f"- {c}" for c in conditions)

        # Include temporal context if available
        temporal_str = ""
        if temporal_context:
            temporal_str = f"""
Temporal Context:
- Evidence timeframe: {temporal_context.get("evidence_timeframe", "Not specified")}
- Aspectual status: {temporal_context.get("aspectual_status", "Not specified")}
- Judge standpoint: {temporal_context.get("judge_standpoint", "Present")}
- Scope note: {temporal_context.get("scope_note", "Standard assessment")}
"""

        user = f"""Proposition: {proposition}

Evidence: {evidence}

Conditions for truth:
{conditions_str}
{temporal_str}
Domain: {domain}

Evaluate each condition against the evidence with attention to temporal structure. Return ONLY valid JSON."""

        response = self._call(system, user, max_tokens=16384, temperature=0.3)
        return self._parse_json(response, "judgment")


# ============ ANTHROPIC GENERATOR ============


class AnthropicGenerator(BaseGenerator):
    """Generate scenarios and judgments using Anthropic Claude."""

    def __init__(self, model: str = "anthropic/claude-sonnet-4.5"):
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = ANTHROPIC_MODELS.get(model, model.replace("anthropic/", ""))
        logger.info(f"Initialized Anthropic generator with model: {self.model}")

    def _call(
        self, system: str, user: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        start = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        elapsed = time.time() - start
        block = response.content[0]
        if hasattr(block, "text"):
            text: str = block.text  # type: ignore[union-attr]
            logger.debug(f"API call: {elapsed:.1f}s, response: {len(text)} chars")
            return text
        raise ValueError(f"Unexpected response block type: {type(block)}")


# ============ GEMINI GENERATOR ============


class GeminiGenerator(BaseGenerator):
    """Generate scenarios and judgments using Google Gemini."""

    def __init__(self, model: str = "gemini/gemini-2.5-pro"):
        from google import genai

        self.client = genai.Client()
        self.model = GEMINI_MODELS.get(model, model.replace("gemini/", ""))
        logger.info(f"Initialized Gemini generator with model: {self.model}")

    def _call(
        self, system: str, user: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        from google.genai import types

        start = time.time()
        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        elapsed = time.time() - start
        if response.text is None:
            logger.warning(f"API call: {elapsed:.1f}s, response: EMPTY")
            raise ValueError("Gemini returned empty response")
        logger.debug(f"API call: {elapsed:.1f}s, response: {len(response.text)} chars")
        return response.text


# ============ UNIFIED GENERATOR ============


def get_generator(model: str) -> BaseGenerator:
    """Get appropriate generator for model."""
    provider = get_provider(model)
    if provider == "anthropic":
        return AnthropicGenerator(model)
    elif provider == "gemini":
        return GeminiGenerator(model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============ PARALLEL GENERATION SUPPORT ============

from dataclasses import dataclass


@dataclass
class GenerationTask:
    """Parameters for a single generation task."""

    index: int
    total: int
    domain: str
    difficulty: str
    expected_judgment: str
    temporal_focus: Optional[str]
    edge_case: Optional[EdgeCaseType]
    adversarial_context: Optional[AdversarialType]


@dataclass
class GenerationResult:
    """Result of a single generation task."""

    index: int
    success: bool
    example: Optional[Dict] = None
    error: Optional[str] = None
    elapsed: float = 0.0


def _generate_single_example(
    task: GenerationTask,
    generator: BaseGenerator,
    file_lock: threading.Lock,
    output_path: Path,
    print_lock: threading.Lock,
) -> GenerationResult:
    """Worker function to generate a single example.

    Thread-safe: uses locks for file writing and printing.
    """
    example_start = time.time()

    # Build display string
    modifiers = []
    if task.temporal_focus:
        modifiers.append(task.temporal_focus)
    if task.edge_case:
        modifiers.append(f"edge:{task.edge_case.value}")
    if task.adversarial_context:
        modifiers.append(f"adv:{task.adversarial_context.value}")
    modifier_str = f" [{', '.join(modifiers)}]" if modifiers else ""

    with print_lock:
        print(
            f"[{task.index + 1}/{task.total}] {task.domain}/{task.difficulty}/{task.expected_judgment}{modifier_str}...",
            end=" ",
            flush=True,
        )

    try:
        # Generate scenario
        scenario = generator.generate_scenario(
            domain=task.domain,
            difficulty=task.difficulty,
            expected_judgment=task.expected_judgment,
            temporal_focus=task.temporal_focus,
            edge_case=task.edge_case,
            adversarial_context=task.adversarial_context,
        )

        # Generate judgment
        judgment = generator.generate_judgment(
            proposition=scenario["proposition"],
            evidence=scenario["evidence"],
            conditions=scenario["conditions_for_truth"],
            domain=task.domain,
            temporal_context=scenario.get("temporal_context"),
        )

        # Build training example with varied system prompt
        system_prompt = generate_varied_training_prompt(generator)

        conditions_str = "\n".join(f"- {c}" for c in scenario["conditions_for_truth"])

        # Include temporal context in user content if present
        temporal_str = ""
        if "temporal_context" in scenario:
            tc = scenario["temporal_context"]
            temporal_str = f"""

Temporal context:
- Evidence timeframe: {tc.get("evidence_timeframe", "Not specified")}
- Aspectual status: {tc.get("aspectual_status", "Not specified")}"""

        user_content = f"""Proposition: {scenario["proposition"]}

Evidence: {scenario["evidence"]}

Conditions for truth:
{conditions_str}{temporal_str}"""

        # Handle both old-style and new-style judgment formats
        if isinstance(judgment.get("condition_fulfillment"), list):
            if isinstance(judgment["condition_fulfillment"][0], dict):
                # New style with detailed fulfillment
                fulfillment_lines = []
                for cf in judgment["condition_fulfillment"]:
                    status = cf.get("status", "unknown")
                    evidence = cf.get("evidence_bearing", "")
                    condition = cf.get("condition", "")
                    fulfillment_lines.append(f"- {condition}: {status} ({evidence})")
                fulfillment_str = "\n".join(fulfillment_lines)
            else:
                # Old style with simple strings
                fulfillment_str = "\n".join(
                    f"- {c}: {f}"
                    for c, f in zip(
                        scenario["conditions_for_truth"],
                        judgment["condition_fulfillment"],
                    )
                )
        else:
            fulfillment_str = str(judgment.get("condition_fulfillment", ""))

        # Include scope assessment if present
        scope_str = ""
        if "scope_assessment" in judgment:
            sa = judgment["scope_assessment"]
            scope_str = f"""

Scope assessment:
- Evidence warrants: {sa.get("evidence_warrants", "Not specified")}
- Stronger claim would require: {sa.get("stronger_claim_would_require", "Not specified")}"""

        # Format assistant content with JUDGMENT:/REASONING: markers
        assistant_content = f"""JUDGMENT: {judgment["judgment"]}
REASONING: {judgment["reasoning"]}

Condition fulfillment:
{fulfillment_str}{scope_str}"""

        # Get mode if principled domain
        mode = None
        if task.domain in PRINCIPLED_DOMAINS:
            spec = get_principled_domain(task.domain)
            mode = spec.mode.value

        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "level": "judgment",
                "domain": task.domain,
                "mode": mode,
                "judgment_type": judgment["judgment"],
                "difficulty": task.difficulty,
                "expected_judgment": task.expected_judgment,
                "generated_by": generator.model,
                "temporal_focus": task.temporal_focus,
                "edge_case": task.edge_case.value if task.edge_case else None,
                "adversarial_context": task.adversarial_context.value
                if task.adversarial_context
                else None,
                "has_temporal_context": "temporal_context" in scenario,
                "scenario_notes": scenario.get("scenario_notes", ""),
            },
        }

        elapsed = time.time() - example_start

        # Thread-safe file write
        with file_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(example) + "\n")

        with print_lock:
            print(f"✓ {judgment['judgment']} ({elapsed:.1f}s)")

        logger.info(f"Generated example {task.index + 1} to {output_path}")

        return GenerationResult(
            index=task.index,
            success=True,
            example=example,
            elapsed=elapsed,
        )

    except Exception as e:
        elapsed = time.time() - example_start
        with print_lock:
            print(f"✗ {e} ({elapsed:.1f}s)")
        logger.error(f"Example {task.index + 1} failed: {e}")

        return GenerationResult(
            index=task.index,
            success=False,
            error=str(e),
            elapsed=elapsed,
        )


def generate_examples(
    model: str,
    count: int,
    domains: List[str],
    difficulties: List[str],
    judgments: List[str],
    output_dir: Path,
    temporal_focus_ratio: float = 0.3,
    edge_case_enabled: bool = True,
    adversarial_ratio: float = 0.15,
    workers: int = 1,
) -> tuple[List[Dict], Path]:
    """Generate training examples with integrated temporal awareness.

    Args:
        model: Model to use for generation
        count: Number of examples to generate
        domains: List of domains to sample from
        difficulties: List of difficulty levels
        judgments: List of expected judgment types
        output_dir: Directory for output file
        temporal_focus_ratio: Fraction of examples with specific temporal focus (0.0-1.0)
        edge_case_enabled: Whether to probabilistically roll for edge cases
        adversarial_ratio: Fraction of examples with adversarial epistemic context
        workers: Number of parallel workers (1 = sequential)

    Returns:
        Tuple of (examples list, output path)
    """

    registry = DataRegistry()
    rng = random.Random()

    # Generate unique output filename
    output_filename = registry.generate_unique_filename("judgment", model)
    output_path = output_dir / output_filename

    print(f"\n{'=' * 60}")
    print(f"GENERATING {count} EXAMPLES WITH {model}")
    print(f"{'=' * 60}")
    print(f"Output: {output_path}")
    print(f"Workers: {workers}")
    print(f"Domains: {len(domains)} domains")
    print(f"  Principled: {sum(1 for d in domains if d in PRINCIPLED_DOMAINS)}")
    print(f"  Legacy: {sum(1 for d in domains if d not in PRINCIPLED_DOMAINS)}")
    print(f"Difficulties: {difficulties}")
    print(f"Judgments: {judgments}")
    print(f"Temporal focus ratio: {temporal_focus_ratio:.0%}")
    print(f"Edge cases: {'enabled' if edge_case_enabled else 'disabled'}")
    print(f"Adversarial ratio: {adversarial_ratio:.0%}")
    print()

    # Initialize output file (clear if exists)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        pass  # Create/truncate file
    logger.info(f"Initialized output file: {output_path}")

    # Track statistics
    stats = {
        "temporal_focus": 0,
        "edge_cases": {t.value: 0 for t in EdgeCaseType},
        "adversarial": {t.value: 0 for t in AdversarialType},
        "by_mode": {m.value: 0 for m in FulfillmentMode},
    }

    # Build cell distribution
    cells = [(d, diff, j) for d in domains for diff in difficulties for j in judgments]

    # Pre-generate all tasks with their parameters
    tasks: List[GenerationTask] = []
    for i in range(count):
        domain, difficulty, expected_judgment = random.choice(cells)

        # Determine temporal focus
        temporal_focus = None
        if rng.random() < temporal_focus_ratio:
            temporal_focus = random.choice(TEMPORAL_FOCUS_TYPES)
            stats["temporal_focus"] += 1

        # Roll for edge case (probabilistic based on mode)
        edge_case = None
        if edge_case_enabled and domain in PRINCIPLED_DOMAINS:
            spec = get_principled_domain(domain)
            edge_case = roll_edge_case(spec.mode, rng)
            if edge_case:
                stats["edge_cases"][edge_case.value] += 1

        # Roll for adversarial context
        adversarial_context = None
        if rng.random() < adversarial_ratio:
            adversarial_context = rng.choice(list(AdversarialType))
            stats["adversarial"][adversarial_context.value] += 1

        # Track mode statistics
        if domain in PRINCIPLED_DOMAINS:
            spec = get_principled_domain(domain)
            stats["by_mode"][spec.mode.value] += 1

        tasks.append(
            GenerationTask(
                index=i,
                total=count,
                domain=domain,
                difficulty=difficulty,
                expected_judgment=expected_judgment,
                temporal_focus=temporal_focus,
                edge_case=edge_case,
                adversarial_context=adversarial_context,
            )
        )

    # Execute generation
    examples: List[Dict] = []
    errors: List[Dict] = []

    file_lock = threading.Lock()
    print_lock = threading.Lock()

    if workers == 1:
        # Sequential execution
        generator = get_generator(model)
        for task in tasks:
            result = _generate_single_example(
                task=task,
                generator=generator,
                file_lock=file_lock,
                output_path=output_path,
                print_lock=print_lock,
            )
            if result.success and result.example:
                examples.append(result.example)
            elif not result.success:
                errors.append(
                    {
                        "index": task.index,
                        "domain": task.domain,
                        "temporal_focus": task.temporal_focus,
                        "edge_case": task.edge_case.value if task.edge_case else None,
                        "adversarial_context": task.adversarial_context.value
                        if task.adversarial_context
                        else None,
                        "error": result.error,
                    }
                )
    else:
        # Parallel execution
        # Create one generator per worker to avoid thread safety issues with API clients
        def worker_fn(task: GenerationTask) -> GenerationResult:
            # Each thread gets its own generator instance
            thread_generator = get_generator(model)
            return _generate_single_example(
                task=task,
                generator=thread_generator,
                file_lock=file_lock,
                output_path=output_path,
                print_lock=print_lock,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker_fn, task): task for task in tasks}

            for future in as_completed(futures):
                result = future.result()
                if result.success and result.example:
                    examples.append(result.example)
                elif not result.success:
                    task = futures[future]
                    errors.append(
                        {
                            "index": task.index,
                            "domain": task.domain,
                            "temporal_focus": task.temporal_focus,
                            "edge_case": task.edge_case.value
                            if task.edge_case
                            else None,
                            "adversarial_context": task.adversarial_context.value
                            if task.adversarial_context
                            else None,
                            "error": result.error,
                        }
                    )

    # Log to registry
    run = GenerationRun.create(
        model=model,
        num_examples=len(examples),
        output_file=str(output_path),
        domains=domains,
        difficulties=difficulties,
        judgments=judgments,
        config={
            "errors": len(errors),
            "temporal_focus_ratio": temporal_focus_ratio,
            "temporal_focus_types": TEMPORAL_FOCUS_TYPES,
            "edge_case_enabled": edge_case_enabled,
            "adversarial_ratio": adversarial_ratio,
            "statistics": stats,
        },
        notes=f"Generated {len(examples)} examples ({temporal_focus_ratio:.0%} temporal, {adversarial_ratio:.0%} adversarial), {len(errors)} errors",
    )
    registry.log_run(run)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Examples generated: {len(examples)}")
    print(f"  By mode:")
    for mode, count in stats["by_mode"].items():
        if count > 0:
            print(f"    - {mode}: {count}")
    print(f"  With temporal focus: {stats['temporal_focus']}")
    edge_total = sum(stats["edge_cases"].values())
    if edge_total > 0:
        print(f"  With edge cases: {edge_total}")
        for etype, count in stats["edge_cases"].items():
            if count > 0:
                print(f"    - {etype}: {count}")
    adv_total = sum(stats["adversarial"].values())
    if adv_total > 0:
        print(f"  With adversarial context: {adv_total}")
        for atype, count in stats["adversarial"].items():
            if count > 0:
                print(f"    - {atype}: {count}")
    print(f"Errors: {len(errors)}")
    print(f"Output: {output_path}")
    print(f"Run ID: {run.run_id}")

    if errors:
        print(f"\nErrors:")
        for err in errors[:5]:
            print(f"  [{err['index']}] {err['domain']}: {err['error'][:80]}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    return examples, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate training examples with temporal awareness"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4.5",
        help="Model to use (anthropic/claude-sonnet-4.5, gemini/gemini-2.5-pro, etc.)",
    )
    parser.add_argument(
        "--count", type=int, default=20, help="Number of examples to generate"
    )

    # Domain selection - support both principled and legacy
    principled_domain_names = list(PRINCIPLED_DOMAINS.keys())
    parser.add_argument(
        "--domains",
        nargs="+",
        default=principled_domain_names,  # Default to principled domains
        help=f"Domains to generate for. Use --list-domains to see all.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["formal", "empirical", "common_sense", "dialectical", "all"],
        default=None,
        help="Filter to domains with specific fulfillment mode. Overrides --domains.",
    )
    parser.add_argument(
        "--empirical-level",
        type=str,
        choices=["physical", "biological", "sensitive", "intelligent"],
        default=None,
        help="Filter empirical domains to specific integration level.",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List all available domains and exit",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easy", "medium", "hard"],
        help="Difficulty levels",
    )
    parser.add_argument(
        "--judgments",
        nargs="+",
        default=["Yes", "No", "Insufficient"],
        help="Judgment types to generate",
    )
    parser.add_argument(
        "--temporal-focus-ratio",
        type=float,
        default=0.3,
        help="Fraction of examples with specific temporal focus (0.0-1.0).",
    )
    parser.add_argument(
        "--edge-cases",
        action="store_true",
        default=True,
        help="Enable probabilistic edge case modifiers (default: True)",
    )
    parser.add_argument(
        "--no-edge-cases",
        action="store_true",
        help="Disable edge case modifiers",
    )
    parser.add_argument(
        "--adversarial-ratio",
        type=float,
        default=0.15,
        help="Fraction of examples with adversarial epistemic context (0.0-1.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/oracle_generated"),
        help="Output directory",
    )
    parser.add_argument(
        "--show-registry", action="store_true", help="Show registry summary and exit"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file for debug output (enables DEBUG level to file)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for generation (default: 1 = sequential)",
    )

    args = parser.parse_args()

    # Set up file logging if requested
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(args.log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
            )
        )
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.info(f"Debug logging to: {args.log_file}")

    if args.show_registry:
        registry = DataRegistry()
        print(registry.get_summary())
        return

    if args.list_domains:
        from src.domains import print_schema_summary

        print_schema_summary()
        return

    # Determine domains based on mode filter
    domains = args.domains
    if args.mode:
        if args.mode == "all":
            domains = list(PRINCIPLED_DOMAINS.keys())
        else:
            mode = FulfillmentMode(args.mode)
            if args.empirical_level and mode == FulfillmentMode.EMPIRICAL:
                level = EmpiricalLevel(args.empirical_level)
                domains = [d.name for d in get_domains_by_empirical_level(level)]
            else:
                domains = [d.name for d in get_domains_by_mode(mode)]
        print(f"Filtered to {len(domains)} domains by mode={args.mode}")
        if args.empirical_level:
            print(f"  (empirical level: {args.empirical_level})")

    edge_case_enabled = args.edge_cases and not args.no_edge_cases

    generate_examples(
        model=args.model,
        count=args.count,
        domains=domains,
        difficulties=args.difficulties,
        judgments=args.judgments,
        output_dir=args.output_dir,
        temporal_focus_ratio=args.temporal_focus_ratio,
        edge_case_enabled=edge_case_enabled,
        adversarial_ratio=args.adversarial_ratio,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
