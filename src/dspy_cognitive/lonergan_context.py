"""
Lonergan Context for Oracle Augmentation

Provides focused excerpts from Lonergan's cognitional theory to augment
the oracle's judgment generation, ensuring alignment with the theoretical
framework rather than generic LLM priors.

Source material: docs/lonergan_synopsis.md (full synopsis)
This module provides compact, task-focused extracts for efficient context usage.
"""

from pathlib import Path

# Path to full synopsis
SYNOPSIS_PATH = Path(__file__).parent.parent.parent / "docs" / "lonergan_synopsis.md"

# Operational anchors in cognitively precise languages
OPERATIONAL_ANCHORS = """
## Operational Anchors (Cross-Linguistic Precision)

The cognitive operations have precise terms in languages with systematic philosophical traditions:

| Operation | Greek | Hebrew | Sanskrit | Latin |
|-----------|-------|--------|----------|-------|
| **Attention** | *prosoche* (heeding) | *shema* (hear/attend) | *dharana* (holding) | *attentio* |
| **Understanding** | *noesis* (intellectual grasp) | *binah* (discernment) | *vijnana* (discriminating awareness) | *intelligere* (to read between) |
| **Judgment** | *krisis* (separation/decision) | *mishpat* (right-ordering) | *viveka* (discernment real/unreal) | *iudicium* |
| **Decision** | *prohairesis* (deliberate choice) | *ratzon* (will) | *sankalpa* (resolve) | *electio* |

**Key distinctions these languages preserve:**

- Greek *episteme* (knowledge through demonstration) vs *doxa* (opinion) - the difference between judgment and mere assertion
- Hebrew *da'at* (intimate knowing) vs *yada* (to know factually) - knowledge that transforms the knower
- Sanskrit *buddhi* (discriminative intellect) vs *manas* (mind) - the faculty that discerns truth from appearance
- Latin *intellectus* (insight INTO) vs *ratio* (discursive reasoning) - direct grasp vs inferential process

**The virtually unconditioned** (*conditionale conditionatum*) is that which HAS conditions but whose conditions ARE fulfilled. *Krisis* (judgment) occurs when *viveka* (discernment) grasps the fulfillment of conditions through *noesis* (intellectual intuition), not mere *ratio* (calculation).
"""

# Compact context for judgment operations - focused on the virtually unconditioned
JUDGMENT_CONTEXT = """
## The Virtually Unconditioned: Structure of Judgment (*Krisis* / *Viveka*)

Judgment (*krisis*, *mishpat*) is the act of grasping that conditions for a proposition have been fulfilled (or not). It is *noesis* (intellectual grasp), not mere *doxa* (opinion).

**The Schema:**
1. **The Conditioned**: The proposition being tested
2. **The Link**: The connection between proposition and its conditions (if X, Y, Z, then proposition)
3. **The Fulfillment**: Whether conditions are fulfilled in the evidence

**Three Possible Judgments:**
- **Yes (Affirm)**: All conditions ARE fulfilled - grasp the virtually unconditioned
- **No (Deny)**: At least one condition is DEFINITIVELY NOT fulfilled
- **Insufficient (Withhold)**: Cannot determine whether conditions are fulfilled

**CRITICAL DISTINCTION - "No" vs "Insufficient":**
- "Insufficient" = We lack the evidence to evaluate the conditions
- "No" = We CAN evaluate the conditions and they are NOT met

## Domain-Specific Fulfillment

**Mathematical**: Conditions fulfilled in formal coherence (proof validity)
**Empirical Scientific**: Conditions fulfilled by reversion to sensible data (experimental verification)
**Common Sense**: Conditions fulfilled in practical situation ("it works")
**Pop Science / Common Nonsense**: PRESENTS unfulfilled conditions AS IF fulfilled
**Philosophic**: Conditions fulfilled in performative self-consistency

## The Pop Science Counterposition

Pop science mimics the FORM of scientific judgment without the SUBSTANCE of verification.

**Key markers:**
- Appeals to authority over evidence
- Correlation claimed as causation WITHOUT establishing causal mechanism
- Uses scientific vocabulary without understanding
- Presents preliminary findings as established fact

**CRITICAL**: When a pop science claim asserts "X is proven" but the conditions for proof are NOT fulfilled:
- The judgment is "No" (the claim of proof is FALSE)
- NOT "Insufficient" (we CAN evaluate that the proof claim is unfounded)

The evidence may show: "This study found correlation" - but if the claim is "This proves causation,"
we can DEFINITIVELY judge "No" because the conditions for causal proof are identifiably unfulfilled.

## Counterpositions Are Self-Defeating

A counterposition contradicts the performance of affirming it.

Example: "Knowledge is nothing but neural activity"
- To affirm this requires intelligent grasping and reasonable affirming
- But if true, this very affirming is "nothing but" neural activity
- Neural activity qua neural activity cannot be TRUE - only neural
- The position defeats itself

Judgment on counterpositions: "No" - because the conditions for coherent affirmation are not met
(affirming reasonably presupposes that affirmation is more than what the counterposition claims).
"""

# Extended context including domain scheme for complex cases
EXTENDED_JUDGMENT_CONTEXT = """
## The Four Levels of Cognitive Operation

| Level | Operation | Question | Output |
|-------|-----------|----------|--------|
| 1. Empirical | Sensing, perceiving | (presents data) | Presentations |
| 2. Intelligent | Inquiry, insight, formulation | What is it? Why? | Propositions (hypotheses) |
| 3. Rational | Reflection, grasping unconditioned | Is it so? | Judgment (yes/no/insufficient) |
| 4. Responsible | Deliberation, evaluation, decision | Is it good? | Commitment to action |

## Structure of Judgment by Domain

### Mathematical Domain
- **Conditioned**: Theorem/proposition
- **Link**: If axioms/definitions, then theorem
- **Fulfillment**: Grasped in the PROOF - internal coherence
- **No reversion to sense data** - fulfilled in intelligibility itself

### Empirical Scientific Domain
- **Conditioned**: Hypothesis/theory
- **Link**: If hypothesis, then these observable predictions
- **Fulfillment**: Experimental verification - data matches predictions
- **Requires reversion to sense data**

### Common Sense Domain
- **Conditioned**: "This is what's happening / This will work"
- **Link**: Accumulated insights about how things work
- **Fulfillment**: Pragmatic - does it work?

### Pop Science / Common Nonsense Domain
- **Conditioned**: Pseudo-proposition presented as scientific
- **Link**: Appeals to authority, popularity, vocabulary
- **Fulfillment**: NOT VERIFIED - but presented as if it were
- **Detection**: "The question is NOT how many people say it is obvious, nor how great is their authority, but simply what is the evidence"

### Philosophic Domain
- **Conditioned**: Position on knowing, being, objectivity
- **Link**: If grasping intelligently and affirming reasonably, then coherent with that activity
- **Fulfillment**: The act of knowing does not contradict the content known
- **Criterion**: Can this be held coherently by one who grasps it intelligently?

## Positions vs Counterpositions

**Basic Position (three components):**
1. The Real = what is intelligently grasped and reasonably affirmed
2. Knowing = intelligent inquiry + critical reflection
3. Objectivity = consequence of authentic cognitive operation

**Basic Counterposition (three components):**
1. The Real = "already out there now" (biological criterion)
2. Knowing = "taking a look" (ocular metaphor)
3. Objectivity = extroversion, correlation, consensus

Counterpositions are SELF-DEFEATING because affirming them reasonably presupposes the positions they deny.

## The Canon of Parsimony

"The question is NOT how many people say it is obvious, nor how great is their authority and renown, but simply what is the evidence."

- Exclude the unverifiable
- Distinguish formally unconditioned from virtually unconditioned
- Use evidence, not authority
"""

# Context specifically for pop science detection
POP_SCIENCE_CONTEXT = """
## Detecting Pop Science / Common Nonsense

Pop science uses the FORM of scientific judgment without the SUBSTANCE of verification.

### What Pop Science Does
1. Reports correlation as if it established causation
2. Uses scientific vocabulary without operational understanding
3. Appeals to authority ("scientists say") over evidence
4. Treats theoretical constructs as "tiny picture-able things"
5. Presents preliminary findings as established facts

### The Critical Distinction

**Empirical Science:**
- Hypothesis → Predictions → Experimental verification → Judgment
- Conditions for truth: controlled experiment, replication, causal mechanism

**Pop Science:**
- Claim → Authority appeal → Confident assertion
- Conditions for truth: NOT EVALUATED, merely assumed

### Judgment on Pop Science Claims

When a pop science source claims "X is proven to cause Y":

**Check the conditions:**
1. Was there a controlled experiment? (Often: observational study only)
2. Was correlation distinguished from causation? (Often: conflated)
3. Was there independent replication? (Often: single study)
4. Is the causal mechanism established? (Often: speculation)

**If conditions are identifiably NOT met:**
- Judgment = "No" (the claim of proof is FALSE)
- NOT "Insufficient" - we CAN determine the conditions are unfulfilled

**"Insufficient" applies when:**
- Evidence is genuinely ambiguous
- We cannot evaluate whether conditions are met
- The question exceeds available data

**"No" applies when:**
- Conditions are identifiably unfulfilled
- The claim asserts more than the evidence supports
- The "proof" is demonstrably not a proof

### Example Analysis

Claim: "Eating chocolate is proven to improve memory, according to a study."

Conditions for this claim to be true:
1. Controlled experiment (not just observation)
2. Causal mechanism established (not just correlation)
3. Independent replication
4. Effect size clinically significant

Evidence typically available:
- Single observational study
- Correlation found, causation assumed
- No replication mentioned
- Author may sell chocolate products

Judgment: **No** - The claim that chocolate is PROVEN to improve memory is FALSE.
The conditions for proof are identifiably unfulfilled.
(NOT "Insufficient" - we can definitively evaluate that the proof claim fails.)
"""


def get_judgment_context(domain: str = None, include_extended: bool = False, include_anchors: bool = True) -> str:
    """
    Get appropriate Lonergan context for judgment generation.

    Args:
        domain: Specific domain if known (for focused context)
        include_extended: Whether to include extended domain scheme
        include_anchors: Whether to include cross-linguistic operational anchors

    Returns:
        Context string to include in oracle prompt
    """
    base = OPERATIONAL_ANCHORS + "\n" if include_anchors else ""

    if domain == "pop_science":
        return base + JUDGMENT_CONTEXT + "\n" + POP_SCIENCE_CONTEXT
    elif include_extended:
        return base + JUDGMENT_CONTEXT + "\n" + EXTENDED_JUDGMENT_CONTEXT
    else:
        return base + JUDGMENT_CONTEXT


PROBABILISTIC_JUDGMENT_CONTEXT = """
## Probabilistic Judgment: Three Distinct Questions

Lonergan treats probability not as guessing but as a specific type of judgment with distinct components:

### 1. Ideal/Hypothetical Frequency (Theoretical Soundness)
**Question**: Is this probability theoretically coherent?
**Example**: "Is P(rolling 6 on fair die) = 1/6?"
**Fulfillment**: In the intelligibility of the concepts (what "fair die" means)
**Type**: Mathematical/conceptual - no reversion to sense data required

### 2. Actual Empirical Frequency (Data Verification)
**Question**: What does the data actually show?
**Example**: "In 1000 rolls, how many sixes did we observe?"
**Fulfillment**: By reversion to the measured data
**Type**: Empirical - requires actual measurement

### 3. Systematic vs. Nonsystematic Divergence (Statistical Judgment)
**Question**: Does the actual frequency diverge from the ideal only randomly (nonsystematically)?
**Example**: "We got 140 sixes instead of expected 167. Is this within random variation, or does it indicate bias?"
**Fulfillment**: In whether the divergence pattern is intelligible as systematic or not

**Key insight**: Classical laws abstract from nonsystematic divergence. Statistical laws concern the *frequency of nonsystematic divergence itself*.

### Judgment on Probabilistic Claims

When evaluating a probabilistic proposition:

1. **If the question is about theoretical soundness**:
   - Judgment = Yes/No based on conceptual coherence
   - "P(heads) = 0.5 for a fair coin" → Yes (by definition of fair)

2. **If the question is about empirical validation**:
   - Judgment = Yes/No/Insufficient based on whether data matches ideal within expected variation
   - Requires: actual frequency data, sample size, statistical test for significance

3. **If the question is about whether a process is random**:
   - Judgment = Yes (nonsystematic) / No (systematic pattern detected) / Insufficient
   - Requires: enough data to distinguish random from patterned divergence

**Warning**: Pop science often conflates these levels:
- "Studies show X happens 30% of the time" - Is this an ideal frequency, an empirical observation, or both?
- "The probability is 0.05" - Is this a theoretical derivation or an empirical estimate?
"""


def get_verification_context() -> str:
    """Get context for judgment verification."""
    return JUDGMENT_CONTEXT + "\n" + POP_SCIENCE_CONTEXT


def load_full_synopsis() -> str:
    """
    Load the full Lonergan synopsis from docs/lonergan_synopsis.md.

    Use this for comprehensive context when compact excerpts are insufficient.
    Note: This adds significant tokens to the context.
    """
    if SYNOPSIS_PATH.exists():
        return SYNOPSIS_PATH.read_text()
    else:
        raise FileNotFoundError(f"Synopsis not found at {SYNOPSIS_PATH}")


def get_judgment_context_with_synopsis(domain: str = None) -> str:
    """
    Get judgment context augmented with the full synopsis.

    Use for difficult cases requiring comprehensive theoretical grounding.
    """
    base_context = get_judgment_context(domain=domain)
    synopsis = load_full_synopsis()

    return f"""# Focused Judgment Guidelines
{base_context}

# Full Theoretical Background (from Lonergan's Insight)
{synopsis}"""
