"""
Principled Domain Schema for Judgment Training

Based on Lonergan's analysis of judgment structure in *Insight*, this schema
differentiates domains by MODE OF FULFILLMENT - where the virtually unconditioned
is grasped - rather than by arbitrary subject-matter aggregation.

Key principles:
1. Mode of fulfillment as primary differentiator (Formal, Empirical, Common Sense, Dialectical)
2. Common sense as ground - the ocean in which islands of methodical intelligence float
3. Epistemic context as dimension - cooperative/adversarial cuts across all modes
4. Edge cases as probabilistic modifiers - not separate domains

See: docs/principled_domain_schema.md for full philosophical grounding.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random


# ============================================================================
# PART 1: PRINCIPLED SCHEMA (NEW)
# ============================================================================


class FulfillmentMode(Enum):
    """
    Primary genus of judgment - WHERE the virtually unconditioned is grasped.

    This is the fundamental differentiator of judgment types.
    """

    FORMAL = "formal"  # Intelligibility itself (no reversion to sense data)
    EMPIRICAL = "empirical"  # Sense data via verified correlations
    COMMON_SENSE = "common_sense"  # Pragmatic situation as related to us
    DIALECTICAL = "dialectical"  # Performative self-consistency


class EmpiricalLevel(Enum):
    """
    Within Empirical mode, sciences are differentiated by HIGHER INTEGRATION.
    Each level makes systematic what the lower must regard as coincidental.
    """

    PHYSICAL = "physical"  # Mass, charge, field vectors (physics, chemistry)
    BIOLOGICAL = "biological"  # Organic schemes of recurrence (biology, biochemistry)
    SENSITIVE = (
        "sensitive"  # Percepts, images, affects (animal psychology, neuroscience)
    )
    INTELLIGENT = (
        "intelligent"  # Insights, judgments, decisions (human sciences, history)
    )


class CommonSenseVariety(Enum):
    """
    Varieties of common sense judgment.
    """

    EVERYDAY = "everyday"  # Cooking, navigation, household
    SOCIAL = "social"  # Reading situations, predicting behavior
    PROFESSIONAL = "professional"  # Methodically-informed practical judgment
    GENERATIONAL = "generational"  # Proverbs, traditional practices
    SUBCULTURAL = "subcultural"  # In-group knowledge


class EpistemicContext(Enum):
    """
    Epistemic context is a DIMENSION cutting across all modes, not a separate domain.

    It characterizes the speaker's orientation in expression:
    - Cooperative: Speaker's F is oriented toward communicating A accurately
    - Adversarial: Speaker's F is oriented toward producing false belief about A
    """

    COOPERATIVE = "cooperative"
    ADVERSARIAL = "adversarial"


class AdversarialType(Enum):
    """
    Types of adversarial epistemic context.
    """

    DECEPTION = "deception"  # Lies, fraud, scams
    MANIPULATION = "manipulation"  # Emotional manipulation, gaslighting
    PROPAGANDA = "propaganda"  # Institutional messaging to persuade
    STRATEGIC_AMBIGUITY = "strategic_ambiguity"  # Deliberately vague
    ADVERSARIAL_DEBATE = "adversarial_debate"  # Arguments to win, not find truth
    SARCASM_IRONY = (
        "sarcasm_irony"  # Non-literal (cooperative intent, adversarial form)
    )


class EdgeCaseType(Enum):
    """
    Edge cases are PROBABILISTIC MODIFIERS that can complicate any judgment.
    Not separate domains but complications that can arise in any context.
    """

    SELF_REFERENCE = "self_reference"  # Proposition refers to itself
    VAGUENESS = "vagueness"  # No sharp boundary (sorites)
    FUTURE_CONTINGENT = "future_contingent"  # Event hasn't occurred
    COUNTERFACTUAL = "counterfactual"  # No actual data to revert to
    CATEGORY_ERROR = "category_error"  # Wrong categorical presupposition


@dataclass
class ModeSpec:
    """Specification for a mode of fulfillment."""

    mode: FulfillmentMode
    description: str
    fulfillment_location: str
    reversion_required: bool
    evidential_standards: str
    characteristic_pitfalls: List[str]


@dataclass
class DomainSpec:
    """
    A specific domain within the principled schema.

    Combines mode, level (if empirical), variety (if common sense),
    and provides domain-specific guidance.
    """

    name: str
    mode: FulfillmentMode
    empirical_level: Optional[EmpiricalLevel] = None
    common_sense_variety: Optional[CommonSenseVariety] = None
    description: str = ""
    evidential_standards: str = ""
    characteristic_pitfalls: List[str] = field(default_factory=list)
    example_propositions: List[str] = field(default_factory=list)

    @property
    def full_path(self) -> str:
        """Return the full hierarchical path of this domain."""
        parts = [self.mode.value]
        if self.empirical_level:
            parts.append(self.empirical_level.value)
        if self.common_sense_variety:
            parts.append(self.common_sense_variety.value)
        parts.append(self.name)
        return "/".join(parts)


@dataclass
class EdgeCaseSpec:
    """Specification for an edge case modifier."""

    edge_type: EdgeCaseType
    description: str
    challenge: str
    probability: float  # Base probability of occurrence (0.0 to 1.0)
    compatible_modes: List[FulfillmentMode]  # Which modes this can apply to
    generation_prompt: str  # How to instruct generator to include this


# ============================================================================
# MODE DEFINITIONS
# ============================================================================

MODE_SPECS: Dict[FulfillmentMode, ModeSpec] = {
    FulfillmentMode.FORMAL: ModeSpec(
        mode=FulfillmentMode.FORMAL,
        description="Conditions fulfilled in intelligibility itself - no reversion to sense data required",
        fulfillment_location="Coherence of the proof/system",
        reversion_required=False,
        evidential_standards="Deductive proof from axioms. Logical derivation. Internal consistency.",
        characteristic_pitfalls=[
            "Confusing plausibility with proof",
            "Hidden assumptions",
            "Incomplete induction (checking cases vs proving universally)",
            "Confusing validity with soundness",
        ],
    ),
    FulfillmentMode.EMPIRICAL: ModeSpec(
        mode=FulfillmentMode.EMPIRICAL,
        description="Conditions fulfilled by reversion to sense data via verified correlations",
        fulfillment_location="Measurements, observations, experimental results",
        reversion_required=True,
        evidential_standards="Controlled experiments, reproducible measurements, statistical analysis.",
        characteristic_pitfalls=[
            "Confusing models with reality",
            "Extrapolating beyond tested regimes",
            "Correlation-causation conflation",
            "Ignoring measurement uncertainty",
        ],
    ),
    FulfillmentMode.COMMON_SENSE: ModeSpec(
        mode=FulfillmentMode.COMMON_SENSE,
        description="Conditions fulfilled in pragmatic situation as related to us",
        fulfillment_location="Concrete particular circumstances, 'it works'",
        reversion_required=True,
        evidential_standards="Direct experience, practical success, accumulated situational insights.",
        characteristic_pitfalls=[
            "General bias (disregarding larger issues, long-term results)",
            "Overgeneralizing from limited experience",
            "Confusing familiarity with understanding",
            "Assuming what works here works everywhere",
        ],
    ),
    FulfillmentMode.DIALECTICAL: ModeSpec(
        mode=FulfillmentMode.DIALECTICAL,
        description="Conditions fulfilled in performative self-consistency",
        fulfillment_location="Whether the act of affirming contradicts the content affirmed",
        reversion_required=True,  # To the act of affirming itself
        evidential_standards="Can this position be held coherently by one who grasps it intelligently and affirms it reasonably?",
        characteristic_pitfalls=[
            "Missing self-defeating implications",
            "Failing to distinguish positions from counterpositions",
            "Confusing rhetorical strength with logical coherence",
        ],
    ),
}


# ============================================================================
# DOMAIN SPECIFICATIONS (PRINCIPLED)
# ============================================================================

PRINCIPLED_DOMAINS: Dict[str, DomainSpec] = {}

# ----- FORMAL MODE -----

_formal_domains = [
    DomainSpec(
        name="pure_mathematics",
        mode=FulfillmentMode.FORMAL,
        description="Number theory, algebra, analysis - truth by proof",
        evidential_standards="Deductive proof from axioms. No empirical evidence needed.",
        characteristic_pitfalls=[
            "Confusing plausibility with proof",
            "Incomplete induction (checking cases vs proving universally)",
            "Hidden assumptions",
        ],
        example_propositions=[
            "All prime numbers greater than 2 are odd",
            "The square root of 2 is irrational",
            "There exist infinitely many twin primes",
        ],
    ),
    DomainSpec(
        name="formal_logic",
        mode=FulfillmentMode.FORMAL,
        description="Propositional and predicate logic, validity, soundness",
        evidential_standards="Logical derivation. Truth-preserving inference.",
        characteristic_pitfalls=[
            "Affirming the consequent",
            "Denying the antecedent",
            "Confusing validity with soundness",
        ],
        example_propositions=[
            "This argument is valid",
            "The conclusion follows from the premises",
            "These two statements are logically equivalent",
        ],
    ),
    DomainSpec(
        name="set_theory",
        mode=FulfillmentMode.FORMAL,
        description="Sets, cardinality, infinity, foundations",
        evidential_standards="Axiomatic derivation (ZFC or alternatives).",
        characteristic_pitfalls=[
            "Naive set comprehension (Russell's paradox)",
            "Confusing countable and uncountable",
            "Axiom of choice implications",
        ],
        example_propositions=[
            "The set of all sets does not exist",
            "The real numbers are uncountable",
            "Every vector space has a basis",
        ],
    ),
]

# ----- EMPIRICAL MODE - PHYSICAL LEVEL -----

_empirical_physical = [
    DomainSpec(
        name="physics",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.PHYSICAL,
        description="Mechanics, electromagnetism, thermodynamics, quantum, relativity",
        evidential_standards="Controlled experiments, mathematical models, reproducible measurements.",
        characteristic_pitfalls=[
            "Confusing models with reality",
            "Extrapolating beyond tested regimes",
            "Ignoring measurement uncertainty",
        ],
        example_propositions=[
            "The speed of light is constant in all reference frames",
            "Energy is conserved in this closed system",
            "This particle has a definite position before measurement",
        ],
    ),
    DomainSpec(
        name="chemistry",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.PHYSICAL,
        description="Atomic structure, reactions, bonding, thermochemistry",
        evidential_standards="Spectroscopic data, reaction yields, stoichiometry.",
        characteristic_pitfalls=[
            "Ignoring reaction conditions",
            "Confusing correlation with mechanism",
            "Overlooking competing reactions",
        ],
        example_propositions=[
            "This reaction is exothermic",
            "The compound is an acid",
            "These molecules will form a stable bond",
        ],
    ),
    DomainSpec(
        name="astronomy",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.PHYSICAL,
        description="Stellar evolution, cosmology, planetary science",
        evidential_standards="Spectroscopy, parallax, gravitational effects, cosmic background.",
        characteristic_pitfalls=[
            "Selection effects in observations",
            "Model dependence of distance measurements",
            "Confusing absence of evidence with evidence of absence",
        ],
        example_propositions=[
            "This star will become a supernova",
            "The universe is expanding",
            "This exoplanet could support life",
        ],
    ),
    DomainSpec(
        name="geology",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.PHYSICAL,
        description="Earth history, plate tectonics, stratigraphy",
        evidential_standards="Radiometric dating, fossil record, seismic data.",
        characteristic_pitfalls=[
            "Uniformitarianism vs catastrophism balance",
            "Dating method limitations",
            "Incomplete stratigraphic record",
        ],
        example_propositions=[
            "This rock formation is older than that one",
            "These continents were once joined",
            "This event caused the mass extinction",
        ],
    ),
]

# ----- EMPIRICAL MODE - BIOLOGICAL LEVEL -----

_empirical_biological = [
    DomainSpec(
        name="biochemistry",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.BIOLOGICAL,
        description="Proteins, enzymes, metabolism, molecular biology",
        evidential_standards="Assays, crystallography, sequencing, kinetic data.",
        characteristic_pitfalls=[
            "In vitro vs in vivo differences",
            "Ignoring cellular context",
            "Overgeneralizing from model organisms",
        ],
        example_propositions=[
            "This enzyme catalyzes the reaction",
            "The protein folds into this structure",
            "This pathway is the primary metabolic route",
        ],
    ),
    DomainSpec(
        name="biology",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.BIOLOGICAL,
        description="Ecology, evolution, genetics, organismal biology",
        evidential_standards="Field observations, controlled experiments, phylogenetic analysis.",
        characteristic_pitfalls=[
            "Just-so stories (adaptationist fallacy)",
            "Confusing correlation with causation in ecology",
            "Genetic determinism",
        ],
        example_propositions=[
            "This trait is an adaptation",
            "These species share a common ancestor",
            "The population is in Hardy-Weinberg equilibrium",
        ],
    ),
]

# ----- EMPIRICAL MODE - SENSITIVE LEVEL -----

_empirical_sensitive = [
    DomainSpec(
        name="neuroscience",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.SENSITIVE,
        description="Neural mechanisms, brain function, cognition",
        evidential_standards="Imaging, electrophysiology, lesion studies, behavioral data.",
        characteristic_pitfalls=[
            "Reverse inference (inferring mental states from activation)",
            "Localizationism (one area = one function)",
            "Conflating correlation with causation",
        ],
        example_propositions=[
            "This brain region is necessary for the function",
            "The neural activity causes the behavior",
            "Consciousness arises from this process",
        ],
    ),
    DomainSpec(
        name="animal_cognition",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.SENSITIVE,
        description="Non-human animal minds, comparative psychology",
        evidential_standards="Controlled behavioral experiments, avoiding anthropomorphism and its denial.",
        characteristic_pitfalls=[
            "Anthropomorphism (over-attributing human traits)",
            "Anthropodenial (under-attributing mental states)",
            "Morgan's Canon misapplication",
        ],
        example_propositions=[
            "This animal understands causation",
            "The behavior indicates self-awareness",
            "This species has theory of mind",
        ],
    ),
]

# ----- EMPIRICAL MODE - INTELLIGENT LEVEL (Human Sciences) -----

_empirical_intelligent = [
    DomainSpec(
        name="experimental_psychology",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.INTELLIGENT,
        description="Controlled studies of human cognition and behavior",
        evidential_standards="Randomized experiments, statistical significance, replication.",
        characteristic_pitfalls=[
            "WEIRD samples (Western, Educated, Industrialized, Rich, Democratic)",
            "Demand characteristics",
            "Publication bias / replication crisis",
        ],
        example_propositions=[
            "This intervention improves memory",
            "The effect is statistically significant",
            "The finding replicates across populations",
        ],
    ),
    DomainSpec(
        name="developmental_psychology",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.INTELLIGENT,
        description="Infant cognition, child development, lifespan changes",
        evidential_standards="Longitudinal studies, looking-time paradigms, cross-sectional comparisons.",
        characteristic_pitfalls=[
            "Confusing competence with performance",
            "Cohort effects vs age effects",
            "Over-interpreting infant looking times",
        ],
        example_propositions=[
            "Infants understand object permanence by this age",
            "This ability is innate rather than learned",
            "Development proceeds through these stages",
        ],
    ),
    DomainSpec(
        name="sociology",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.INTELLIGENT,
        description="Social structures, institutions, collective behavior",
        evidential_standards="Surveys, ethnography, statistical analysis of social data.",
        characteristic_pitfalls=[
            "Ecological fallacy (group to individual)",
            "Confusing correlation with causation",
            "Reifying social constructs",
        ],
        example_propositions=[
            "This policy caused the social change",
            "The correlation reflects a causal relationship",
            "This social category is natural rather than constructed",
        ],
    ),
    DomainSpec(
        name="anthropology",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.INTELLIGENT,
        description="Cultural variation, human origins, ethnography",
        evidential_standards="Participant observation, comparative analysis, archaeological evidence.",
        characteristic_pitfalls=[
            "Ethnocentrism",
            "Over-generalizing from single cultures",
            "Imposing Western categories",
        ],
        example_propositions=[
            "This practice is universal across cultures",
            "The behavior is culturally determined",
            "This interpretation reflects the emic perspective",
        ],
    ),
    DomainSpec(
        name="economics",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.INTELLIGENT,
        description="Markets, incentives, macroeconomic phenomena",
        evidential_standards="Natural experiments, econometric analysis, model predictions.",
        characteristic_pitfalls=[
            "Assuming rationality",
            "Ceteris paribus violations",
            "Confusing models with reality",
        ],
        example_propositions=[
            "This policy will reduce inflation",
            "The market is efficient",
            "Incentives explain the behavior",
        ],
    ),
    DomainSpec(
        name="history",
        mode=FulfillmentMode.EMPIRICAL,
        empirical_level=EmpiricalLevel.INTELLIGENT,
        description="Past events, causation, interpretation of sources",
        evidential_standards="Primary sources, corroboration, contextual analysis.",
        characteristic_pitfalls=[
            "Presentism (judging past by present standards)",
            "Survivorship bias in sources",
            "Narrative fallacy",
        ],
        example_propositions=[
            "This event caused the war",
            "The source is reliable",
            "This interpretation best fits the evidence",
        ],
    ),
]

# ----- COMMON SENSE MODE -----

_common_sense_domains = [
    DomainSpec(
        name="everyday_practical",
        mode=FulfillmentMode.COMMON_SENSE,
        common_sense_variety=CommonSenseVariety.EVERYDAY,
        description="Mundane practical reasoning - cooking, navigation, household",
        evidential_standards="Direct experience, folk knowledge, trial and error.",
        characteristic_pitfalls=[
            "Overgeneralizing from limited experience",
            "Superstition and folk errors",
            "Confusing familiarity with understanding",
        ],
        example_propositions=[
            "Taking this route will be faster",
            "The food is safe to eat",
            "It will rain today based on the clouds",
        ],
    ),
    DomainSpec(
        name="social_common_sense",
        mode=FulfillmentMode.COMMON_SENSE,
        common_sense_variety=CommonSenseVariety.SOCIAL,
        description="Reading social situations, predicting behavior",
        evidential_standards="Social cues, past patterns, cultural knowledge.",
        characteristic_pitfalls=[
            "Fundamental attribution error",
            "Projection of own values",
            "Missing cultural context",
        ],
        example_propositions=[
            "She is angry based on her expression",
            "He will keep his promise",
            "They are being sincere",
        ],
    ),
    DomainSpec(
        name="professional_judgment",
        mode=FulfillmentMode.COMMON_SENSE,
        common_sense_variety=CommonSenseVariety.PROFESSIONAL,
        description="Domain expertise applied practically - tradecraft, methodically-informed common sense",
        evidential_standards="Professional experience, heuristics, tacit knowledge, protocols.",
        characteristic_pitfalls=[
            "Expert overconfidence",
            "Anchoring on initial assessment",
            "Ignoring base rates",
        ],
        example_propositions=[
            "This patient needs immediate intervention",
            "The structure is sound",
            "This code is production-ready",
        ],
    ),
    DomainSpec(
        name="generational_folk_wisdom",
        mode=FulfillmentMode.COMMON_SENSE,
        common_sense_variety=CommonSenseVariety.GENERATIONAL,
        description="Knowledge passed down - proverbs, traditional practices",
        evidential_standards="Tradition, anecdote, perceived ancestral authority.",
        characteristic_pitfalls=[
            "Survivorship bias (remembering hits, forgetting misses)",
            "Changed contexts invalidating old wisdom",
            "Contradictory proverbs",
        ],
        example_propositions=[
            "Feed a cold, starve a fever",
            "Red sky at night, sailor's delight",
            "Spare the rod, spoil the child",
        ],
    ),
    DomainSpec(
        name="subcultural_knowledge",
        mode=FulfillmentMode.COMMON_SENSE,
        common_sense_variety=CommonSenseVariety.SUBCULTURAL,
        description="In-group knowledge - professional, hobby, regional",
        evidential_standards="Community consensus, in-group experience, jargon.",
        characteristic_pitfalls=[
            "Assuming universal applicability",
            "Echo chamber effects",
            "Gatekeeping valid but unstated knowledge",
        ],
        example_propositions=[
            "This technique is correct form",
            "That's not how we do things here",
            "You need X years to understand this",
        ],
    ),
]

# ----- DIALECTICAL MODE -----

_dialectical_domains = [
    DomainSpec(
        name="philosophical",
        mode=FulfillmentMode.DIALECTICAL,
        description="Positions vs counterpositions, performative self-consistency",
        evidential_standards="Can this position be held coherently by one who grasps it intelligently and affirms it reasonably?",
        characteristic_pitfalls=[
            "Missing self-defeating implications",
            "Confusing rhetorical force with logical coherence",
            "Failing to identify counterpositions",
        ],
        example_propositions=[
            "Knowledge is merely neural activity",
            "There is no truth",
            "All knowledge is relative to culture",
        ],
    ),
    DomainSpec(
        name="pop_science",
        mode=FulfillmentMode.DIALECTICAL,  # It's a counterposition, not a parallel domain
        description="Common sense appropriating scientific vocabulary while refusing scientific detachment - a dialectical counterposition",
        evidential_standards="Detection: Are conditions actually fulfilled or merely claimed? Is there vocabulary without understanding?",
        characteristic_pitfalls=[
            "Vocabulary without correlative insight",
            "Authority appeal over evidence",
            "Conditions claimed but not fulfilled",
            "Form of scientific judgment without substance of verification",
        ],
        example_propositions=[
            "A study shows X correlates with Y, therefore X causes Y",
            "Scientists say...",
            "Research proves...",
        ],
    ),
]

# ----- ADVERSARIAL CONTEXT DOMAINS -----
# Adversarial is a CONTEXT (dimension), not a mode
# These use common_sense mode with adversarial epistemic context

_adversarial_context_domains = [
    DomainSpec(
        name="deliberate_deception",
        mode=FulfillmentMode.COMMON_SENSE,
        common_sense_variety=CommonSenseVariety.SOCIAL,
        description="Speaker actively trying to mislead - lies, fraud, scams",
        evidential_standards="Must detect inconsistencies, verify independently, question motives.",
        characteristic_pitfalls=["Assuming good faith", "Trusting surface coherence"],
        example_propositions=["The seller is being honest about the product"],
    ),
    DomainSpec(
        name="manipulation",
        mode=FulfillmentMode.COMMON_SENSE,
        common_sense_variety=CommonSenseVariety.SOCIAL,
        description="Emotional manipulation, gaslighting, strategic framing",
        evidential_standards="Track record, independent verification, emotional distance.",
        characteristic_pitfalls=["Emotional reasoning", "Doubting own perception"],
        example_propositions=["Their apology is sincere"],
    ),
    DomainSpec(
        name="propaganda",
        mode=FulfillmentMode.COMMON_SENSE,
        common_sense_variety=CommonSenseVariety.SOCIAL,
        description="Institutional/political messaging designed to persuade",
        evidential_standards="Source analysis, cui bono, cross-reference independent sources.",
        characteristic_pitfalls=[
            "Trusting official sources uncritically",
            "False balance",
        ],
        example_propositions=["This news report is accurate"],
    ),
    DomainSpec(
        name="sarcasm_irony",
        mode=FulfillmentMode.COMMON_SENSE,
        common_sense_variety=CommonSenseVariety.SOCIAL,
        description="Non-literal meaning, saying opposite of intent",
        evidential_standards="Context, tone, speaker's known views, pragmatic implicature.",
        characteristic_pitfalls=["Taking literally what's meant ironically"],
        example_propositions=["The speaker genuinely believes what they said"],
    ),
    DomainSpec(
        name="adversarial_debate",
        mode=FulfillmentMode.COMMON_SENSE,
        common_sense_variety=CommonSenseVariety.SOCIAL,
        description="Arguments designed to win rather than find truth",
        evidential_standards="Evaluate argument independent of rhetorical skill.",
        characteristic_pitfalls=["Confusing eloquence with correctness"],
        example_propositions=["The more confident speaker is correct"],
    ),
    DomainSpec(
        name="strategic_ambiguity",
        mode=FulfillmentMode.COMMON_SENSE,
        common_sense_variety=CommonSenseVariety.SOCIAL,
        description="Deliberately vague or deniable statements",
        evidential_standards="Pin down specific claims, reject false precision, note evasion.",
        characteristic_pitfalls=["Filling in ambiguity charitably"],
        example_propositions=["The politician committed to the policy"],
    ),
]

# ----- EDGE CASE DOMAINS -----
# Edge cases are complications that can arise in any mode
# Assign them to appropriate modes based on their nature

_edge_case_domains = [
    DomainSpec(
        name="self_reference",
        mode=FulfillmentMode.FORMAL,  # Self-reference is primarily a formal/logical issue
        description="Propositions about themselves - liar paradox territory",
        evidential_standards="Recognize when self-reference creates undecidability.",
        characteristic_pitfalls=["Treating all self-reference as paradoxical"],
        example_propositions=["This sentence is false"],
    ),
    DomainSpec(
        name="category_errors",
        mode=FulfillmentMode.DIALECTICAL,  # Category errors are philosophical/dialectical
        description="Questions that presuppose wrong categories",
        evidential_standards="Identify category mistakes before attempting judgment.",
        characteristic_pitfalls=["Accepting malformed questions"],
        example_propositions=["What color is jealousy?"],
    ),
    DomainSpec(
        name="vagueness_sorites",
        mode=FulfillmentMode.COMMON_SENSE,  # Vagueness is primarily a common sense issue
        common_sense_variety=CommonSenseVariety.EVERYDAY,
        description="Heap paradoxes, borderline cases, fuzzy boundaries",
        evidential_standards="Recognize vagueness rather than forcing binary judgment.",
        characteristic_pitfalls=["Demanding precision where none exists"],
        example_propositions=["This person is bald"],
    ),
    DomainSpec(
        name="counterfactuals",
        mode=FulfillmentMode.COMMON_SENSE,  # Counterfactuals are practical reasoning
        common_sense_variety=CommonSenseVariety.EVERYDAY,
        description="What would have happened if...",
        evidential_standards="Possible worlds reasoning, causal models, similarity metrics.",
        characteristic_pitfalls=[
            "Hindsight bias",
            "False certainty about alternatives",
        ],
        example_propositions=["If X hadn't happened, Y wouldn't have either"],
    ),
    DomainSpec(
        name="future_contingents",
        mode=FulfillmentMode.COMMON_SENSE,  # Future contingents are practical
        common_sense_variety=CommonSenseVariety.EVERYDAY,
        description="Propositions about undetermined future events",
        evidential_standards="Recognize genuine indeterminacy vs. epistemic uncertainty.",
        characteristic_pitfalls=["Treating all futures as determined"],
        example_propositions=["There will be a sea battle tomorrow"],
    ),
]

# Register all principled domains
for domain_list in [
    _formal_domains,
    _empirical_physical,
    _empirical_biological,
    _empirical_sensitive,
    _empirical_intelligent,
    _common_sense_domains,
    _dialectical_domains,
    _adversarial_context_domains,
    _edge_case_domains,
]:
    for domain in domain_list:
        PRINCIPLED_DOMAINS[domain.name] = domain


# ============================================================================
# EDGE CASE SPECIFICATIONS
# ============================================================================

EDGE_CASE_SPECS: Dict[EdgeCaseType, EdgeCaseSpec] = {
    EdgeCaseType.SELF_REFERENCE: EdgeCaseSpec(
        edge_type=EdgeCaseType.SELF_REFERENCE,
        description="Propositions that refer to themselves or their own affirmation",
        challenge="Recognize when self-reference creates undecidability or paradox",
        probability=0.05,
        compatible_modes=[FulfillmentMode.FORMAL, FulfillmentMode.DIALECTICAL],
        generation_prompt="Include a self-referential element where the proposition or claim refers to itself, potentially creating paradox or requiring meta-level analysis.",
    ),
    EdgeCaseType.VAGUENESS: EdgeCaseSpec(
        edge_type=EdgeCaseType.VAGUENESS,
        description="No sharp boundary for predicate application (sorites)",
        challenge="Recognize borderline cases rather than forcing binary judgment",
        probability=0.10,
        compatible_modes=[FulfillmentMode.COMMON_SENSE, FulfillmentMode.EMPIRICAL],
        generation_prompt="Include vague predicates with no sharp boundary (e.g., 'tall', 'heap', 'bald') where the judgment requires recognizing genuine borderline cases.",
    ),
    EdgeCaseType.FUTURE_CONTINGENT: EdgeCaseSpec(
        edge_type=EdgeCaseType.FUTURE_CONTINGENT,
        description="Event has not yet occurred - genuine indeterminacy",
        challenge="Distinguish genuine indeterminacy from epistemic uncertainty",
        probability=0.15,
        compatible_modes=[FulfillmentMode.COMMON_SENSE, FulfillmentMode.EMPIRICAL],
        generation_prompt="Include a proposition about a future event that has not yet occurred, where the judgment must recognize the temporal limitation on available evidence.",
    ),
    EdgeCaseType.COUNTERFACTUAL: EdgeCaseSpec(
        edge_type=EdgeCaseType.COUNTERFACTUAL,
        description="No actual data to revert to - 'what would have happened if...'",
        challenge="Reason about alternative possibilities without overclaiming certainty",
        probability=0.10,
        compatible_modes=[
            FulfillmentMode.COMMON_SENSE,
            FulfillmentMode.EMPIRICAL,
            FulfillmentMode.DIALECTICAL,
        ],
        generation_prompt="Include a counterfactual proposition ('if X had happened, then Y would have...') where no actual data exists to verify the claim.",
    ),
    EdgeCaseType.CATEGORY_ERROR: EdgeCaseSpec(
        edge_type=EdgeCaseType.CATEGORY_ERROR,
        description="Question presupposes wrong categories",
        challenge="Identify category mistakes before attempting judgment",
        probability=0.05,
        compatible_modes=list(FulfillmentMode),  # Can occur in any mode
        generation_prompt="Include a proposition or question that commits a category error - applying concepts to domains where they don't belong.",
    ),
}


# ============================================================================
# ADVERSARIAL CONTEXT SPECIFICATIONS
# ============================================================================


@dataclass
class AdversarialSpec:
    """Specification for an adversarial context type."""

    adversarial_type: AdversarialType
    description: str
    detection_challenge: str
    key_pitfall: str
    generation_prompt: str


ADVERSARIAL_SPECS: Dict[AdversarialType, AdversarialSpec] = {
    AdversarialType.DECEPTION: AdversarialSpec(
        adversarial_type=AdversarialType.DECEPTION,
        description="Speaker actively trying to mislead - lies, fraud, scams",
        detection_challenge="Detect inconsistencies, verify independently, question motives",
        key_pitfall="Assuming good faith",
        generation_prompt="Create a scenario where a speaker is actively deceiving - their expressed claims do not match their actual knowledge or intent.",
    ),
    AdversarialType.MANIPULATION: AdversarialSpec(
        adversarial_type=AdversarialType.MANIPULATION,
        description="Emotional manipulation, gaslighting, strategic framing",
        detection_challenge="Maintain emotional distance, verify independently, trust own perception",
        key_pitfall="Doubting own perception when manipulated",
        generation_prompt="Create a scenario involving emotional manipulation where the speaker uses psychological tactics rather than evidence.",
    ),
    AdversarialType.PROPAGANDA: AdversarialSpec(
        adversarial_type=AdversarialType.PROPAGANDA,
        description="Institutional/political messaging designed to persuade",
        detection_challenge="Source analysis, cui bono, cross-reference independent sources",
        key_pitfall="Trusting official sources uncritically",
        generation_prompt="Create a scenario involving institutional messaging where claims serve institutional interests and may not be independently verified.",
    ),
    AdversarialType.STRATEGIC_AMBIGUITY: AdversarialSpec(
        adversarial_type=AdversarialType.STRATEGIC_AMBIGUITY,
        description="Deliberately vague or deniable statements",
        detection_challenge="Pin down specific claims, reject false precision, note evasion",
        key_pitfall="Filling in ambiguity charitably",
        generation_prompt="Create a scenario where a speaker is deliberately vague, allowing multiple interpretations or plausible deniability.",
    ),
    AdversarialType.ADVERSARIAL_DEBATE: AdversarialSpec(
        adversarial_type=AdversarialType.ADVERSARIAL_DEBATE,
        description="Arguments designed to win rather than find truth",
        detection_challenge="Evaluate argument independent of rhetorical skill",
        key_pitfall="Confusing eloquence with correctness",
        generation_prompt="Create a debate scenario where at least one party is arguing to win rather than to discover truth.",
    ),
    AdversarialType.SARCASM_IRONY: AdversarialSpec(
        adversarial_type=AdversarialType.SARCASM_IRONY,
        description="Non-literal meaning - saying opposite of intent (cooperative intent, adversarial form)",
        detection_challenge="Recognize non-literal meaning from context and tone",
        key_pitfall="Taking literally what's meant ironically",
        generation_prompt="Create a scenario involving sarcasm or irony where the literal meaning differs from the intended meaning.",
    ),
}


# ============================================================================
# UNIFIED API: ALL_DOMAINS = PRINCIPLED_DOMAINS
# ============================================================================

# ALL_DOMAINS is now simply an alias to PRINCIPLED_DOMAINS
# All 35 domains (24 principled + 6 adversarial + 5 edge case) are DomainSpec objects
ALL_DOMAINS: Dict[str, DomainSpec] = PRINCIPLED_DOMAINS


# ============================================================================
# API FUNCTIONS
# ============================================================================


def get_all_domain_names() -> List[str]:
    """Get all domain names."""
    return list(ALL_DOMAINS.keys())


def get_principled_domain(name: str) -> DomainSpec:
    """Get a domain specification from the principled schema."""
    if name not in PRINCIPLED_DOMAINS:
        raise ValueError(
            f"Unknown principled domain: {name}. Available: {list(PRINCIPLED_DOMAINS.keys())}"
        )
    return PRINCIPLED_DOMAINS[name]


def get_domains_by_mode(mode: FulfillmentMode) -> List[DomainSpec]:
    """Get all domains with a specific fulfillment mode."""
    return [d for d in PRINCIPLED_DOMAINS.values() if d.mode == mode]


def get_domains_by_empirical_level(level: EmpiricalLevel) -> List[DomainSpec]:
    """Get all empirical domains at a specific integration level."""
    return [
        d
        for d in PRINCIPLED_DOMAINS.values()
        if d.mode == FulfillmentMode.EMPIRICAL and d.empirical_level == level
    ]


def get_domains_by_common_sense_variety(
    variety: CommonSenseVariety,
) -> List[DomainSpec]:
    """Get all common sense domains of a specific variety."""
    return [
        d
        for d in PRINCIPLED_DOMAINS.values()
        if d.mode == FulfillmentMode.COMMON_SENSE and d.common_sense_variety == variety
    ]


def get_mode_spec(mode: FulfillmentMode) -> ModeSpec:
    """Get the specification for a fulfillment mode."""
    return MODE_SPECS[mode]


def get_edge_case_spec(edge_type: EdgeCaseType) -> EdgeCaseSpec:
    """Get the specification for an edge case type."""
    return EDGE_CASE_SPECS[edge_type]


def get_adversarial_spec(adv_type: AdversarialType) -> AdversarialSpec:
    """Get the specification for an adversarial context type."""
    return ADVERSARIAL_SPECS[adv_type]


def roll_edge_case(
    mode: FulfillmentMode, rng: Optional[random.Random] = None
) -> Optional[EdgeCaseType]:
    """
    Probabilistically roll for an edge case complication.

    Returns an EdgeCaseType if the roll succeeds, None otherwise.
    Only returns edge cases compatible with the given mode.
    """
    if rng is None:
        rng = random.Random()

    compatible = [
        (etype, spec)
        for etype, spec in EDGE_CASE_SPECS.items()
        if mode in spec.compatible_modes
    ]

    for etype, spec in compatible:
        if rng.random() < spec.probability:
            return etype

    return None


def get_principled_prompt_context(name: str) -> str:
    """Get comprehensive prompt context for a principled domain."""
    spec = get_principled_domain(name)
    mode_spec = MODE_SPECS[spec.mode]

    pitfalls = "\n".join(f"- {p}" for p in spec.characteristic_pitfalls)
    mode_pitfalls = "\n".join(f"- {p}" for p in mode_spec.characteristic_pitfalls)

    context = f"""Domain: {spec.name}
Mode of Fulfillment: {spec.mode.value}
{f"Empirical Level: {spec.empirical_level.value}" if spec.empirical_level else ""}
{f"Common Sense Variety: {spec.common_sense_variety.value}" if spec.common_sense_variety else ""}

Description: {spec.description}

MODE CHARACTERISTICS:
{mode_spec.description}
Fulfillment Location: {mode_spec.fulfillment_location}
Reversion Required: {"Yes" if mode_spec.reversion_required else "No"}

EVIDENTIAL STANDARDS:
{spec.evidential_standards}

MODE-LEVEL PITFALLS:
{mode_pitfalls}

DOMAIN-SPECIFIC PITFALLS:
{pitfalls}"""

    return context.strip()


# ============================================================================
# PART 4: SUMMARY AND CLI
# ============================================================================


def get_schema_summary() -> Dict:
    """Get a summary of the principled schema."""
    return {
        "modes": {
            mode.value: {
                "description": spec.description,
                "domain_count": len(get_domains_by_mode(mode)),
                "domains": [d.name for d in get_domains_by_mode(mode)],
            }
            for mode, spec in MODE_SPECS.items()
        },
        "empirical_levels": {
            level.value: {
                "domain_count": len(get_domains_by_empirical_level(level)),
                "domains": [d.name for d in get_domains_by_empirical_level(level)],
            }
            for level in EmpiricalLevel
        },
        "common_sense_varieties": {
            variety.value: {
                "domain_count": len(get_domains_by_common_sense_variety(variety)),
                "domains": [
                    d.name for d in get_domains_by_common_sense_variety(variety)
                ],
            }
            for variety in CommonSenseVariety
        },
        "edge_cases": {
            etype.value: {
                "probability": spec.probability,
                "compatible_modes": [m.value for m in spec.compatible_modes],
            }
            for etype, spec in EDGE_CASE_SPECS.items()
        },
        "adversarial_types": [t.value for t in AdversarialType],
        "total_principled_domains": len(PRINCIPLED_DOMAINS),
        "total_domains": len(ALL_DOMAINS),
    }


def print_schema_summary():
    """Print a summary of the principled schema."""
    summary = get_schema_summary()

    print("=" * 60)
    print("PRINCIPLED DOMAIN SCHEMA")
    print("=" * 60)

    print("\nMODES OF FULFILLMENT:")
    for mode, info in summary["modes"].items():
        print(f"\n  {mode.upper()} ({info['domain_count']} domains)")
        print(f"    {info['description'][:70]}...")
        for d in info["domains"]:
            print(f"      - {d}")

    print("\n" + "-" * 60)
    print("EMPIRICAL LEVELS (higher integration):")
    for level, info in summary["empirical_levels"].items():
        print(f"  {level}: {info['domains']}")

    print("\n" + "-" * 60)
    print("COMMON SENSE VARIETIES:")
    for variety, info in summary["common_sense_varieties"].items():
        print(f"  {variety}: {info['domains']}")

    print("\n" + "-" * 60)
    print("EDGE CASE MODIFIERS (probabilistic):")
    for etype, info in summary["edge_cases"].items():
        print(
            f"  {etype}: p={info['probability']:.0%}, modes={info['compatible_modes']}"
        )

    print("\n" + "-" * 60)
    print("ADVERSARIAL CONTEXT TYPES:")
    for atype in summary["adversarial_types"]:
        print(f"  - {atype}")

    print("\n" + "=" * 60)
    print(f"Total principled domains: {summary['total_principled_domains']}")
    print(f"Total domains (all): {summary['total_domains']}")


if __name__ == "__main__":
    print_schema_summary()
