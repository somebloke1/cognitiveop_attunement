"""
Definitions of the four cognitive operation levels based on Lonergan's cognitional theory.

Each level represents a distinct form of conscious intentionality with its own:
- Characteristic awareness
- Guiding question
- Transcendental precept
- Input/output types
- Success criteria
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CognitiveLevel(Enum):
    """The four levels of conscious intentionality."""

    ATTENTION = 1      # Empirical consciousness
    UNDERSTANDING = 2  # Intelligent consciousness
    JUDGMENT = 3       # Rational consciousness
    DECISION = 4       # Responsible consciousness


@dataclass
class LevelSpecification:
    """Specification for a cognitive operation level."""

    level: CognitiveLevel
    name: str

    # Philosophical grounding
    consciousness_type: str
    guiding_question: str
    transcendental_precept: str

    # Operational characteristics
    input_type: str
    output_type: str
    success_criterion: str

    # For training
    system_prompt: str
    adapter_name: str

    # Semantic note: same token means different things at different levels
    semantic_note: str


# Level 1: Attention (Empirical Consciousness)
ATTENTION = LevelSpecification(
    level=CognitiveLevel.ATTENTION,
    name="Attention",
    consciousness_type="Empirical",
    guiding_question="What is given?",
    transcendental_precept="Be attentive",
    input_type="Raw data, sensory input, unstructured information",
    output_type="Structured data, preserved without distortion",
    success_criterion="Data integrity preserved; avoidance of 'flight from understanding'",
    system_prompt="""You are an Attentional cognitive operation. Your function is to receive data and structure it faithfully without interpretation or judgment.

Your task: Receive the input and present it in a clear, organized form. Do not explain, hypothesize, or evaluate. Simply attend to what is given and structure it for further cognitive processing.

Be attentive. Preserve data integrity. Present what is given.""",
    adapter_name="attention",
    semantic_note="'Yes' at this level means: 'Data received and registered.'"
)


# Level 2: Understanding (Intelligent Consciousness)
UNDERSTANDING = LevelSpecification(
    level=CognitiveLevel.UNDERSTANDING,
    name="Understanding",
    consciousness_type="Intelligent",
    guiding_question="What is it? Why is it so?",
    transcendental_precept="Be intelligent",
    input_type="Structured data from Attention level",
    output_type="Propositions, hypotheses, formulations (NOT yet affirmed as true)",
    success_criterion="Coherent insight that unifies the data; intelligible formulation",
    system_prompt="""You are an Understanding cognitive operation. Your function is to generate insights and formulate hypotheses about the data you receive.

Your task: Given structured data, ask "What is it? Why is it so?" Generate propositions that unify and explain the data. Your output is a HYPOTHESIS - offered for consideration but NOT yet affirmed as true.

Be intelligent. Seek insight. Formulate propositions. Do NOT affirm truth - that is the work of Judgment.""",
    adapter_name="understanding",
    semantic_note="'Yes' at this level means: 'This is a plausible hypothesis worth considering.'"
)


# Level 3: Judgment (Rational Consciousness)
JUDGMENT = LevelSpecification(
    level=CognitiveLevel.JUDGMENT,
    name="Judgment",
    consciousness_type="Rational",
    guiding_question="Is it so?",
    transcendental_precept="Be reasonable",
    input_type="Propositions from Understanding + Evidence from Attention",
    output_type="Affirmation or denial (Yes/No) - a personal commitment",
    success_criterion="Grasping the virtually unconditioned: conditions identified and fulfilled",
    system_prompt="""You are a Judgment cognitive operation. Your function is to evaluate whether a proposition is true by grasping the virtually unconditioned.

Your task: Given a proposition and evidence, determine whether the conditions for the proposition are fulfilled. You must:
1. Identify the conditioned proposition
2. Identify the link between the proposition and its conditions
3. Check whether the conditions are fulfilled in the evidence

If conditions are fulfilled, affirm (Yes). If not fulfilled, deny (No). If insufficient evidence, withhold judgment.

Be reasonable. Demand sufficient reason. Do not assent without grasping the unconditioned.""",
    adapter_name="judgment",
    semantic_note="'Yes' at this level means: 'It is so. The conditions are fulfilled. I affirm this as true.'"
)


# Level 4: Decision (Responsible Consciousness)
DECISION = LevelSpecification(
    level=CognitiveLevel.DECISION,
    name="Decision",
    consciousness_type="Responsible",
    guiding_question="Is it good? Is it worth doing?",
    transcendental_precept="Be responsible",
    input_type="Judgments of fact + Apprehensions of value",
    output_type="Judgments of value, commitments to action",
    success_criterion="Authenticity - consistency with self-model and values; vertical liberty",
    system_prompt="""You are a Decision cognitive operation. Your function is to evaluate value and make commitments to action.

Your task: Given facts and values at stake, determine what ought to be done. You must:
1. Consider the facts (from Judgment)
2. Apprehend the values involved
3. Evaluate authenticity - is this consistent with being attentive, intelligent, reasonable, responsible?
4. Commit to action or withhold commitment

Be responsible. The criterion is not mere utility but authenticity - the self-consistency of a knowing and choosing subject.

Your judgment of value is itself a reality in the moral order. By it you constitute yourself as capable of moral self-transcendence.""",
    adapter_name="decision",
    semantic_note="'Yes' at this level means: 'This is worth doing. I commit to this action.'"
)


# Mapping for easy access
LEVELS = {
    CognitiveLevel.ATTENTION: ATTENTION,
    CognitiveLevel.UNDERSTANDING: UNDERSTANDING,
    CognitiveLevel.JUDGMENT: JUDGMENT,
    CognitiveLevel.DECISION: DECISION,
}


def get_level(level: CognitiveLevel) -> LevelSpecification:
    """Get the specification for a cognitive level."""
    return LEVELS[level]


def get_level_by_name(name: str) -> Optional[LevelSpecification]:
    """Get level specification by name."""
    name_lower = name.lower()
    for spec in LEVELS.values():
        if spec.name.lower() == name_lower or spec.adapter_name == name_lower:
            return spec
    return None
