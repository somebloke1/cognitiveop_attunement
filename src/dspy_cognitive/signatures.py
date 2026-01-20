"""
DSPy Signatures for Cognitive Operations

Each signature defines the typed interface for a cognitive level:
- Input fields: What the operation receives
- Output fields: What the operation produces (with constraints)

The Literal type hints enforce discrete outputs, preventing the model
from hedging with probabilistic language.
"""

from typing import List, Literal
import dspy


class AttentionSignature(dspy.Signature):
    """
    Level 1: Empirical Consciousness - Present data for inquiry.

    Precept: BE ATTENTIVE

    The attention operation selects and presents data as material for
    understanding. It does not interpret - it presents what is given.
    """

    raw_input: str = dspy.InputField(
        desc="Raw input data, text, or situation description"
    )
    domain: Literal["mathematical", "empirical", "common_sense", "pop_science", "philosophic"] = dspy.InputField(
        desc="Domain of inquiry"
    )

    presented_data: str = dspy.OutputField(
        desc="Data selected and presented for inquiry - what is given, not interpreted"
    )
    salient_features: List[str] = dspy.OutputField(
        desc="Key features attended to - patterns, elements, structures noticed"
    )
    image_for_insight: str = dspy.OutputField(
        desc="The 'phantasm' - concrete image/representation that insight will grasp"
    )


class UnderstandingSignature(dspy.Signature):
    """
    Level 2: Intelligent Consciousness - Grasp intelligibility through insight.

    Precept: BE INTELLIGENT

    The understanding operation inquires into the image and grasps the
    intelligible unity. Output is a HYPOTHESIS, not yet affirmed as true.
    """

    image: str = dspy.InputField(
        desc="The image/phantasm from attention - concrete representation to understand"
    )
    domain: Literal["mathematical", "empirical", "common_sense", "pop_science", "philosophic"] = dspy.InputField(
        desc="Domain of inquiry"
    )
    question: str = dspy.InputField(
        desc="The question driving inquiry: What is it? Why? How?"
    )

    insight: str = dspy.OutputField(
        desc="The act of understanding - grasping the intelligible unity in the image"
    )
    formulation: str = dspy.OutputField(
        desc="Conceptual formulation of the insight - the hypothesis"
    )
    conditions_for_truth: List[str] = dspy.OutputField(
        desc="What conditions would need to be fulfilled for this to be true"
    )


class JudgmentSignature(dspy.Signature):
    """
    Level 3: Rational Consciousness - Grasp the virtually unconditioned.

    Precept: BE REASONABLE

    The judgment operation evaluates whether conditions are fulfilled.
    Output is a DISCRETE COMMITMENT: Yes, No, or Insufficient evidence.

    The virtually unconditioned: a conditioned whose conditions are fulfilled.
    """

    proposition: str = dspy.InputField(
        desc="The conditioned - proposition to be judged"
    )
    evidence: str = dspy.InputField(
        desc="Available evidence bearing on the proposition"
    )
    conditions: List[str] = dspy.InputField(
        desc="Conditions that must be fulfilled for the proposition to be true"
    )
    domain: Literal["mathematical", "empirical", "common_sense", "pop_science", "philosophic"] = dspy.InputField(
        desc="Domain determines HOW conditions are fulfilled"
    )

    condition_fulfillment: List[Literal["Fulfilled", "Not fulfilled"]] = dspy.OutputField(
        desc="For each condition: is it fulfilled in the evidence? DISCRETE answer."
    )
    judgment: Literal["Yes", "No", "Insufficient"] = dspy.OutputField(
        desc="DISCRETE judgment: Yes (affirm), No (deny), Insufficient (withhold)"
    )
    reasoning: str = dspy.OutputField(
        desc="How the virtually unconditioned is or is not grasped"
    )


class DecisionSignature(dspy.Signature):
    """
    Level 4: Responsible Consciousness - Deliberate and commit to action.

    Precept: BE RESPONSIBLE

    The decision operation evaluates what is good and commits to action
    based on judgments of value, not just judgments of fact.
    """

    situation: str = dspy.InputField(
        desc="The situation requiring decision"
    )
    judgments: List[str] = dspy.InputField(
        desc="Relevant judgments of fact established at Level 3"
    )
    values_at_stake: List[str] = dspy.InputField(
        desc="Values relevant to this decision"
    )
    options: List[str] = dspy.InputField(
        desc="Available courses of action"
    )

    deliberation: str = dspy.OutputField(
        desc="Weighing of options against values and judgments"
    )
    value_judgment: str = dspy.OutputField(
        desc="Judgment of value: what is truly good here, not just apparently good"
    )
    decision: str = dspy.OutputField(
        desc="Commitment to action - what will be done"
    )
    responsibility: str = dspy.OutputField(
        desc="Acknowledgment of responsibility for this decision"
    )


# ============ COUNTERPOSITION DETECTION SIGNATURE ============

class CounterpositionDetectionSignature(dspy.Signature):
    """
    Specialized signature for detecting counterpositions.

    A counterposition is self-defeating: affirming it reasonably
    presupposes what it denies.
    """

    claim: str = dspy.InputField(
        desc="The claim to evaluate for self-consistency"
    )

    is_counterposition: Literal["Yes", "No", "Unclear"] = dspy.OutputField(
        desc="Does affirming this claim contradict the conditions of affirmation?"
    )
    self_reference_analysis: str = dspy.OutputField(
        desc="Analysis of what the claim presupposes vs. what it asserts"
    )
    reversal_or_development: Literal["Invites reversal", "Invites development", "Neither"] = dspy.OutputField(
        desc="Counterpositions invite reversal; positions invite development"
    )
