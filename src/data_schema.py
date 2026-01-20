"""
Training data schemas for cognitive operation fine-tuning.

Each cognitive level requires training data with specific characteristics
that embody the operational semantics of that level.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
from pathlib import Path

from .cognitive_levels import CognitiveLevel, get_level


@dataclass
class AttentionSample:
    """
    Training sample for Level 1: Attention

    Input: Raw, unstructured data
    Output: Structured presentation of the same data (no interpretation)

    Success criterion: Data integrity preserved
    """

    raw_input: str                    # Unstructured data
    structured_output: str            # Same data, organized and clear
    data_type: str                    # Type of data (text, numbers, mixed, etc.)
    integrity_notes: Optional[str] = None  # Notes on what must be preserved


@dataclass
class UnderstandingSample:
    """
    Training sample for Level 2: Understanding

    Input: Structured data (from Attention)
    Output: Propositions/hypotheses explaining the data (NOT affirmed as true)

    Success criterion: Coherent insight that unifies the data
    """

    structured_input: str             # Data from Attention level
    question: str                     # "What is it?" or "Why is it so?"
    proposition: str                  # The hypothesis/insight generated
    reasoning: str                    # How the insight unifies the data
    alternatives: List[str] = field(default_factory=list)  # Other possible hypotheses
    confidence_note: str = "This is a hypothesis, not yet affirmed as true."


@dataclass
class JudgmentSample:
    """
    Training sample for Level 3: Judgment

    Input: A proposition (from Understanding) + Evidence (from Attention)
    Output: Affirmation (Yes), Denial (No), or Withheld judgment

    Success criterion: Correctly grasps the virtually unconditioned

    The virtually unconditioned has three components:
    1. The conditioned (the proposition to be judged)
    2. The link (the conditions that would make it true)
    3. The fulfillment (whether conditions are met in evidence)
    """

    proposition: str                  # The conditioned - what is to be judged
    evidence: str                     # Data bearing on the proposition
    conditions: List[str]             # What must be true for proposition to be true
    conditions_fulfilled: Dict[str, bool]  # Which conditions are met
    judgment: str                     # "Yes", "No", or "Insufficient evidence"
    judgment_reasoning: str           # How the virtually unconditioned was grasped
    is_virtually_unconditioned: bool  # Whether conditions suffice for judgment


@dataclass
class DecisionSample:
    """
    Training sample for Level 4: Decision

    Input: Judgments of fact + Values at stake
    Output: Judgment of value and commitment (or withholding)

    Success criterion: Authenticity - consistency with being attentive,
                      intelligent, reasonable, responsible
    """

    facts: List[str]                  # Relevant judgments of fact
    values_at_stake: List[str]        # What values are involved
    options: List[str]                # Possible courses of action
    evaluation: str                   # Assessment of options against values
    decision: str                     # The commitment made (or withheld)
    authenticity_check: str           # How this aligns with transcendental precepts
    responsibility_note: str          # Acknowledgment of responsibility for decision


@dataclass
class CognitivePipelineSample:
    """
    A complete sample showing all four levels processing the same situation.

    This demonstrates:
    1. How the levels build on each other
    2. How semantic meaning shifts across levels
    3. The cumulative nature of cognitional process
    """

    situation: str                    # The initial situation/data

    # Level 1
    attention: AttentionSample

    # Level 2
    understanding: UnderstandingSample

    # Level 3
    judgment: JudgmentSample

    # Level 4
    decision: Optional[DecisionSample] = None  # May not always reach this level


def sample_to_training_format(
    sample: Any,
    level: CognitiveLevel
) -> Dict[str, str]:
    """
    Convert a sample to the training format expected by the model.

    Returns a dict with 'input' and 'output' fields for supervised fine-tuning.
    """

    level_spec = get_level(level)

    if level == CognitiveLevel.ATTENTION:
        return {
            "system": level_spec.system_prompt,
            "input": sample.raw_input,
            "output": sample.structured_output,
        }

    elif level == CognitiveLevel.UNDERSTANDING:
        return {
            "system": level_spec.system_prompt,
            "input": f"Data: {sample.structured_input}\n\nQuestion: {sample.question}",
            "output": f"Proposition: {sample.proposition}\n\nReasoning: {sample.reasoning}\n\nNote: {sample.confidence_note}",
        }

    elif level == CognitiveLevel.JUDGMENT:
        conditions_text = "\n".join(f"- {c}" for c in sample.conditions)
        fulfillment_text = "\n".join(
            f"- {c}: {'Fulfilled' if f else 'Not fulfilled'}"
            for c, f in sample.conditions_fulfilled.items()
        )
        return {
            "system": level_spec.system_prompt,
            "input": f"Proposition: {sample.proposition}\n\nEvidence: {sample.evidence}\n\nConditions for truth:\n{conditions_text}",
            "output": f"Condition fulfillment:\n{fulfillment_text}\n\nJudgment: {sample.judgment}\n\nReasoning: {sample.judgment_reasoning}",
        }

    elif level == CognitiveLevel.DECISION:
        facts_text = "\n".join(f"- {f}" for f in sample.facts)
        values_text = "\n".join(f"- {v}" for v in sample.values_at_stake)
        options_text = "\n".join(f"- {o}" for o in sample.options)
        return {
            "system": level_spec.system_prompt,
            "input": f"Facts:\n{facts_text}\n\nValues at stake:\n{values_text}\n\nOptions:\n{options_text}",
            "output": f"Evaluation: {sample.evaluation}\n\nDecision: {sample.decision}\n\nAuthenticity: {sample.authenticity_check}\n\nResponsibility: {sample.responsibility_note}",
        }

    raise ValueError(f"Unknown level: {level}")


def format_for_chat(training_sample: Dict[str, str]) -> List[Dict[str, str]]:
    """Convert training sample to chat format for instruction tuning."""

    return [
        {"role": "system", "content": training_sample["system"]},
        {"role": "user", "content": training_sample["input"]},
        {"role": "assistant", "content": training_sample["output"]},
    ]


def save_dataset(
    samples: List[Any],
    level: CognitiveLevel,
    output_path: Path,
    format: str = "jsonl"
) -> None:
    """Save samples to a training dataset file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in samples:
            training_format = sample_to_training_format(sample, level)
            chat_format = format_for_chat(training_format)
            f.write(json.dumps({"messages": chat_format}) + "\n")

    print(f"Saved {len(samples)} samples to {output_path}")
