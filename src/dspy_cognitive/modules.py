"""
DSPy Modules for Cognitive Operations

Each module wraps a signature with optional chain-of-thought reasoning
and can be composed into a CognitiveProgram.
"""

from typing import Optional, List
import dspy

from .signatures import (
    AttentionSignature,
    UnderstandingSignature,
    JudgmentSignature,
    DecisionSignature,
    CounterpositionDetectionSignature,
)


class AttentionModule(dspy.Module):
    """
    Level 1: Attention - Present data for inquiry.

    This module takes raw input and presents it as data for understanding.
    It selects salient features and forms the image for insight.
    """

    def __init__(self):
        super().__init__()
        self.attend = dspy.ChainOfThought(AttentionSignature)

    def forward(self, raw_input: str, domain: str):
        return self.attend(raw_input=raw_input, domain=domain)


class UnderstandingModule(dspy.Module):
    """
    Level 2: Understanding - Grasp intelligibility through insight.

    This module inquires into the image and formulates hypotheses.
    Output is proposals for consideration, NOT yet affirmed.
    """

    def __init__(self):
        super().__init__()
        self.understand = dspy.ChainOfThought(UnderstandingSignature)

    def forward(self, image: str, domain: str, question: str):
        return self.understand(image=image, domain=domain, question=question)


class JudgmentModule(dspy.Module):
    """
    Level 3: Judgment - Grasp the virtually unconditioned.

    This module evaluates whether conditions are fulfilled and makes
    a DISCRETE commitment: Yes, No, or Insufficient.

    This is the critical operation where models typically fail by
    hedging instead of committing.
    """

    def __init__(self, use_chain_of_thought: bool = True):
        super().__init__()
        if use_chain_of_thought:
            self.judge = dspy.ChainOfThought(JudgmentSignature)
        else:
            self.judge = dspy.Predict(JudgmentSignature)

    def forward(
        self,
        proposition: str,
        evidence: str,
        conditions: List[str],
        domain: str,
    ):
        return self.judge(
            proposition=proposition,
            evidence=evidence,
            conditions=conditions,
            domain=domain,
        )


class DecisionModule(dspy.Module):
    """
    Level 4: Decision - Deliberate and commit to action.

    This module weighs options against values and commits to action.
    """

    def __init__(self):
        super().__init__()
        self.decide = dspy.ChainOfThought(DecisionSignature)

    def forward(
        self,
        situation: str,
        judgments: List[str],
        values_at_stake: List[str],
        options: List[str],
    ):
        return self.decide(
            situation=situation,
            judgments=judgments,
            values_at_stake=values_at_stake,
            options=options,
        )


class CounterpositionDetector(dspy.Module):
    """
    Specialized module for detecting self-defeating claims.

    Used in Tier 2 verification for philosophic domain judgments.
    """

    def __init__(self):
        super().__init__()
        self.detect = dspy.ChainOfThought(CounterpositionDetectionSignature)

    def forward(self, claim: str):
        return self.detect(claim=claim)


class CognitiveProgram(dspy.Module):
    """
    Full cognitive program composing all four levels.

    This is the complete pattern:
    Attention → Understanding → Judgment → Decision

    With potential feedback loops (handled by LangGraph at inference time,
    but the modules themselves are trained here).
    """

    def __init__(self):
        super().__init__()
        self.attention = AttentionModule()
        self.understanding = UnderstandingModule()
        self.judgment = JudgmentModule()
        self.decision = DecisionModule()

    def forward(
        self,
        raw_input: str,
        domain: str,
        question: str,
        values_at_stake: Optional[List[str]] = None,
        options: Optional[List[str]] = None,
    ):
        """
        Run the full cognitive pattern.

        For training, we typically train each level separately.
        This method shows how they compose at inference.
        """
        # Level 1: Attention
        attention_result = self.attention(raw_input=raw_input, domain=domain)

        # Level 2: Understanding
        understanding_result = self.understanding(
            image=attention_result.image_for_insight,
            domain=domain,
            question=question,
        )

        # Level 3: Judgment
        judgment_result = self.judgment(
            proposition=understanding_result.formulation,
            evidence=attention_result.presented_data,
            conditions=understanding_result.conditions_for_truth,
            domain=domain,
        )

        # Level 4: Decision (if values and options provided)
        decision_result = None
        if values_at_stake and options:
            decision_result = self.decision(
                situation=raw_input,
                judgments=[f"{understanding_result.formulation}: {judgment_result.judgment}"],
                values_at_stake=values_at_stake,
                options=options,
            )

        return dspy.Prediction(
            attention=attention_result,
            understanding=understanding_result,
            judgment=judgment_result,
            decision=decision_result,
        )


# ============ JUDGMENT-FOCUSED PROGRAM ============

class JudgmentProgram(dspy.Module):
    """
    Judgment-focused program for training the critical Level 3 operation.

    This is what we'll train with GRPO to develop correct judgment.
    """

    def __init__(self):
        super().__init__()
        self.judge = JudgmentModule(use_chain_of_thought=True)
        self.counterposition_detector = CounterpositionDetector()

    def forward(
        self,
        proposition: str,
        evidence: str,
        conditions: List[str],
        domain: str,
    ):
        # Main judgment
        result = self.judge(
            proposition=proposition,
            evidence=evidence,
            conditions=conditions,
            domain=domain,
        )

        # For philosophic domain, also run counterposition detection
        counterposition_result = None
        if domain == "philosophic":
            counterposition_result = self.counterposition_detector(claim=proposition)

        return dspy.Prediction(
            condition_fulfillment=result.condition_fulfillment,
            judgment=result.judgment,
            reasoning=result.reasoning,
            counterposition_analysis=counterposition_result,
        )
