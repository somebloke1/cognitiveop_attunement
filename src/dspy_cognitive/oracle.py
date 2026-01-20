"""
Oracle Judge Module

Uses a stronger model (Gemini Pro, Claude, GPT-4) to evaluate
whether a judgment is correct. This is Tier 3 verification.

The oracle provides the training signal for difficult cases where
deterministic and heuristic checks are insufficient.
"""

from typing import Optional, Literal
import dspy


class OracleJudgmentSignature(dspy.Signature):
    """
    Signature for oracle evaluation of a judgment.

    The oracle receives the original task and the model's judgment,
    then evaluates correctness.
    """

    # Original task
    proposition: str = dspy.InputField(desc="The proposition that was judged")
    evidence: str = dspy.InputField(desc="The evidence provided")
    conditions: str = dspy.InputField(desc="The conditions for truth")
    domain: str = dspy.InputField(desc="The domain of inquiry")

    # Model's judgment
    model_condition_fulfillment: str = dspy.InputField(
        desc="The model's assessment of condition fulfillment"
    )
    model_judgment: str = dspy.InputField(
        desc="The model's judgment (Yes/No/Insufficient)"
    )
    model_reasoning: str = dspy.InputField(
        desc="The model's reasoning for the judgment"
    )

    # Oracle evaluation
    is_correct: Literal["Correct", "Partially correct", "Incorrect"] = dspy.OutputField(
        desc="Is the model's judgment correct given the evidence and conditions?"
    )
    correctness_score: float = dspy.OutputField(
        desc="Score from 0.0 to 1.0 for judgment correctness"
    )
    evaluation_reasoning: str = dspy.OutputField(
        desc="Explanation of why the judgment is or isn't correct"
    )
    correct_judgment: Optional[str] = dspy.OutputField(
        desc="What the correct judgment should be (if model was wrong)"
    )


class OracleJudge(dspy.Module):
    """
    Oracle judge that evaluates judgment correctness.

    Uses a stronger model to provide training signal for GRPO.
    """

    def __init__(self, model: str = "gemini/gemini-2.0-flash"):
        super().__init__()
        self.model = model
        # Create a separate LM configuration for the oracle
        self.oracle_lm = dspy.LM(model)
        self.evaluate_judgment = dspy.Predict(OracleJudgmentSignature)

    def forward(
        self,
        proposition: str,
        evidence: str,
        conditions: str,
        domain: str,
        model_condition_fulfillment: str,
        model_judgment: str,
        model_reasoning: str,
    ):
        """Run oracle evaluation."""
        with dspy.context(lm=self.oracle_lm):
            return self.evaluate_judgment(
                proposition=proposition,
                evidence=evidence,
                conditions=conditions,
                domain=domain,
                model_condition_fulfillment=model_condition_fulfillment,
                model_judgment=model_judgment,
                model_reasoning=model_reasoning,
            )

    def evaluate(self, gold, pred) -> tuple[float, str]:
        """
        Evaluate a prediction against gold standard.

        Returns (score, explanation) for use in metrics.
        """
        try:
            # Format conditions as string
            conditions_str = (
                str(gold.conditions) if hasattr(gold, 'conditions')
                else "Not specified"
            )

            # Format model's condition fulfillment
            fulfillment_str = (
                str(pred.condition_fulfillment) if hasattr(pred, 'condition_fulfillment')
                else "Not provided"
            )

            result = self.forward(
                proposition=gold.proposition,
                evidence=gold.evidence,
                conditions=conditions_str,
                domain=getattr(gold, 'domain', 'unknown'),
                model_condition_fulfillment=fulfillment_str,
                model_judgment=pred.judgment,
                model_reasoning=getattr(pred, 'reasoning', 'No reasoning provided'),
            )

            # Extract score
            score = float(result.correctness_score)
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            explanation = f"{result.is_correct}: {result.evaluation_reasoning}"

            return score, explanation

        except Exception as e:
            # If oracle fails, return neutral score
            return 0.5, f"Oracle evaluation failed: {str(e)}"


# ============ SPECIALIZED ORACLE FOR MATHEMATICAL DOMAIN ============

class MathematicalVerifierSignature(dspy.Signature):
    """
    Specialized oracle for mathematical judgment verification.

    Mathematical judgments have deterministic correctness - proofs are valid or not.
    """

    proposition: str = dspy.InputField(desc="Mathematical proposition")
    proof_or_evidence: str = dspy.InputField(desc="Proof or evidence provided")
    model_judgment: str = dspy.InputField(desc="Model's Yes/No/Insufficient judgment")

    proof_is_valid: Literal["Valid", "Invalid", "Incomplete"] = dspy.OutputField(
        desc="Is the proof logically valid?"
    )
    proposition_is_true: Literal["True", "False", "Undetermined"] = dspy.OutputField(
        desc="Is the proposition mathematically true?"
    )
    judgment_is_correct: bool = dspy.OutputField(
        desc="Did the model make the correct judgment?"
    )
    explanation: str = dspy.OutputField(
        desc="Explanation of the mathematical verification"
    )


class MathematicalOracle(dspy.Module):
    """
    Specialized oracle for mathematical domain.

    Could potentially use a formal verification system in addition
    to LLM judgment for higher reliability.
    """

    def __init__(self, model: str = "gemini/gemini-2.0-flash"):
        super().__init__()
        self.oracle_lm = dspy.LM(model)
        self.verify = dspy.Predict(MathematicalVerifierSignature)

    def forward(self, proposition: str, proof_or_evidence: str, model_judgment: str):
        with dspy.context(lm=self.oracle_lm):
            return self.verify(
                proposition=proposition,
                proof_or_evidence=proof_or_evidence,
                model_judgment=model_judgment,
            )

    def evaluate(self, gold, pred) -> tuple[float, str]:
        """Evaluate mathematical judgment."""
        try:
            result = self.forward(
                proposition=gold.proposition,
                proof_or_evidence=gold.evidence,
                model_judgment=pred.judgment,
            )

            score = 1.0 if result.judgment_is_correct else 0.0
            return score, result.explanation

        except Exception as e:
            return 0.5, f"Mathematical oracle failed: {str(e)}"


# ============ DOMAIN-AWARE ORACLE ROUTER ============

class DomainAwareOracle:
    """
    Routes to specialized oracle based on domain.

    - Mathematical: MathematicalOracle (could add formal verification)
    - Philosophic: OracleJudge with counterposition awareness
    - Empirical/Common Sense/Pop Science: General OracleJudge
    """

    def __init__(self, model: str = "gemini/gemini-2.0-flash"):
        self.general_oracle = OracleJudge(model=model)
        self.math_oracle = MathematicalOracle(model=model)

    def evaluate(self, gold, pred) -> tuple[float, str]:
        """Route to appropriate oracle based on domain."""
        domain = getattr(gold, 'domain', 'unknown')

        if domain == "mathematical":
            return self.math_oracle.evaluate(gold, pred)
        else:
            return self.general_oracle.evaluate(gold, pred)
