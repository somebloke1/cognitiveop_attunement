"""
Multi-Tier Verification Metrics for Cognitive Operations

Tier 1: Deterministic - Structural checks, logical consistency (free)
Tier 2: Heuristic - Pattern matching, counterposition detection (free)
Tier 3: Oracle - Defer to stronger model (Gemini Pro) for difficult cases (API cost)

The metric function returns a score [0.0, 1.0] used by GRPO for training.
"""

from typing import List, Optional, Callable
from dataclasses import dataclass
import dspy


@dataclass
class VerificationResult:
    """Result of verification at any tier."""
    score: float  # 0.0 to 1.0
    tier_used: int  # 1, 2, or 3
    explanation: str
    passed_structural: bool
    passed_consistency: bool
    oracle_score: Optional[float] = None


class JudgmentVerifier:
    """
    Multi-tier verifier for judgment correctness.

    Tier 1: Structural checks
    - Output format correct (Fulfilled/Not fulfilled, Yes/No/Insufficient)
    - All conditions addressed

    Tier 2: Consistency checks
    - Judgment consistent with condition fulfillment
    - Domain-appropriate reasoning

    Tier 3: Oracle (Gemini Pro or similar)
    - Full correctness evaluation by stronger model
    """

    def __init__(
        self,
        oracle_model: Optional[str] = None,
        use_oracle: bool = True,
        oracle_threshold: float = 0.7,  # Use oracle if Tier 1+2 score below this
    ):
        self.oracle_model = oracle_model or "gemini/gemini-2.0-flash"
        self.use_oracle = use_oracle
        self.oracle_threshold = oracle_threshold
        self._oracle_judge = None

    def _init_oracle(self):
        """Lazy initialization of oracle judge."""
        if self._oracle_judge is None and self.use_oracle:
            from .oracle import OracleJudge
            self._oracle_judge = OracleJudge(model=self.oracle_model)

    # ============ TIER 1: STRUCTURAL CHECKS ============

    def check_structural(self, pred) -> tuple[float, str]:
        """
        Tier 1: Verify output structure is correct.

        Returns (score, explanation)
        """
        issues = []

        # Check condition_fulfillment format
        if not hasattr(pred, 'condition_fulfillment'):
            return 0.0, "Missing condition_fulfillment field"

        fulfillment = pred.condition_fulfillment
        if not isinstance(fulfillment, list):
            issues.append("condition_fulfillment is not a list")
        else:
            valid_values = {"Fulfilled", "Not fulfilled"}
            for i, f in enumerate(fulfillment):
                if f not in valid_values:
                    issues.append(f"condition_fulfillment[{i}] = '{f}' not in {valid_values}")

        # Check judgment format
        if not hasattr(pred, 'judgment'):
            return 0.0, "Missing judgment field"

        valid_judgments = {"Yes", "No", "Insufficient"}
        if pred.judgment not in valid_judgments:
            issues.append(f"judgment = '{pred.judgment}' not in {valid_judgments}")

        # Check reasoning exists
        if not hasattr(pred, 'reasoning') or not pred.reasoning:
            issues.append("Missing or empty reasoning")

        if issues:
            return 0.5, "; ".join(issues)
        return 1.0, "Structural check passed"

    # ============ TIER 2: CONSISTENCY CHECKS ============

    def check_consistency(self, gold, pred) -> tuple[float, str]:
        """
        Tier 2: Verify internal consistency.

        - Judgment should be consistent with condition fulfillment
        - Domain-appropriate reasoning patterns
        """
        issues = []
        score = 1.0

        # Consistency: All fulfilled → should affirm (unless counterposition)
        if hasattr(pred, 'condition_fulfillment') and isinstance(pred.condition_fulfillment, list):
            all_fulfilled = all(f == "Fulfilled" for f in pred.condition_fulfillment)
            any_unfulfilled = any(f == "Not fulfilled" for f in pred.condition_fulfillment)

            if all_fulfilled and pred.judgment == "No":
                # This might be valid for counterpositions, but flag it
                if gold.domain != "philosophic":
                    issues.append("All conditions fulfilled but judgment is No (without philosophic justification)")
                    score -= 0.3

            if any_unfulfilled and pred.judgment == "Yes":
                issues.append("Some conditions not fulfilled but judgment is Yes")
                score -= 0.5

        # Domain-specific checks
        if hasattr(gold, 'domain'):
            if gold.domain == "mathematical":
                # Mathematical judgments should reference proof/derivation
                if "proof" not in pred.reasoning.lower() and "derive" not in pred.reasoning.lower():
                    if pred.judgment in ["Yes", "No"]:
                        issues.append("Mathematical judgment without reference to proof")
                        score -= 0.1

            elif gold.domain == "empirical":
                # Empirical judgments should reference data/observation
                if "data" not in pred.reasoning.lower() and "observ" not in pred.reasoning.lower():
                    if pred.judgment in ["Yes", "No"]:
                        issues.append("Empirical judgment without reference to data")
                        score -= 0.1

            elif gold.domain == "pop_science":
                # Pop science should detect unfulfilled conditions
                if pred.judgment == "Yes":
                    issues.append("Pop science claim affirmed without scrutiny")
                    score -= 0.3

        if issues:
            return max(0.0, score), "; ".join(issues)
        return 1.0, "Consistency check passed"

    # ============ TIER 3: ORACLE ============

    def check_oracle(self, gold, pred) -> tuple[float, str]:
        """
        Tier 3: Defer to stronger model for correctness.

        This is the expensive tier - only used when Tier 1+2 are insufficient.
        """
        self._init_oracle()

        if self._oracle_judge is None:
            return 0.5, "Oracle not available"

        return self._oracle_judge.evaluate(gold, pred)

    # ============ COMBINED METRIC ============

    def verify(self, gold, pred, use_oracle_if_needed: bool = True) -> VerificationResult:
        """
        Run multi-tier verification.

        Returns VerificationResult with score and explanation.
        """
        # Tier 1: Structural
        structural_score, structural_explanation = self.check_structural(pred)
        if structural_score < 0.5:
            return VerificationResult(
                score=structural_score * 0.5,  # Heavy penalty for structural failure
                tier_used=1,
                explanation=f"Tier 1 (structural): {structural_explanation}",
                passed_structural=False,
                passed_consistency=False,
            )

        # Tier 2: Consistency
        consistency_score, consistency_explanation = self.check_consistency(gold, pred)
        tier_1_2_score = (structural_score + consistency_score) / 2

        if tier_1_2_score >= self.oracle_threshold:
            return VerificationResult(
                score=tier_1_2_score,
                tier_used=2,
                explanation=f"Tier 1+2: {structural_explanation}; {consistency_explanation}",
                passed_structural=True,
                passed_consistency=consistency_score > 0.7,
            )

        # Tier 3: Oracle (if enabled and score below threshold)
        if use_oracle_if_needed and self.use_oracle:
            oracle_score, oracle_explanation = self.check_oracle(gold, pred)
            # Weight oracle heavily but keep some structural/consistency signal
            final_score = 0.3 * tier_1_2_score + 0.7 * oracle_score
            return VerificationResult(
                score=final_score,
                tier_used=3,
                explanation=f"Tier 3 (oracle): {oracle_explanation}",
                passed_structural=True,
                passed_consistency=consistency_score > 0.7,
                oracle_score=oracle_score,
            )

        return VerificationResult(
            score=tier_1_2_score,
            tier_used=2,
            explanation=f"Tier 1+2 (oracle skipped): {consistency_explanation}",
            passed_structural=True,
            passed_consistency=consistency_score > 0.7,
        )


# ============ METRIC FUNCTION FOR GRPO ============

# Global verifier instance (initialized on first use)
_verifier: Optional[JudgmentVerifier] = None


def get_verifier(
    oracle_model: Optional[str] = None,
    use_oracle: bool = True,
) -> JudgmentVerifier:
    """Get or create global verifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = JudgmentVerifier(
            oracle_model=oracle_model,
            use_oracle=use_oracle,
        )
    return _verifier


def judgment_metric(
    gold,
    pred,
    trace=None,
    use_oracle: bool = True,
) -> float:
    """
    DSPy metric function for judgment correctness.

    Used by GRPO for training signal.

    Args:
        gold: The example with expected values
        pred: The model's prediction
        trace: Optional trace (used by DSPy optimizers)

    Returns:
        Float score [0.0, 1.0] for GRPO optimization
        If trace is provided, returns bool for pass/fail
    """
    verifier = get_verifier(use_oracle=use_oracle)
    result = verifier.verify(gold, pred)

    # If trace is provided (optimization context), return bool
    if trace is not None:
        return result.score >= 0.7

    return result.score


def judgment_metric_no_oracle(gold, pred, trace=None) -> float:
    """Metric without oracle - for fast iteration."""
    return judgment_metric(gold, pred, trace, use_oracle=False)
