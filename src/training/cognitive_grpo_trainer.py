#!/usr/bin/env python3
"""
CognitiveGRPOTrainer - Custom GRPOTrainer that preserves cognitive operation columns.

The base TRL GRPOTrainer has a hardcoded _set_signature_columns_if_needed() that only
preserves ["prompt", "image", "images"]. This causes all custom columns needed for
our reward function (expected_judgment, domain, input_text, etc.) to be dropped.

This subclass overrides that method to include our custom columns, enabling the
reward function to receive the data it needs for correctness-dominant scoring.

Usage:
    from src.training.cognitive_grpo_trainer import CognitiveGRPOTrainer

    trainer = CognitiveGRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )
"""

from trl import GRPOTrainer


class CognitiveGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer subclass that preserves columns needed for cognitive operation training.

    The key fix is overriding _set_signature_columns_if_needed() to include:
    - expected_judgment: Oracle-verified correct judgment for correctness scoring
    - domain: Domain type for domain-appropriate verification
    - input_text: Original input for evidence grounding checks
    - citations: JSON-encoded expected citations
    - verification_chain: JSON-encoded domain-specific verification requirements

    Without this fix, these columns are removed before training, causing the reward
    function to receive empty values and return identical rewards for all completions,
    which results in zero gradient (no learning).
    """

    # Columns required by our cognitive training pipeline
    COGNITIVE_COLUMNS = [
        # TRL defaults
        "prompt",
        "image",
        "images",
        # Core cognitive operation columns
        "expected_judgment",  # Oracle-verified correct judgment (Yes/No/Insufficient)
        "domain",  # Domain type (e.g., astronomy, ethics, mathematics)
        "input_text",  # Original user input for evidence grounding
        "proposition",  # The proposition being judged (for hybrid LLM evaluator)
        "evidence",  # Evidence provided (for hybrid LLM evaluator)
        # Oracle answer keys (for clean post-processed data)
        "oracle_conditions",  # Oracle conditions for evaluator grading
        "oracle_temporal_context",  # Oracle temporal context for evaluator grading
        # Legacy field for backward compatibility
        "conditions",  # Conditions for truth (legacy, used if oracle_conditions not present)
        # Enhanced signal columns (from principled schema)
        "mode",  # Fulfillment mode (formal/empirical/common_sense/dialectical)
        "difficulty",  # Difficulty level (easy/medium/hard)
        "temporal_focus",  # Temporal challenge type if any
        "temporal_context",  # Temporal situation details from scenario
        "edge_case",  # Edge case type (vagueness/self_reference/future_contingent/etc.)
        "adversarial_context",  # Adversarial context type if any
        "scenario_notes",  # Oracle's reasoning about the correct judgment
        # Legacy columns
        "citations",  # JSON-encoded expected citations (legacy)
        "verification_chain",  # JSON-encoded domain verification requirements (legacy)
    ]

    def _set_signature_columns_if_needed(self):
        """
        Override to preserve cognitive operation columns in addition to TRL defaults.

        The base GRPOTrainer only preserves ["prompt", "image", "images"], which causes
        our custom columns to be removed even when remove_unused_columns=False.

        This override ensures all columns needed by our reward function are preserved
        through the dataloader and passed to _calculate_rewards().
        """
        if self._signature_columns is None:
            self._signature_columns = self.COGNITIVE_COLUMNS.copy()
