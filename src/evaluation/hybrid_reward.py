"""
Hybrid Reward Function for Cognitive Operation Training.

This module combines local surface analysis with LLM semantic evaluation
to produce a TRL-compatible reward function for GRPO training.

Architecture:
1. Surface analysis runs locally (fast, free)
2. LLM evaluation runs in cloud (semantic depth)
3. Rewards combine both with configurable weighting

The reward function is designed for the judgment cognitive operation but
the architecture supports future extension to other operations.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .surface_analyzer import SurfaceAnalyzer, SurfaceAnalysis
from .llm_evaluator import LlmEvaluator, SemanticEvaluation, EvaluationDimension
from .logging_config import get_eval_logger, log_function_call, get_gemini_logger
from .async_evaluator import AsyncPrefetchEvaluator


@dataclass
class HybridRewardConfig:
    """Configuration for hybrid reward computation."""

    # LLM provider settings
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3-flash-preview"
    llm_api_key: Optional[str] = None  # If None, reads from env

    # Reward scaling
    # Final reward = correctness_weight * correctness + semantic_weight * holistic_score
    correctness_weight: float = 0.3  # Binary correctness component
    semantic_weight: float = 0.7  # LLM semantic evaluation component

    # Correctness reward values
    correct_judgment_reward: float = 1.0
    incorrect_judgment_reward: float = -0.5
    no_judgment_reward: float = -0.3

    # Dimension weights for holistic score (passed to LlmEvaluator)
    dimension_weights: Dict[EvaluationDimension, float] = field(
        default_factory=lambda: {
            EvaluationDimension.CONDITION_IDENTIFICATION: 0.15,
            EvaluationDimension.EVIDENCE_MAPPING: 0.20,
            EvaluationDimension.REASONING_VALIDITY: 0.25,
            EvaluationDimension.JUDGMENT_COHERENCE: 0.20,
            EvaluationDimension.OPERATIONAL_FIDELITY: 0.15,
            EvaluationDimension.AUTHENTIC_INTENT: 0.05,
        }
    )

    # Parallel evaluation
    max_workers: int = 8  # Max concurrent LLM calls

    # Batched evaluation: send multiple completions in one LLM call
    # Only works when all completions share the same context (GRPO case)
    # Set to False to always use individual parallel calls
    use_batched_evaluation: bool = True

    # Gemini context caching: cache system instruction to reduce token costs
    # ~50% reduction in input tokens for repeated evaluations
    use_gemini_cache: bool = True
    gemini_cache_ttl: int = 172800  # 48 hours (in seconds)

    # Local result caching (future)
    enable_local_cache: bool = False
    local_cache_dir: Optional[str] = None
    
    # Async prefetch: fire Gemini requests ahead of when results are needed
    # This overlaps API latency with local computation
    use_async_prefetch: bool = False  # Disabled by default for backward compat


class HybridRewardFunction:
    """
    Combines local surface analysis with LLM semantic evaluation.

    This class manages the evaluation pipeline and produces rewards
    compatible with TRL's GRPOTrainer.
    """

    def __init__(self, config: Optional[HybridRewardConfig] = None):
        """
        Initialize the hybrid reward function.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or HybridRewardConfig()

        # Initialize components
        self.surface_analyzer = SurfaceAnalyzer()
        self.llm_evaluator = LlmEvaluator(
            provider=self.config.llm_provider,
            model_name=self.config.llm_model,
            api_key=self.config.llm_api_key,
            weights=self.config.dimension_weights,
            use_cache=self.config.use_gemini_cache,
            cache_ttl=self.config.gemini_cache_ttl,
        )

        # Track initialization state
        self._initialized = False
        
        # Async prefetch evaluator (initialized in initialize() if enabled)
        self._async_evaluator: Optional[AsyncPrefetchEvaluator] = None
        self._prefetch_step_counter = 0
        self._pending_prefetch: Optional[int] = None

    def initialize(self) -> None:
        """
        Initialize resources before training loop.

        Creates Gemini cache if caching is enabled. Call this before
        starting the training loop.
        """
        if self._initialized:
            return

        if self.config.use_gemini_cache:
            cache_name = self.llm_evaluator.create_cache()
            if cache_name:
                get_eval_logger().logger.info(f"Gemini cache initialized: {cache_name}")
            else:
                get_eval_logger().logger.warning(
                    "Failed to create Gemini cache, proceeding without caching"
                )
        
        # Initialize async prefetch evaluator if enabled
        if self.config.use_async_prefetch:
            self._async_evaluator = AsyncPrefetchEvaluator(self.llm_evaluator)
            self._async_evaluator.start()
            get_eval_logger().logger.info("Async prefetch evaluator started")

        self._initialized = True

    def cleanup(self) -> None:
        """
        Cleanup resources after training loop.

        Deletes Gemini cache if it was created. Call this after
        training completes (in a finally block to ensure cleanup).
        """
        # Stop async evaluator first
        if self._async_evaluator:
            self._async_evaluator.stop()
            self._async_evaluator = None
            get_eval_logger().logger.info("Async prefetch evaluator stopped")
        
        if self.llm_evaluator.cache_active:
            if self.llm_evaluator.delete_cache():
                get_eval_logger().logger.info("Gemini cache cleaned up")

        self._initialized = False

    def __enter__(self):
        """Context manager entry - initializes resources."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleans up resources."""
        self.cleanup()
        return False

    def compute_reward(
        self,
        completion: str,
        proposition: str,
        evidence: str,
        conditions: str,
        expected_judgment: str,
        domain: str,
        mode: str = "",
        difficulty: str = "medium",
        temporal_focus: str = "",
        temporal_context: str = "",
        edge_case: str = "",
        adversarial_context: str = "",
        scenario_notes: str = "",
    ) -> float:
        """
        Compute reward for a single completion.

        Args:
            completion: Student model's completion
            proposition: The proposition being judged
            evidence: Evidence provided
            conditions: Conditions for truth
            expected_judgment: Oracle's correct judgment
            domain: Domain type
            mode: Fulfillment mode (formal/empirical/common_sense/dialectical)
            difficulty: Difficulty level (easy/medium/hard)
            temporal_focus: Type of temporal challenge if any
            temporal_context: Temporal context from the scenario
            edge_case: Edge case type if any (vagueness, self_reference, etc.)
            adversarial_context: Adversarial context if any (deception, manipulation, etc.)
            scenario_notes: Oracle's reasoning about the correct judgment

        Returns:
            Reward value (typically -0.5 to 1.0)
        """
        # 1. Surface analysis (local, fast)
        surface = self.surface_analyzer.analyze(completion, domain)

        # 2. If surface_only mode, skip LLM and use fallback
        if self.config.surface_only:
            return self._compute_fallback_reward(surface, expected_judgment)

        # 3. LLM semantic evaluation with full context
        semantic = self.llm_evaluator.evaluate(
            completion=completion,
            proposition=proposition,
            evidence=evidence,
            conditions=conditions,
            expected_judgment=expected_judgment,
            domain=domain,
            surface_analysis=surface,
            mode=mode,
            difficulty=difficulty,
            temporal_focus=temporal_focus,
            temporal_context=temporal_context,
            edge_case=edge_case,
            adversarial_context=adversarial_context,
            scenario_notes=scenario_notes,
        )

        # 4. Compute composite reward
        return self._compute_composite_reward(surface, semantic, expected_judgment)

    def compute_rewards_batch(
        self,
        completions: List[str],
        propositions: List[str],
        evidences: List[str],
        oracle_conditions_list: List[str],
        oracle_temporal_contexts: List[str],
        expected_judgments: List[str],
        domains: List[str],
        prompts: Optional[List[str]] = None,
        modes: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        temporal_focuses: Optional[List[str]] = None,
        temporal_contexts: Optional[List[str]] = None,
        edge_cases: Optional[List[str]] = None,
        adversarial_contexts: Optional[List[str]] = None,
        scenario_notes_list: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Compute rewards for a batch of completions.

        Uses parallel execution for LLM calls.

        Args:
            completions: List of student completions
            propositions: List of propositions
            evidences: List of evidence texts
            oracle_conditions_list: List of oracle conditions (answer key for grading)
            oracle_temporal_contexts: List of oracle temporal contexts (answer key)
            expected_judgments: List of expected judgments
            domains: List of domain types
            modes: List of fulfillment modes
            difficulties: List of difficulty levels
            temporal_focuses: List of temporal focus types
            temporal_contexts: List of temporal context strings (not oracle, just context)
            edge_cases: List of edge case types
            adversarial_contexts: List of adversarial context types
            scenario_notes_list: List of oracle reasoning notes

        Returns:
            List of reward values
        """
        n = len(completions)
        rewards = [0.0] * n

        # Default empty lists for optional parameters
        prompts = prompts or [""] * n
        modes = modes or [""] * n
        difficulties = difficulties or ["medium"] * n
        temporal_focuses = temporal_focuses or [""] * n
        temporal_contexts = temporal_contexts or [""] * n
        edge_cases = edge_cases or [""] * n
        adversarial_contexts = adversarial_contexts or [""] * n
        scenario_notes_list = scenario_notes_list or [""] * n

        # Run surface analysis locally (fast)
        surface_analyses = [
            self.surface_analyzer.analyze(comp, dom)
            for comp, dom in zip(completions, domains)
        ]

        # Check if all completions share the same context (GRPO case)
        # If so, use single-call batch evaluation for efficiency
        all_same_context = (
            n > 1
            and len(set(propositions)) == 1
            and len(set(evidences)) == 1
            and len(set(oracle_conditions_list)) == 1
            and len(set(expected_judgments)) == 1
            and len(set(domains)) == 1
        )

        if all_same_context and self.config.use_batched_evaluation:
            # Use single LLM call for all completions
            eval_logger = get_eval_logger()
            eval_logger.logger.info(
                f"Using batched single-call evaluation for {n} completions"
            )

            semantic_evals = self.llm_evaluator.evaluate_batch_single_call(
                completions=completions,
                proposition=propositions[0],
                evidence=evidences[0],
                expected_judgment=expected_judgments[0],
                domain=domains[0],
                surface_analyses=surface_analyses,
                mode=modes[0] if modes else "",
                oracle_conditions=oracle_conditions_list[0],
                oracle_temporal_context=oracle_temporal_contexts[0] if oracle_temporal_contexts else "",
                difficulty=difficulties[0] if difficulties else "medium",
                temporal_focus=temporal_focuses[0] if temporal_focuses else "",
                edge_case=edge_cases[0] if edge_cases else "",
                adversarial_context=adversarial_contexts[0] if adversarial_contexts else "",
                scenario_notes=scenario_notes_list[0] if scenario_notes_list else "",
            )

            for i, (surface, semantic) in enumerate(
                zip(surface_analyses, semantic_evals)
            ):
                rewards[i] = self._compute_composite_reward(
                    surface, semantic, expected_judgments[i]
                )
                eval_logger.log_reward_computation(
                    index=i,
                    completion_preview=completions[i][:200],
                    expected_judgment=expected_judgments[i],
                    extracted_judgment=surface.judgment_extracted or "NONE",
                    correctness_score=self._get_correctness_score(
                        surface, expected_judgments[i]
                    ),
                    semantic_score=semantic.holistic_score,
                    final_reward=rewards[i],
                    prompt=prompts[i],
                    full_completion=completions[i],
                )

            return rewards

        # Different contexts - use parallel individual calls (original behavior)
        surface_analyses = [
            self.surface_analyzer.analyze(comp, dom)
            for comp, dom in zip(completions, domains)
        ]

        # Run LLM evaluations in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            for i in range(n):
                future = executor.submit(
                    self.llm_evaluator.evaluate,
                    completion=completions[i],
                    proposition=propositions[i],
                    evidence=evidences[i],
                    # Pass oracle_conditions as 'conditions' (legacy interface)
                    conditions=oracle_conditions_list[i] if i < len(oracle_conditions_list) else "",
                    expected_judgment=expected_judgments[i],
                    domain=domains[i],
                    surface_analysis=surface_analyses[i],
                    mode=modes[i] if i < len(modes) else "",
                    difficulty=difficulties[i] if i < len(difficulties) else "medium",
                    temporal_focus=temporal_focuses[i]
                    if i < len(temporal_focuses)
                    else "",
                    # Pass oracle_temporal_context as 'temporal_context' (legacy interface)
                    temporal_context=oracle_temporal_contexts[i]
                    if i < len(oracle_temporal_contexts)
                    else "",
                    edge_case=edge_cases[i] if i < len(edge_cases) else "",
                    adversarial_context=adversarial_contexts[i]
                    if i < len(adversarial_contexts)
                    else "",
                    scenario_notes=scenario_notes_list[i]
                    if i < len(scenario_notes_list)
                    else "",
                )
                futures[future] = i

            eval_logger = get_eval_logger()
            for future in as_completed(futures):
                i = futures[future]
                try:
                    semantic = future.result()
                    rewards[i] = self._compute_composite_reward(
                        surface_analyses[i],
                        semantic,
                        expected_judgments[i],
                    )
                    # Log individual reward computation
                    eval_logger.log_reward_computation(
                        index=i,
                        completion_preview=completions[i][:200],
                        expected_judgment=expected_judgments[i],
                        extracted_judgment=surface_analyses[i].judgment_extracted
                        or "NONE",
                        correctness_score=self._get_correctness_score(
                            surface_analyses[i], expected_judgments[i]
                        ),
                        semantic_score=semantic.holistic_score,
                        final_reward=rewards[i],
                        prompt=prompts[i],
                        full_completion=completions[i],
                    )
                except Exception as e:
                    eval_logger.logger.error(f"LLM evaluation failed for idx={i}: {e}")
                    # On error, fall back to surface-only reward
                    rewards[i] = self._compute_fallback_reward(
                        surface_analyses[i],
                        expected_judgments[i],
                    )
                    eval_logger.log_reward_computation(
                        index=i,
                        completion_preview=completions[i][:200],
                        expected_judgment=expected_judgments[i],
                        extracted_judgment=surface_analyses[i].judgment_extracted
                        or "NONE",
                        correctness_score=self._get_correctness_score(
                            surface_analyses[i], expected_judgments[i]
                        ),
                        semantic_score=0.0,  # Fallback, no LLM score
                        final_reward=rewards[i],
                        prompt=prompts[i],
                        full_completion=completions[i],
                    )

        return rewards

    def _get_correctness_score(
        self, surface: SurfaceAnalysis, expected_judgment: str
    ) -> float:
        """Helper to get correctness score for logging."""
        if surface.judgment_extracted is None:
            return self.config.no_judgment_reward
        elif surface.judgment_extracted.lower() == expected_judgment.lower():
            return self.config.correct_judgment_reward
        else:
            return self.config.incorrect_judgment_reward

    def _compute_composite_reward(
        self,
        surface: SurfaceAnalysis,
        semantic: SemanticEvaluation,
        expected_judgment: str,
    ) -> float:
        """
        Compute composite reward from surface and semantic evaluations.

        Reward = correctness_weight * correctness + semantic_weight * holistic_score

        Where correctness is:
        - correct_judgment_reward if judgment matches expected (reduced by 50% for fallback)
        - incorrect_judgment_reward if judgment doesn't match
        - no_judgment_reward if no judgment detected

        When semantic evaluation is a fallback (LLM parse failed), correctness
        is reduced to reflect lower confidence in the evaluation.
        """
        # Correctness component
        if surface.judgment_extracted is None:
            correctness = self.config.no_judgment_reward
        elif surface.judgment_extracted.lower() == expected_judgment.lower():
            correctness = self.config.correct_judgment_reward
            # Reduce correctness for fallback evaluations (lower confidence)
            if semantic.is_fallback:
                correctness *= 0.5
        else:
            correctness = self.config.incorrect_judgment_reward

        # Semantic component (0.0 to 1.0)
        # Scale to reward range
        semantic_score = semantic.holistic_score

        # Composite
        reward = (
            self.config.correctness_weight * correctness
            + self.config.semantic_weight * semantic_score
        )

        return reward

    def _compute_fallback_reward(
        self,
        surface: SurfaceAnalysis,
        expected_judgment: str,
    ) -> float:
        """
        Compute fallback reward when LLM evaluation fails.

        Uses only surface analysis with reduced confidence.
        """
        # Basic correctness
        if surface.judgment_extracted is None:
            correctness = self.config.no_judgment_reward
        elif surface.judgment_extracted.lower() == expected_judgment.lower():
            correctness = self.config.correct_judgment_reward * 0.5  # Reduced
        else:
            correctness = self.config.incorrect_judgment_reward

        # Simple surface-based semantic proxy
        surface_score = 0.0
        if surface.has_condition_language:
            surface_score += 0.2
        if surface.has_fulfillment_language:
            surface_score += 0.2
        if surface.has_link_language:
            surface_score += 0.2
        if surface.has_quotations:
            surface_score += 0.2
        if surface.judgment_extracted is not None:
            surface_score += 0.2

        return (
            self.config.correctness_weight * correctness
            + self.config.semantic_weight * surface_score * 0.5  # Reduced confidence
        )


# Global instance for TRL compatibility
_hybrid_reward_instance: Optional[HybridRewardFunction] = None


def get_hybrid_reward_instance(
    config: Optional[HybridRewardConfig] = None,
) -> HybridRewardFunction:
    """Get or create the global hybrid reward instance."""
    global _hybrid_reward_instance
    if _hybrid_reward_instance is None or config is not None:
        _hybrid_reward_instance = HybridRewardFunction(config)
    return _hybrid_reward_instance


def hybrid_reward_fn(
    completions: List[str],
    prompts: List[str],
    expected_judgment: Optional[List[str]] = None,
    domain: Optional[List[str]] = None,
    input_text: Optional[List[str]] = None,
    proposition: Optional[List[str]] = None,
    evidence: Optional[List[str]] = None,
    # Oracle answer keys (for evaluator grading - students don't see these)
    oracle_conditions: Optional[List[str]] = None,
    oracle_temporal_context: Optional[List[str]] = None,
    # Legacy field - kept for backward compatibility, maps to oracle_conditions
    conditions: Optional[List[str]] = None,
    # Enhanced context fields
    mode: Optional[List[str]] = None,
    difficulty: Optional[List[str]] = None,
    temporal_focus: Optional[List[str]] = None,
    temporal_context: Optional[List[str]] = None,
    edge_case: Optional[List[str]] = None,
    adversarial_context: Optional[List[str]] = None,
    scenario_notes: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """
    TRL-compatible reward function using hybrid evaluation.

    This function is passed to GRPOTrainer as reward_funcs.

    Args:
        completions: List of model completions
        prompts: List of prompts (from TRL, may contain full context)
        expected_judgment: List of expected judgments
        domain: List of domain types
        input_text: List of original input texts (fallback for parsing)
        proposition: List of propositions (if available separately)
        evidence: List of evidence texts (if available separately)
        oracle_conditions: List of oracle conditions (answer key for grading)
        oracle_temporal_context: List of oracle temporal contexts (answer key)
        conditions: LEGACY - maps to oracle_conditions for backward compatibility
        mode: List of fulfillment modes (formal/empirical/common_sense/dialectical)
        difficulty: List of difficulty levels
        temporal_focus: List of temporal focus types
        temporal_context: List of temporal context strings
        edge_case: List of edge case types
        adversarial_context: List of adversarial context types
        scenario_notes: List of oracle reasoning notes
        **kwargs: Additional TRL-provided fields

    Returns:
        List of reward floats
    """
    if not completions:
        return []

    n = len(completions)

    # Get or create evaluator instance
    evaluator = get_hybrid_reward_instance()

    # Handle missing fields with defaults
    if expected_judgment is None:
        expected_judgment = [""] * n
    if domain is None:
        domain = ["empirical"] * n
    if input_text is None:
        input_text = [""] * n

    # Parse proposition, evidence from input_text if not provided separately
    if proposition is None or evidence is None:
        propositions = []
        evidences = []

        for i in range(n):
            text = input_text[i] if i < len(input_text) else ""
            prop, evid, _ = _parse_input_text(text)
            propositions.append(
                proposition[i] if proposition and i < len(proposition) else prop
            )
            evidences.append(evidence[i] if evidence and i < len(evidence) else evid)
    else:
        propositions = list(proposition)
        evidences = list(evidence)

    # Handle oracle answer keys - prefer explicit oracle_* fields, fall back to legacy
    # oracle_conditions: answer key for evaluator to grade condition identification
    # oracle_temporal_context: answer key for evaluator to grade temporal analysis
    if oracle_conditions is not None:
        oracle_conditions_list = list(oracle_conditions)
    elif conditions is not None:
        # Legacy fallback - old format used 'conditions' for this
        oracle_conditions_list = list(conditions)
    else:
        oracle_conditions_list = [""] * n
    
    if oracle_temporal_context is not None:
        oracle_temporal_contexts = list(oracle_temporal_context)
    else:
        oracle_temporal_contexts = [""] * n

    # Ensure all lists are the right length
    expected_judgment = _pad_list(expected_judgment, n, "")
    domain = _pad_list(domain, n, "empirical")
    propositions = _pad_list(propositions, n, "")
    evidences = _pad_list(evidences, n, "")
    oracle_conditions_list = _pad_list(oracle_conditions_list, n, "")
    oracle_temporal_contexts = _pad_list(oracle_temporal_contexts, n, "")

    # Enhanced fields with defaults
    modes = _pad_list(mode or [], n, "")
    difficulties = _pad_list(difficulty or [], n, "medium")
    temporal_focuses = _pad_list(temporal_focus or [], n, "")
    temporal_contexts = _pad_list(temporal_context or [], n, "")
    edge_cases = _pad_list(edge_case or [], n, "")
    adversarial_contexts = _pad_list(adversarial_context or [], n, "")
    scenario_notes_list = _pad_list(scenario_notes or [], n, "")

    # Log batch start
    eval_logger = get_eval_logger()
    eval_logger.logger.info(f"HYBRID_REWARD_FN called with {n} completions")

    # Compute rewards with full context
    rewards = evaluator.compute_rewards_batch(
        completions=completions,
        propositions=propositions,
        evidences=evidences,
        oracle_conditions_list=oracle_conditions_list,
        oracle_temporal_contexts=oracle_temporal_contexts,
        expected_judgments=expected_judgment,
        domains=domain,
        prompts=prompts,
        modes=modes,
        difficulties=difficulties,
        temporal_focuses=temporal_focuses,
        temporal_contexts=temporal_contexts,
        edge_cases=edge_cases,
        adversarial_contexts=adversarial_contexts,
        scenario_notes_list=scenario_notes_list,
    )

    # Log batch summary
    import statistics

    mean_reward = statistics.mean(rewards) if rewards else 0.0
    std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0.0

    # Count correct vs failures for console summary
    correct_count = sum(
        1
        for i, r in enumerate(rewards)
        if r > 0.5  # Rough threshold for "correct"
    )
    failure_count = n - correct_count

    eval_logger.log_batch_summary(
        n,
        rewards,
        mean_reward,
        std_reward,
        correct_count=correct_count,
        failure_count=failure_count,
    )

    return rewards


def _parse_input_text(text: str) -> tuple:
    """
    Parse proposition, evidence, and conditions from input text.

    Expected format:
        Proposition: ...
        Evidence: ...
        Conditions for truth:
        - condition 1
        - condition 2

        Temporal context:
        - Evidence timeframe: ...
        - Aspectual status: ...

    Returns:
        (proposition, evidence, conditions) tuple
    """
    proposition = ""
    evidence = ""
    conditions = ""

    if not text:
        return proposition, evidence, conditions

    lines = text.split("\n")
    current_section = None
    condition_lines = []

    # Section headers that end the conditions section
    section_headers = (
        "Proposition:",
        "Evidence:",
        "Conditions for truth:",
        "Temporal context:",
    )

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith("Proposition:"):
            current_section = "proposition"
            proposition = line_stripped.replace("Proposition:", "").strip()
        elif line_stripped.startswith("Evidence:"):
            current_section = "evidence"
            evidence = line_stripped.replace("Evidence:", "").strip()
        elif line_stripped.startswith("Conditions for truth:"):
            current_section = "conditions"
        elif line_stripped.startswith("Temporal context:"):
            current_section = "temporal"  # Stop collecting conditions
        elif current_section == "conditions" and line_stripped.startswith("-"):
            condition_lines.append(line_stripped)
        elif current_section == "evidence" and not line_stripped.startswith(
            section_headers
        ):
            # Multi-line evidence
            if line_stripped:
                evidence += " " + line_stripped

    conditions = "\n".join(condition_lines)

    return proposition.strip(), evidence.strip(), conditions.strip()


def _pad_list(lst: List, target_len: int, default: Any) -> List:
    """Pad a list to target length with default value."""
    if len(lst) >= target_len:
        return lst[:target_len]
    return lst + [default] * (target_len - len(lst))
