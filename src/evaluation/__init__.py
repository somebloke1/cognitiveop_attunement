"""
Evaluation module for cognitive operation training.

This module implements a hybrid evaluation architecture:
1. Surface Analyzer: Fast local deterministic checks
2. LLM Evaluator: Deep semantic/logical analysis (Gemini, extensible)
3. Hybrid Reward: Combines both for TRL-compatible reward function

The architecture supports future decomposition into distinct judgment phases:
- Wonder/Context surfacing
- Condition identification and link formulation
- Fulfillment assessment
- Judgment rendering
"""

from .surface_analyzer import SurfaceAnalyzer, SurfaceAnalysis
from .llm_evaluator import LlmEvaluator, SemanticEvaluation, EvaluationDimension
from .hybrid_reward import hybrid_reward_fn, HybridRewardConfig

__all__ = [
    "SurfaceAnalyzer",
    "SurfaceAnalysis",
    "LlmEvaluator",
    "SemanticEvaluation",
    "EvaluationDimension",
    "hybrid_reward_fn",
    "HybridRewardConfig",
]
