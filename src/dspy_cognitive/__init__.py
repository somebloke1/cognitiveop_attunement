"""
DSPy-based Cognitive Operations

This module implements the four cognitive operations (Attention, Understanding,
Judgment, Decision) as DSPy Signatures and Modules, with multi-tier verification
and GRPO-based reinforcement learning.

Architecture:
- Signatures: Typed interfaces for each cognitive operation
- Modules: DSPy programs implementing the operations
- Metrics: Multi-tier verification (deterministic → heuristic → oracle)
- Training: ArborGRPO for reinforcement learning with LoRA
"""

from .signatures import (
    AttentionSignature,
    UnderstandingSignature,
    JudgmentSignature,
    DecisionSignature,
)

from .modules import (
    AttentionModule,
    UnderstandingModule,
    JudgmentModule,
    DecisionModule,
    CognitiveProgram,
)

from .metrics import (
    judgment_metric,
    JudgmentVerifier,
)

from .data_generation import (
    OracleDataPipeline,
    GeneratedExample,
    ScenarioGenerator,
    JudgmentGenerator,
)

from .lonergan_context import (
    get_judgment_context,
    get_verification_context,
    load_full_synopsis,
    JUDGMENT_CONTEXT,
    POP_SCIENCE_CONTEXT,
    OPERATIONAL_ANCHORS,
)

from .invulnerability import (
    InvulnerabilityVerifier,
    InvulnerabilityResult,
    LevelThemedChallengeGenerator,
    LEVEL_CHALLENGE_PROMPTS,
)

__all__ = [
    # Signatures
    "AttentionSignature",
    "UnderstandingSignature",
    "JudgmentSignature",
    "DecisionSignature",
    # Modules
    "AttentionModule",
    "UnderstandingModule",
    "JudgmentModule",
    "DecisionModule",
    "CognitiveProgram",
    # Metrics
    "judgment_metric",
    "JudgmentVerifier",
    # Data Generation
    "OracleDataPipeline",
    "GeneratedExample",
    "ScenarioGenerator",
    "JudgmentGenerator",
    # Lonergan Context
    "get_judgment_context",
    "get_verification_context",
    "load_full_synopsis",
    "JUDGMENT_CONTEXT",
    "POP_SCIENCE_CONTEXT",
    "OPERATIONAL_ANCHORS",
    # Invulnerability
    "InvulnerabilityVerifier",
    "InvulnerabilityResult",
    "LevelThemedChallengeGenerator",
    "LEVEL_CHALLENGE_PROMPTS",
]
