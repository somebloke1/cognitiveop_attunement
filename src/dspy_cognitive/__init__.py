"""
DSPy-based Cognitive Operations

This module provides Lonergan operational context and data generation
for the cognitive operations training pipeline.

Kept components:
- lonergan_context: Operational anchors (Greek, Hebrew, Sanskrit, Latin terms)
- data_generation: Scenario and judgment generation signatures
"""

from .lonergan_context import (
    get_judgment_context,
    get_verification_context,
    load_full_synopsis,
    JUDGMENT_CONTEXT,
    POP_SCIENCE_CONTEXT,
    OPERATIONAL_ANCHORS,
)

from .data_generation import (
    GenerateScenarioSignature,
    GenerateJudgmentSignature,
    OracleDataPipeline,
    GeneratedExample,
    ScenarioGenerator,
    JudgmentGenerator,
)

__all__ = [
    # Lonergan Context
    "get_judgment_context",
    "get_verification_context",
    "load_full_synopsis",
    "JUDGMENT_CONTEXT",
    "POP_SCIENCE_CONTEXT",
    "OPERATIONAL_ANCHORS",
    # Data Generation
    "GenerateScenarioSignature",
    "GenerateJudgmentSignature",
    "OracleDataPipeline",
    "GeneratedExample",
    "ScenarioGenerator",
    "JudgmentGenerator",
]
