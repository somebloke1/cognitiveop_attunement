"""
Schema module for Cognitive Organism Training Pipeline.

Exports all schema classes and utilities for training data structures.
"""

from .extended_schema import (
    # Enums
    DomainType,
    JudgmentType,
    DistractorType,
    # Dataclasses
    EvidenceCitation,
    DomainVerificationChain,
    StudentPacket,
    EvaluatorPacket,
    DualPacket,
    CoverageCell,
    EnhancedJudgmentSample,
    DistractorSample,
    # Validation helpers
    validate_domain,
    validate_judgment,
    validate_difficulty,
    VALID_DOMAINS,
    VALID_JUDGMENTS,
)

__all__ = [
    # Enums
    "DomainType",
    "JudgmentType",
    "DistractorType",
    # Dataclasses
    "EvidenceCitation",
    "DomainVerificationChain",
    "StudentPacket",
    "EvaluatorPacket",
    "DualPacket",
    "CoverageCell",
    "EnhancedJudgmentSample",
    "DistractorSample",
    # Validation helpers
    "validate_domain",
    "validate_judgment",
    "validate_difficulty",
    "VALID_DOMAINS",
    "VALID_JUDGMENTS",
]
