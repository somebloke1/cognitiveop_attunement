"""
Training module for Cognitive Organism.

Contains:
- enhanced_reward: Enhanced reward function with correctness-dominant scoring
- curriculum: Curriculum scheduling for graduated difficulty
"""

from .enhanced_reward import (
    score_correctness,
    score_evidence_grounding,
    score_domain_verification,
    score_structure,
    compute_reward,
    reward_fn,
    extract_judgment,
    extract_quotes,
)

__all__ = [
    "score_correctness",
    "score_evidence_grounding",
    "score_domain_verification",
    "score_structure",
    "compute_reward",
    "reward_fn",
    "extract_judgment",
    "extract_quotes",
]
