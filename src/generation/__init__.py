# Generation modules for cognitive operation training data

from src.generation.distractor_generator import (
    DistractorType,
    JudgmentSample,
    DistractorSample,
    DistractorGenerator,
    generate_misaligned_phase_distractor,
    generate_inverted_judgment_distractor,
    generate_reasoning_error_distractor,
    generate_evidence_hallucination_distractor,
    is_p2_language,
    is_p3_language,
    is_valid_distractor,
    distractor_preserves_structure,
    generate_distractors_for_example,
)

__all__ = [
    "DistractorType",
    "JudgmentSample",
    "DistractorSample",
    "DistractorGenerator",
    "generate_misaligned_phase_distractor",
    "generate_inverted_judgment_distractor",
    "generate_reasoning_error_distractor",
    "generate_evidence_hallucination_distractor",
    "is_p2_language",
    "is_p3_language",
    "is_valid_distractor",
    "distractor_preserves_structure",
    "generate_distractors_for_example",
]
