"""
Extended Schema Module for Cognitive Organism Training Pipeline.

Implements data structures that operationalize Lonergan's cognitive theory:
- EnhancedJudgmentSample: Extended judgment sample with all fields
- DistractorSample: Contrastive distractor completions
- StudentPacket / EvaluatorPacket: Dual-packet architecture
- DomainVerificationChain: Domain-specific verification structures
- CoverageCell: Coverage matrix cell identification

Key Principle: Form without substance is empty form.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
import json


# =============================================================================
# Enums
# =============================================================================

class DomainType(Enum):
    """Valid domains for judgment samples."""
    MATHEMATICAL = "mathematical"
    EMPIRICAL = "empirical"
    COMMON_SENSE = "common_sense"
    POP_SCIENCE = "pop_science"
    PHILOSOPHIC = "philosophic"


class JudgmentType(Enum):
    """Valid judgment types."""
    YES = "yes"
    NO = "no"
    INSUFFICIENT = "insufficient"


class DistractorType(Enum):
    """Types of distractor completions for contrastive learning."""
    MISALIGNED_PHASE = "misaligned_phase"
    INVERTED_JUDGMENT = "inverted_judgment"
    REASONING_ERROR = "reasoning_error"
    EVIDENCE_HALLUCINATION = "evidence_hallucination"
    DOMAIN_MISMATCH = "domain_mismatch"


# =============================================================================
# Validation Helpers
# =============================================================================

VALID_DOMAINS = {d.value for d in DomainType}
VALID_JUDGMENTS = {j.value for j in JudgmentType}


def validate_domain(domain: str) -> None:
    """Validate domain is a valid domain type string."""
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of: {VALID_DOMAINS}")


def validate_judgment(judgment: str) -> None:
    """Validate judgment is a valid judgment type string."""
    if judgment not in VALID_JUDGMENTS:
        raise ValueError(f"Invalid judgment '{judgment}'. Must be one of: {VALID_JUDGMENTS}")


def validate_difficulty(difficulty: int) -> None:
    """Validate difficulty is an integer in range 1-5."""
    if not isinstance(difficulty, int) or isinstance(difficulty, bool):
        raise TypeError(f"Difficulty must be an integer, got {type(difficulty).__name__}")
    if difficulty < 1 or difficulty > 5:
        raise ValueError(f"Difficulty must be 1-5, got {difficulty}")


# =============================================================================
# EvidenceCitation
# =============================================================================

@dataclass
class EvidenceCitation:
    """
    A citation from the evidence text supporting a condition.

    Attributes:
        text: The cited text
        start_idx: Start index in the evidence text
        end_idx: End index in the evidence text (exclusive)
        supports_condition: Which condition this citation supports
    """
    text: str
    start_idx: int
    end_idx: int
    supports_condition: str

    def __post_init__(self):
        if self.start_idx < 0:
            raise ValueError(f"start_idx must be non-negative, got {self.start_idx}")
        if self.end_idx <= self.start_idx:
            raise ValueError(f"end_idx ({self.end_idx}) must be greater than start_idx ({self.start_idx})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceCitation":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# DomainVerificationChain
# =============================================================================

@dataclass
class DomainVerificationChain:
    """
    Domain-specific verification chain structure.

    The virtually unconditioned has different structures depending on domain:
    - Mathematical: proof steps, axiom references, QED marker
    - Empirical: data points, methodology, replication status
    - Common Sense: practical outcomes, situation factors
    - Pop Science: unfulfilled claims, actual evidence level
    - Philosophic: performative check, self-consistency, counterposition test
    """
    domain: str
    verification_type: str
    required_elements: List[str]

    # Mathematical domain fields
    proof_steps: Optional[List[str]] = None
    axiom_refs: Optional[List[str]] = None
    qed_marker: Optional[bool] = None

    # Empirical domain fields
    data_points: Optional[List[str]] = None
    methodology: Optional[str] = None
    replication: Optional[str] = None

    # Common sense domain fields
    practical_outcomes: Optional[List[str]] = None
    situation_factors: Optional[List[str]] = None

    # Pop science domain fields
    unfulfilled_claims: Optional[List[str]] = None
    actual_evidence_level: Optional[str] = None

    # Philosophic domain fields
    performative_check: Optional[str] = None
    self_consistency: Optional[bool] = None
    counterposition_test: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainVerificationChain":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# StudentPacket
# =============================================================================

@dataclass
class StudentPacket:
    """
    What the model sees during generation.

    DOES NOT contain expected_judgment or any evaluator fields.
    This ensures the model cannot cheat by seeing the answer.
    """
    prompt: str
    evidence: str
    conditions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StudentPacket":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# EvaluatorPacket
# =============================================================================

@dataclass
class EvaluatorPacket:
    """
    What the reward function sees for evaluation.

    Contains ground truth and evaluation metadata.
    DOES NOT contain prompt (that's in StudentPacket).
    """
    expected_judgment: str
    domain: str
    difficulty: int
    evidence_citations: List[Dict[str, Any]]
    distractor_type: Optional[str]
    verification_chain: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluatorPacket":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# DualPacket
# =============================================================================

@dataclass
class DualPacket:
    """
    Combines StudentPacket and EvaluatorPacket.

    Ensures clean separation between what model sees (student)
    and what reward function uses (evaluator).
    """
    student: StudentPacket
    evaluator: EvaluatorPacket

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "student": self.student.to_dict(),
            "evaluator": self.evaluator.to_dict()
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DualPacket":
        """Create from dictionary."""
        return cls(
            student=StudentPacket.from_dict(data["student"]),
            evaluator=EvaluatorPacket.from_dict(data["evaluator"])
        )


# =============================================================================
# CoverageCell
# =============================================================================

@dataclass
class CoverageCell:
    """
    Identifies a cell in the coverage matrix.

    Coverage is tracked across:
    - domain: mathematical, empirical, common_sense, pop_science, philosophic
    - judgment_type: yes, no, insufficient
    - difficulty: 1-5
    - has_distractor: True/False
    """
    domain: str
    judgment_type: str
    difficulty: int
    has_distractor: bool

    @property
    def cell_id(self) -> str:
        """
        Compute deterministic cell ID.

        Format: {domain}_{judgment_type}_{difficulty}_{positive|distractor}
        """
        distractor_suffix = "distractor" if self.has_distractor else "positive"
        return f"{self.domain}_{self.judgment_type}_{self.difficulty}_{distractor_suffix}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["cell_id"] = self.cell_id
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoverageCell":
        """Create from dictionary (ignores cell_id if present)."""
        data = {k: v for k, v in data.items() if k != "cell_id"}
        return cls(**data)


# =============================================================================
# EnhancedJudgmentSample
# =============================================================================

@dataclass
class EnhancedJudgmentSample:
    """
    Extended judgment sample with all fields for training.

    Combines:
    - Core judgment fields (proposition, evidence, conditions, judgment)
    - Metadata (domain, difficulty)
    - Grounding data (evidence_citations)
    - Verification structure (verification_chain)
    - Contrastive data (distractor_type)
    """
    proposition: str
    domain: str
    evidence_text: str
    conditions: List[str]
    expected_judgment: str
    difficulty: int
    evidence_citations: List[Dict[str, Any]] = field(default_factory=list)
    verification_chain: Dict[str, Any] = field(default_factory=dict)
    distractor_type: Optional[str] = None

    def __post_init__(self):
        """Validate fields."""
        validate_domain(self.domain)
        validate_judgment(self.expected_judgment)
        validate_difficulty(self.difficulty)

    def to_dual_packet(self) -> DualPacket:
        """Convert to DualPacket for training."""
        student = StudentPacket(
            prompt=self.proposition,
            evidence=self.evidence_text,
            conditions=self.conditions
        )
        evaluator = EvaluatorPacket(
            expected_judgment=self.expected_judgment,
            domain=self.domain,
            difficulty=self.difficulty,
            evidence_citations=self.evidence_citations,
            distractor_type=self.distractor_type,
            verification_chain=self.verification_chain
        )
        return DualPacket(student=student, evaluator=evaluator)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedJudgmentSample":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# DistractorSample
# =============================================================================

@dataclass
class DistractorSample:
    """
    A contrastive sample pairing correct and distractor completions.

    Used for training the model to distinguish:
    - Correct judgment responses (P3 level)
    - Distractor responses (various error types)
    """
    correct_completion: str
    distractor_completion: str
    distractor_type: DistractorType
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Convert enum to string value
        result["distractor_type"] = self.distractor_type.value
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistractorSample":
        """Create from dictionary."""
        # Convert string to enum if needed
        dt = data["distractor_type"]
        if isinstance(dt, str):
            data = dict(data)
            data["distractor_type"] = DistractorType(dt)
        return cls(**data)


# =============================================================================
# Exports
# =============================================================================

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
