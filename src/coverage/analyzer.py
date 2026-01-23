"""
Coverage Analyzer for Cognitive Training Data

Implements Insight 10: Coverage Enforcement Matrix

Tracks multi-dimensional coverage across:
- Domain: All domains from principled schema (Mode → Level → Variety)
- Judgment: 3 levels (Yes, No, Insufficient)
- Difficulty: 5 levels (1-5)
- Distractor: 2 levels (True, False)

Uses the principled domain schema from docs/principled_domain_schema.md
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from itertools import product

# Import from principled schema
from src.domains import (
    ALL_DOMAINS,
    PRINCIPLED_DOMAINS,
    get_all_domain_names,
    get_principled_domain,
    get_principled_prompt_context,
    FulfillmentMode,
    MODE_SPECS,
)


# =============================================================================
# Constants - Derived from Principled Schema
# =============================================================================


def _get_valid_domains() -> Set[str]:
    """Get all valid domain names from the principled schema (35 domains)."""
    return set(ALL_DOMAINS.keys())


# Dynamic domain set - all 35 principled domains
VALID_DOMAINS = _get_valid_domains()
VALID_JUDGMENTS = {"Yes", "No", "Insufficient"}
VALID_DIFFICULTIES = {1, 2, 3, 4, 5}
VALID_DISTRACTORS = {True, False}


def _build_domain_guidance() -> Dict[str, Dict[str, Any]]:
    """
    Build domain-specific guidance from the principled schema.
    """
    guidance = {}

    # Build from principled schema
    for name, spec in PRINCIPLED_DOMAINS.items():
        mode_spec = MODE_SPECS.get(spec.mode)
        guidance[name] = {
            "description": spec.description,
            "condition_types": [],  # Could be derived if needed
            "hints": spec.characteristic_pitfalls[:3]
            if spec.characteristic_pitfalls
            else [],
            "mode": spec.mode.value,
            "evidential_standards": spec.evidential_standards,
        }

    # All domains are now in PRINCIPLED_DOMAINS, so no need for separate loop
    # (The above loop already covers all 35 domains)

    return guidance


# Domain-specific guidance - built dynamically from schema
DOMAIN_GUIDANCE = _build_domain_guidance()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CoverageDimension:
    """A dimension in the coverage matrix (e.g., domain, judgment type)."""

    name: str
    values: List[Any]

    def __post_init__(self):
        if not self.values:
            raise ValueError(
                f"CoverageDimension '{self.name}' must have at least one value"
            )

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class CoverageGap:
    """Represents a gap in coverage that needs more examples."""

    domain: str
    judgment: str
    difficulty: int
    has_distractor: bool
    current_count: int
    target_count: int
    priority: float = 1.0

    @property
    def cell_id(self) -> Tuple[str, str, int, bool]:
        return (self.domain, self.judgment, self.difficulty, self.has_distractor)

    @property
    def count_needed(self) -> int:
        return max(0, self.target_count - self.current_count)


@dataclass
class CoverageReport:
    """Complete coverage analysis report."""

    matrix: Dict[Tuple[str, str, int, bool], int]
    gaps: List[CoverageGap]
    coverage_percentage: float
    total_cells: int
    filled_cells: int
    total_examples: int


# =============================================================================
# Core Functions
# =============================================================================


def _get_all_cells() -> List[Tuple[str, str, int, bool]]:
    """Generate all possible cells in the coverage matrix."""
    cells = []
    for domain in sorted(VALID_DOMAINS):
        for judgment in sorted(VALID_JUDGMENTS):
            for difficulty in sorted(VALID_DIFFICULTIES):
                for has_distractor in [False, True]:
                    cells.append((domain, judgment, difficulty, has_distractor))
    return cells


def _validate_example(example: Any) -> None:
    """
    Validate that an example has valid dimension values.

    Accepts all 35 domains from the principled schema (ALL_DOMAINS).
    """
    # Get domain - handle both dict and object
    if isinstance(example, dict):
        domain = example.get("domain")
        judgment = example.get("judgment")
        difficulty = example.get("difficulty")
    else:
        domain = getattr(example, "domain", None)
        judgment = getattr(example, "judgment", None)
        difficulty = getattr(example, "difficulty", None)

    # Refresh valid domains in case they were updated
    valid_domains = _get_valid_domains()

    if domain not in valid_domains:
        raise ValueError(
            f"Invalid domain: {domain}. "
            f"Must be one of the {len(valid_domains)} valid domains. "
            f"Sample: {list(valid_domains)[:5]}..."
        )

    if judgment not in VALID_JUDGMENTS:
        raise ValueError(
            f"Invalid judgment: {judgment}. Must be one of {VALID_JUDGMENTS}"
        )

    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(
            f"Invalid difficulty: {difficulty}. Must be one of {VALID_DIFFICULTIES}"
        )


def _example_to_cell(example: Any) -> Tuple[str, str, int, bool]:
    """Extract cell key from an example (dict or object)."""
    if isinstance(example, dict):
        domain = example["domain"]
        judgment = example["judgment"]
        difficulty = example["difficulty"]
        has_distractor = example.get("has_distractor", False)
    else:
        domain = example.domain
        judgment = example.judgment
        difficulty = example.difficulty
        has_distractor = getattr(example, "has_distractor", False)

    return (domain, judgment, difficulty, has_distractor)


def build_coverage_matrix(examples: List[Any]) -> Dict[Tuple[str, str, int, bool], int]:
    """
    Build a coverage matrix from a list of examples.

    Args:
        examples: List of example dicts/objects with domain, judgment, difficulty, has_distractor

    Returns:
        Dict mapping cell_id tuples to counts
        Cell format: (domain, judgment, difficulty, has_distractor)
    """
    # Initialize all cells to 0
    matrix = {cell: 0 for cell in _get_all_cells()}

    # Count examples in each cell
    for example in examples:
        _validate_example(example)
        cell = _example_to_cell(example)
        matrix[cell] += 1

    return matrix


def find_coverage_gaps(
    matrix: Dict[Tuple[str, str, int, bool], int], min_count: int = 2
) -> List[CoverageGap]:
    """
    Find cells with fewer examples than the minimum threshold.

    Args:
        matrix: Coverage matrix from build_coverage_matrix
        min_count: Minimum examples required per cell

    Returns:
        List of CoverageGap objects for cells below threshold
    """
    gaps = []

    for cell_id, count in matrix.items():
        if count < min_count:
            domain, judgment, difficulty, has_distractor = cell_id

            # Calculate priority: lower count = higher priority
            # Also boost priority for rare combinations
            priority = 1.0 - (count / min_count) if min_count > 0 else 1.0

            gap = CoverageGap(
                domain=domain,
                judgment=judgment,
                difficulty=difficulty,
                has_distractor=has_distractor,
                current_count=count,
                target_count=min_count,
                priority=priority,
            )
            gaps.append(gap)

    # Sort by priority (highest first)
    gaps.sort(key=lambda g: (-g.priority, g.current_count, g.domain))

    return gaps


def get_coverage_percentage(
    matrix: Dict[Tuple[str, str, int, bool], int], expected_cells: int = 150
) -> float:
    """
    Calculate the percentage of cells that have at least one example.

    Args:
        matrix: Coverage matrix from build_coverage_matrix
        expected_cells: Total expected cells (default 150 = 5x3x5x2)

    Returns:
        Float percentage 0.0 to 100.0
    """
    if expected_cells == 0:
        return 0.0

    filled_cells = sum(1 for count in matrix.values() if count > 0)
    return (filled_cells / expected_cells) * 100.0


def get_most_sparse_cells(
    matrix: Dict[Tuple[str, str, int, bool], int], n: int = 10
) -> List[Dict[str, Any]]:
    """
    Get the n cells with the lowest counts.

    Args:
        matrix: Coverage matrix from build_coverage_matrix
        n: Number of cells to return

    Returns:
        List of dicts with 'cell_id' and 'count', sorted ascending by count
    """
    # Sort cells by count
    sorted_cells = sorted(matrix.items(), key=lambda x: (x[1], x[0]))

    # Take top n
    sparse = []
    for cell_id, count in sorted_cells[:n]:
        sparse.append(
            {
                "cell_id": cell_id,
                "count": count,
            }
        )

    return sparse


def generate_target_specification(gap: CoverageGap) -> Dict[str, Any]:
    """
    Generate a specification for creating examples to fill a coverage gap.

    Args:
        gap: CoverageGap object describing what's needed

    Returns:
        Dict with generation requirements and domain-specific hints
    """
    guidance = DOMAIN_GUIDANCE.get(gap.domain, {})

    spec = {
        "domain": gap.domain,
        "judgment": gap.judgment,
        "difficulty": gap.difficulty,
        "has_distractor": gap.has_distractor,
        "count_needed": gap.count_needed,
        "priority": gap.priority,
        "domain_guidance": guidance.get("description", ""),
        "hints": guidance.get("hints", []),
        "condition_types": guidance.get("condition_types", []),
    }

    return spec


def balance_batch(examples: List[Any], targets: List[Dict[str, Any]]) -> List[Any]:
    """
    Balance a batch of examples by down-sampling over-represented cells.

    Args:
        examples: List of examples to balance
        targets: List of target specifications from generate_target_specification

    Returns:
        Modified list with better balance across cells
    """
    if not targets:
        return list(examples)

    # Build matrix to see current distribution
    matrix = build_coverage_matrix(examples)

    # Find the maximum count we want (based on targets)
    target_cells = set()
    for target in targets:
        cell = (
            target["domain"],
            target["judgment"],
            target["difficulty"],
            target["has_distractor"],
        )
        target_cells.add(cell)

    # Calculate a reasonable cap: e.g., 2x the average non-zero count
    non_zero_counts = [c for c in matrix.values() if c > 0]
    if non_zero_counts:
        avg_count = sum(non_zero_counts) / len(non_zero_counts)
        cap = max(1, int(avg_count * 2))
    else:
        cap = 1

    # Down-sample over-represented cells
    cell_counts = {cell: 0 for cell in matrix.keys()}
    balanced = []

    for example in examples:
        cell = _example_to_cell(example)

        # Always include examples from target cells (under-represented)
        if cell in target_cells:
            balanced.append(example)
            cell_counts[cell] += 1
        # Cap non-target cells
        elif cell_counts[cell] < cap:
            balanced.append(example)
            cell_counts[cell] += 1

    return balanced


# =============================================================================
# CoverageAnalyzer Class
# =============================================================================


class CoverageAnalyzer:
    """
    High-level coverage analyzer for training data.

    Wraps the core functions into a convenient class interface.
    """

    def __init__(
        self,
        dimensions: Optional[List[CoverageDimension]] = None,
        examples: Optional[List[Any]] = None,
    ):
        """
        Initialize the analyzer.

        Args:
            dimensions: List of CoverageDimension objects defining the matrix
            examples: Optional initial examples to analyze
        """
        if dimensions is None:
            # Use default dimensions
            dimensions = [
                CoverageDimension("domain", list(VALID_DOMAINS)),
                CoverageDimension("judgment", list(VALID_JUDGMENTS)),
                CoverageDimension("difficulty", list(VALID_DIFFICULTIES)),
                CoverageDimension("has_distractor", list(VALID_DISTRACTORS)),
            ]

        self.dimensions = dimensions
        self._matrix: Optional[Dict[Tuple, int]] = None
        self._report: Optional[CoverageReport] = None

        if examples:
            self.analyze(examples)

    @property
    def total_cells(self) -> int:
        """Total number of cells in the coverage matrix."""
        total = 1
        for dim in self.dimensions:
            total *= len(dim)
        return total

    def _build_matrix_for_dimensions(self, examples: List[Any]) -> Dict[Tuple, int]:
        """Build matrix using configured dimensions."""
        # Generate all cells from dimensions
        all_values = [dim.values for dim in self.dimensions]
        all_cells = list(product(*all_values))

        # Map dimension names for extraction
        dim_names = [dim.name for dim in self.dimensions]

        # Initialize matrix
        matrix = {cell: 0 for cell in all_cells}

        # Count examples
        for example in examples:
            cell_values = []
            for name in dim_names:
                if isinstance(example, dict):
                    val = example.get(name)
                else:
                    val = getattr(example, name, None)
                cell_values.append(val)

            cell = tuple(cell_values)
            if cell in matrix:
                matrix[cell] += 1

        return matrix

    def analyze(self, examples: List[Any]) -> CoverageReport:
        """
        Analyze coverage of the given examples.

        Args:
            examples: List of examples to analyze

        Returns:
            CoverageReport with full analysis
        """
        # Build matrix using standard function if dimensions match default
        if len(self.dimensions) == 4:
            # Try standard matrix first
            try:
                self._matrix = build_coverage_matrix(examples)
            except ValueError:
                # Fall back to custom dimensions
                self._matrix = self._build_matrix_for_dimensions(examples)
        else:
            self._matrix = self._build_matrix_for_dimensions(examples)

        # Calculate stats
        filled_cells = sum(1 for count in self._matrix.values() if count > 0)
        total_examples = sum(self._matrix.values())
        coverage_pct = (
            (filled_cells / self.total_cells) * 100.0 if self.total_cells > 0 else 0.0
        )

        # Find gaps
        gaps = []
        min_count = 2  # Default threshold
        for cell_id, count in self._matrix.items():
            if count < min_count and len(cell_id) == 4:
                domain, judgment, difficulty, has_distractor = cell_id
                gap = CoverageGap(
                    domain=domain,
                    judgment=judgment,
                    difficulty=difficulty,
                    has_distractor=has_distractor,
                    current_count=count,
                    target_count=min_count,
                )
                gaps.append(gap)

        self._report = CoverageReport(
            matrix=self._matrix,
            gaps=gaps,
            coverage_percentage=coverage_pct,
            total_cells=self.total_cells,
            filled_cells=filled_cells,
            total_examples=total_examples,
        )

        return self._report

    def get_matrix(self) -> Dict[Tuple, int]:
        """Get the coverage matrix."""
        if self._matrix is None:
            raise RuntimeError("No analysis performed yet. Call analyze() first.")
        return self._matrix

    def get_gaps(self, min_count: int = 2) -> List[CoverageGap]:
        """Get coverage gaps."""
        if self._matrix is None:
            raise RuntimeError("No analysis performed yet. Call analyze() first.")
        return find_coverage_gaps(self._matrix, min_count)

    def get_report(self) -> CoverageReport:
        """Get the full coverage report."""
        if self._report is None:
            raise RuntimeError("No analysis performed yet. Call analyze() first.")
        return self._report

    def visualize(self) -> str:
        """
        Generate a text visualization of coverage.

        Returns:
            String representation of the coverage matrix
        """
        if self._matrix is None:
            return "No analysis performed yet."

        lines = []
        lines.append("=" * 60)
        lines.append("Coverage Analysis Report")
        lines.append("=" * 60)

        if self._report:
            lines.append(f"Total cells: {self._report.total_cells}")
            lines.append(f"Filled cells: {self._report.filled_cells}")
            lines.append(f"Coverage: {self._report.coverage_percentage:.1f}%")
            lines.append(f"Total examples: {self._report.total_examples}")
            lines.append("")

            # Show top sparse cells
            sparse = get_most_sparse_cells(self._matrix, n=10)
            if sparse:
                lines.append("Most sparse cells:")
                for cell in sparse[:5]:
                    lines.append(f"  {cell['cell_id']}: {cell['count']}")

        lines.append("=" * 60)

        return "\n".join(lines)
