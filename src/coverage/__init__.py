"""
Coverage Analyzer Module

Tracks multi-dimensional coverage for cognitive training data:
- Domain: 5 levels (mathematical, empirical, common_sense, pop_science, philosophic)
- Judgment: 3 levels (Yes, No, Insufficient)
- Difficulty: 5 levels (1-5)
- Distractor: 2 levels (True, False)

Total cells: 5 x 3 x 5 x 2 = 150
"""

from .analyzer import (
    build_coverage_matrix,
    find_coverage_gaps,
    get_coverage_percentage,
    get_most_sparse_cells,
    generate_target_specification,
    balance_batch,
    CoverageAnalyzer,
    CoverageDimension,
    CoverageGap,
)

__all__ = [
    "build_coverage_matrix",
    "find_coverage_gaps",
    "get_coverage_percentage",
    "get_most_sparse_cells",
    "generate_target_specification",
    "balance_batch",
    "CoverageAnalyzer",
    "CoverageDimension",
    "CoverageGap",
]
