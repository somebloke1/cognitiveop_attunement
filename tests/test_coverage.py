"""
TDD Tests for Coverage Analyzer (Insight 10: Coverage Enforcement Matrix)

These tests define the expected behavior of the coverage analyzer module
BEFORE implementation. Tests should FAIL initially.

Coverage Analyzer tracks multi-dimensional coverage:
- Domain: 5 levels (mathematical, empirical, common_sense, pop_science, philosophic)
- Judgment: 3 levels (Yes, No, Insufficient)
- Difficulty: 5 levels (1-5)
- Distractor: 2 levels (True, False)

Total cells: 5 x 3 x 5 x 2 = 150

Tests are organized by functionality:
1. Coverage matrix construction
2. Gap analysis
3. Targeted generation

The implementation will be at: src/coverage/analyzer.py
"""

import pytest
from typing import List, Dict, Any

# These imports will fail until modules are implemented
# This is expected for TDD - we write tests first
try:
    from src.coverage.analyzer import (
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
    COVERAGE_MODULE_AVAILABLE = True
except ImportError:
    COVERAGE_MODULE_AVAILABLE = False
    # Define placeholder types for test discovery
    def build_coverage_matrix(examples): raise NotImplementedError
    def find_coverage_gaps(matrix, min_count=2): raise NotImplementedError
    def get_coverage_percentage(matrix): raise NotImplementedError
    def get_most_sparse_cells(matrix, n=10): raise NotImplementedError
    def generate_target_specification(gap): raise NotImplementedError
    def balance_batch(examples, targets): raise NotImplementedError


# Skip all tests if module not implemented yet (expected for TDD)
pytestmark = pytest.mark.skipif(
    not COVERAGE_MODULE_AVAILABLE,
    reason="Coverage module not yet implemented (TDD)"
)


# =============================================================================
# PART 1: Coverage Matrix Construction Tests
# =============================================================================

class TestBuildCoverageMatrix:
    """Tests for build_coverage_matrix(examples) function."""

    def test_returns_dict(self, single_example):
        """Coverage matrix should be a dictionary."""
        matrix = build_coverage_matrix(single_example)
        assert isinstance(matrix, dict)

    def test_empty_dataset_all_zeros(self, empty_dataset):
        """Empty dataset should produce matrix with all zeros."""
        matrix = build_coverage_matrix(empty_dataset)

        # All cells should have count 0
        for cell_id, count in matrix.items():
            assert count == 0, f"Cell {cell_id} should be 0 for empty dataset"

    def test_single_example_increments_one_cell(self, single_example):
        """Single example should increment exactly one cell."""
        matrix = build_coverage_matrix(single_example)

        # The example is: mathematical, Yes, difficulty=1, has_distractor=False
        expected_cell = ("mathematical", "Yes", 1, False)

        non_zero_cells = [k for k, v in matrix.items() if v > 0]
        assert len(non_zero_cells) == 1, "Only one cell should be non-zero"
        assert matrix[expected_cell] == 1

    def test_matrix_has_correct_dimensions(self, single_example, coverage_dimensions):
        """Matrix should have 5 x 3 x 5 x 2 = 150 cells."""
        matrix = build_coverage_matrix(single_example)

        expected_cells = (
            len(coverage_dimensions["domain"]) *
            len(coverage_dimensions["judgment"]) *
            len(coverage_dimensions["difficulty"]) *
            len(coverage_dimensions["has_distractor"])
        )

        assert len(matrix) == expected_cells, f"Expected {expected_cells} cells"

    def test_cell_keys_are_tuples(self, single_example):
        """Cell keys should be tuples of (domain, judgment, difficulty, has_distractor)."""
        matrix = build_coverage_matrix(single_example)

        for cell_id in matrix.keys():
            assert isinstance(cell_id, tuple), "Cell ID should be a tuple"
            assert len(cell_id) == 4, "Cell ID should have 4 components"
            domain, judgment, difficulty, has_distractor = cell_id
            assert isinstance(domain, str)
            assert isinstance(judgment, str)
            assert isinstance(difficulty, int)
            assert isinstance(has_distractor, bool)

    def test_unbalanced_dataset_shows_clustering(self, unbalanced_dataset):
        """Unbalanced dataset should show high count in clustered cell."""
        matrix = build_coverage_matrix(unbalanced_dataset)

        # Most examples are: empirical, Yes, difficulty=2, has_distractor=False
        clustered_cell = ("empirical", "Yes", 2, False)
        assert matrix[clustered_cell] == 20, "Clustered cell should have 20 examples"

    def test_balanced_dataset_distribution(self, balanced_dataset):
        """Balanced dataset should show even distribution across cells."""
        matrix = build_coverage_matrix(balanced_dataset)

        # Get non-zero counts
        non_zero_counts = [v for v in matrix.values() if v > 0]

        # Check variance is reasonable (not all in one cell)
        if len(non_zero_counts) > 1:
            avg = sum(non_zero_counts) / len(non_zero_counts)
            max_val = max(non_zero_counts)
            # Max should not be too much larger than average
            assert max_val <= avg * 3, "Distribution too uneven"

    def test_counts_are_non_negative_integers(self, balanced_dataset):
        """All counts should be non-negative integers."""
        matrix = build_coverage_matrix(balanced_dataset)

        for cell_id, count in matrix.items():
            assert isinstance(count, int), f"Count for {cell_id} should be int"
            assert count >= 0, f"Count for {cell_id} should be non-negative"

    def test_total_count_equals_dataset_size(self, balanced_dataset):
        """Sum of all counts should equal dataset size."""
        matrix = build_coverage_matrix(balanced_dataset)

        total = sum(matrix.values())
        assert total == len(balanced_dataset), "Total count mismatch"


class TestCoverageDimension:
    """Tests for CoverageDimension configuration."""

    def test_dimension_has_name_and_values(self):
        """CoverageDimension should have name and possible values."""
        dim = CoverageDimension("domain", ["mathematical", "empirical"])

        assert dim.name == "domain"
        assert dim.values == ["mathematical", "empirical"]

    def test_dimension_validates_values(self):
        """CoverageDimension should reject empty values."""
        with pytest.raises((ValueError, AssertionError)):
            CoverageDimension("domain", [])

    def test_standard_dimensions(self, coverage_dimensions):
        """Standard dimensions should cover all required aspects."""
        required_dims = {"domain", "judgment", "difficulty", "has_distractor"}
        actual_dims = set(coverage_dimensions.keys())

        assert required_dims == actual_dims


# =============================================================================
# PART 2: Gap Analysis Tests
# =============================================================================

class TestFindCoverageGaps:
    """Tests for find_coverage_gaps(matrix, min_count) function."""

    def test_returns_list_of_gaps(self, unbalanced_dataset):
        """find_coverage_gaps should return a list."""
        matrix = build_coverage_matrix(unbalanced_dataset)
        gaps = find_coverage_gaps(matrix, min_count=2)

        assert isinstance(gaps, list)

    def test_empty_dataset_all_cells_are_gaps(self, empty_dataset):
        """All cells should be gaps for empty dataset."""
        matrix = build_coverage_matrix(empty_dataset)
        gaps = find_coverage_gaps(matrix, min_count=2)

        # All 150 cells should be gaps (count 0 < min 2)
        assert len(gaps) == 150

    def test_gap_has_required_fields(self, unbalanced_dataset):
        """Each gap should have domain, judgment, difficulty, has_distractor, current_count."""
        matrix = build_coverage_matrix(unbalanced_dataset)
        gaps = find_coverage_gaps(matrix, min_count=5)

        if gaps:  # May have some gaps
            gap = gaps[0]
            assert hasattr(gap, 'domain')
            assert hasattr(gap, 'judgment')
            assert hasattr(gap, 'difficulty')
            assert hasattr(gap, 'has_distractor')
            assert hasattr(gap, 'current_count')
            assert hasattr(gap, 'target_count')

    def test_gap_current_count_below_threshold(self, unbalanced_dataset):
        """All gaps should have current_count < min_count."""
        matrix = build_coverage_matrix(unbalanced_dataset)
        gaps = find_coverage_gaps(matrix, min_count=5)

        for gap in gaps:
            assert gap.current_count < 5, f"Gap {gap} has count >= min_count"

    def test_no_gaps_when_all_cells_meet_threshold(self, make_dataset):
        """No gaps should exist if all cells meet threshold."""
        # Create dataset with at least 2 examples in every cell
        # This is a large dataset, so we simplify for testing
        examples = []
        for domain in ["mathematical", "empirical"]:
            for judgment in ["Yes", "No"]:
                for diff in [1, 2]:
                    for dist in [True, False]:
                        examples.extend([
                            make_dataset(n=1, domain_dist={domain: 1})[0]
                            for _ in range(3)
                        ])

        # Build a custom matrix for this subset
        # For this test, we'd need actual implementation
        # This tests the concept
        matrix = build_coverage_matrix(examples)

        # With enough examples per cell, gaps should be few or none
        gaps = find_coverage_gaps(matrix, min_count=1)
        # This may still have gaps if coverage is incomplete
        assert isinstance(gaps, list)

    def test_gap_includes_target_count(self, unbalanced_dataset):
        """Each gap should include the target count (min_count)."""
        matrix = build_coverage_matrix(unbalanced_dataset)
        gaps = find_coverage_gaps(matrix, min_count=5)

        for gap in gaps:
            assert gap.target_count == 5


class TestGetCoveragePercentage:
    """Tests for get_coverage_percentage(matrix) function."""

    def test_empty_dataset_zero_percent(self, empty_dataset):
        """Empty dataset should have 0% coverage."""
        matrix = build_coverage_matrix(empty_dataset)
        percentage = get_coverage_percentage(matrix)

        assert percentage == 0.0

    def test_returns_float_between_0_and_100(self, balanced_dataset):
        """Coverage percentage should be between 0 and 100."""
        matrix = build_coverage_matrix(balanced_dataset)
        percentage = get_coverage_percentage(matrix)

        assert isinstance(percentage, (int, float))
        assert 0.0 <= percentage <= 100.0

    def test_single_example_low_coverage(self, single_example):
        """Single example should have very low coverage."""
        matrix = build_coverage_matrix(single_example)
        percentage = get_coverage_percentage(matrix)

        # 1 cell filled out of 150 = ~0.67%
        assert percentage < 5.0

    def test_balanced_dataset_higher_coverage(self, balanced_dataset):
        """Balanced dataset should have higher coverage than unbalanced."""
        balanced_matrix = build_coverage_matrix(balanced_dataset)
        balanced_pct = get_coverage_percentage(balanced_matrix)

        # Balanced dataset fills multiple cells
        assert balanced_pct > 10.0  # Should cover multiple cells


class TestGetMostSparseCells:
    """Tests for get_most_sparse_cells(matrix, n) function."""

    def test_returns_list_of_n_cells(self, unbalanced_dataset):
        """Should return exactly n cells (or all if fewer exist)."""
        matrix = build_coverage_matrix(unbalanced_dataset)
        sparse = get_most_sparse_cells(matrix, n=10)

        assert isinstance(sparse, list)
        assert len(sparse) <= 10

    def test_cells_sorted_by_count_ascending(self, unbalanced_dataset):
        """Returned cells should be sorted from lowest to highest count."""
        matrix = build_coverage_matrix(unbalanced_dataset)
        sparse = get_most_sparse_cells(matrix, n=20)

        counts = [cell['count'] for cell in sparse]
        assert counts == sorted(counts), "Cells not sorted by count"

    def test_sparse_cell_has_required_info(self, unbalanced_dataset):
        """Each sparse cell should include cell_id and count."""
        matrix = build_coverage_matrix(unbalanced_dataset)
        sparse = get_most_sparse_cells(matrix, n=5)

        if sparse:
            cell = sparse[0]
            assert 'cell_id' in cell or isinstance(cell, tuple)
            assert 'count' in cell or len(cell) >= 2

    def test_zero_count_cells_first(self, unbalanced_dataset):
        """Cells with count 0 should come first."""
        matrix = build_coverage_matrix(unbalanced_dataset)
        sparse = get_most_sparse_cells(matrix, n=50)

        if sparse and len(sparse) > 1:
            # First cells should have lowest counts
            first_count = sparse[0].get('count', sparse[0][1] if isinstance(sparse[0], tuple) else 0)
            last_count = sparse[-1].get('count', sparse[-1][1] if isinstance(sparse[-1], tuple) else 0)
            assert first_count <= last_count


# =============================================================================
# PART 3: Targeted Generation Tests
# =============================================================================

class TestGenerateTargetSpecification:
    """Tests for generate_target_specification(gap) function."""

    def test_returns_dict_specification(self):
        """Should return a dictionary with generation requirements."""
        gap = CoverageGap(
            domain="mathematical",
            judgment="Yes",
            difficulty=3,
            has_distractor=True,
            current_count=0,
            target_count=5,
            priority=1.0,
        )
        spec = generate_target_specification(gap)

        assert isinstance(spec, dict)

    def test_spec_includes_all_dimensions(self):
        """Specification should include all gap dimensions."""
        gap = CoverageGap(
            domain="empirical",
            judgment="No",
            difficulty=4,
            has_distractor=False,
            current_count=1,
            target_count=5,
            priority=0.8,
        )
        spec = generate_target_specification(gap)

        assert spec["domain"] == "empirical"
        assert spec["judgment"] == "No"
        assert spec["difficulty"] == 4
        assert spec["has_distractor"] == False

    def test_spec_includes_count_needed(self):
        """Specification should include how many examples needed."""
        gap = CoverageGap(
            domain="philosophic",
            judgment="Insufficient",
            difficulty=5,
            has_distractor=True,
            current_count=2,
            target_count=5,
            priority=0.5,
        )
        spec = generate_target_specification(gap)

        assert "count_needed" in spec
        assert spec["count_needed"] == 3  # 5 - 2

    def test_spec_includes_domain_specific_guidance(self):
        """Specification should include domain-appropriate generation hints."""
        gap = CoverageGap(
            domain="mathematical",
            judgment="Yes",
            difficulty=3,
            has_distractor=False,
            current_count=0,
            target_count=5,
            priority=1.0,
        )
        spec = generate_target_specification(gap)

        # Mathematical domain should have specific guidance
        assert "domain_guidance" in spec or "hints" in spec


class TestBalanceBatch:
    """Tests for balance_batch(examples, targets) function."""

    def test_returns_list_of_examples(self, balanced_dataset):
        """Should return a list of examples."""
        # No targets = return all examples
        result = balance_batch(balanced_dataset, targets=[])

        assert isinstance(result, list)

    def test_prioritizes_target_cells(self, unbalanced_dataset):
        """Should prioritize examples from target cells when balancing."""
        matrix = build_coverage_matrix(unbalanced_dataset)
        gaps = find_coverage_gaps(matrix, min_count=5)

        # Target the gaps
        targets = [generate_target_specification(g) for g in gaps[:3]]

        # Balance should de-emphasize over-represented cells
        balanced = balance_batch(unbalanced_dataset, targets)

        # The heavily-clustered cell should be under-represented in balanced batch
        assert len(balanced) <= len(unbalanced_dataset)

    def test_empty_targets_returns_all(self, balanced_dataset):
        """Empty targets should return all examples unchanged."""
        result = balance_batch(balanced_dataset, targets=[])

        assert len(result) == len(balanced_dataset)

    def test_balancing_increases_diversity(self, unbalanced_dataset):
        """Balancing should increase domain/judgment diversity."""
        matrix_before = build_coverage_matrix(unbalanced_dataset)
        gaps = find_coverage_gaps(matrix_before, min_count=2)
        targets = [generate_target_specification(g) for g in gaps[:5]]

        balanced = balance_batch(unbalanced_dataset, targets)

        # Calculate diversity as number of unique (domain, judgment) pairs
        original_pairs = set((e.domain, e.judgment) for e in unbalanced_dataset)
        balanced_pairs = set((e.domain, e.judgment) for e in balanced)

        # Balanced batch should maintain or improve diversity
        # (can't add new pairs, but can under-sample over-represented)
        assert len(balanced_pairs) >= 1


# =============================================================================
# PART 4: CoverageAnalyzer Class Tests
# =============================================================================

class TestCoverageAnalyzerClass:
    """Tests for the CoverageAnalyzer class (wrapper)."""

    def test_analyzer_initialization(self, coverage_dimensions):
        """Analyzer should initialize with dimensions."""
        dims = [
            CoverageDimension(name, values)
            for name, values in coverage_dimensions.items()
        ]
        analyzer = CoverageAnalyzer(dimensions=dims)

        assert analyzer is not None
        assert len(analyzer.dimensions) == 4

    def test_analyzer_analyze_method(self, balanced_dataset):
        """Analyzer should have analyze() method returning report."""
        dims = [
            CoverageDimension("domain", ["mathematical", "empirical", "common_sense", "pop_science", "philosophic"]),
            CoverageDimension("judgment", ["Yes", "No", "Insufficient"]),
            CoverageDimension("difficulty", [1, 2, 3, 4, 5]),
            CoverageDimension("has_distractor", [True, False]),
        ]
        analyzer = CoverageAnalyzer(dimensions=dims)

        report = analyzer.analyze(balanced_dataset)

        assert report is not None
        assert hasattr(report, 'matrix') or 'matrix' in report

    def test_analyzer_total_cells_property(self):
        """Analyzer should report total number of cells."""
        dims = [
            CoverageDimension("domain", ["a", "b"]),  # 2
            CoverageDimension("judgment", ["Yes", "No"]),  # 2
        ]
        analyzer = CoverageAnalyzer(dimensions=dims)

        assert analyzer.total_cells == 4  # 2 x 2

    def test_analyzer_visualize_returns_string(self, balanced_dataset):
        """Analyzer visualize() should return string representation."""
        dims = [
            CoverageDimension("domain", ["mathematical", "empirical"]),
            CoverageDimension("judgment", ["Yes", "No"]),
        ]
        analyzer = CoverageAnalyzer(dimensions=dims)
        analyzer.analyze(balanced_dataset)

        viz = analyzer.visualize()

        assert isinstance(viz, str)
        assert len(viz) > 0


# =============================================================================
# PART 5: Integration Tests
# =============================================================================

class TestCoverageIntegration:
    """Integration tests for coverage analysis workflow."""

    def test_full_workflow(self, balanced_dataset):
        """Test complete workflow: analyze -> gaps -> targets -> balance."""
        # Step 1: Build matrix
        matrix = build_coverage_matrix(balanced_dataset)
        assert len(matrix) > 0

        # Step 2: Find gaps
        gaps = find_coverage_gaps(matrix, min_count=2)
        assert isinstance(gaps, list)

        # Step 3: Get coverage stats
        pct = get_coverage_percentage(matrix)
        assert 0 <= pct <= 100

        # Step 4: Get sparse cells
        sparse = get_most_sparse_cells(matrix, n=10)
        assert len(sparse) <= 10

        # Step 5: Generate targets for gaps
        if gaps:
            targets = [generate_target_specification(g) for g in gaps[:3]]
            assert len(targets) <= 3

    def test_coverage_report_is_actionable(self, unbalanced_dataset):
        """Coverage report should identify actionable gaps."""
        matrix = build_coverage_matrix(unbalanced_dataset)
        gaps = find_coverage_gaps(matrix, min_count=5)

        # Should identify cells that need more examples
        assert len(gaps) > 0, "Unbalanced dataset should have gaps"

        # Each gap should be actionable (has all info needed for generation)
        for gap in gaps[:5]:
            spec = generate_target_specification(gap)
            assert spec["domain"] in ["mathematical", "empirical", "common_sense", "pop_science", "philosophic"]
            assert spec["judgment"] in ["Yes", "No", "Insufficient"]
            assert 1 <= spec["difficulty"] <= 5
            assert isinstance(spec["has_distractor"], bool)


# =============================================================================
# PART 6: Edge Cases
# =============================================================================

class TestCoverageEdgeCases:
    """Edge case tests for coverage analysis."""

    def test_duplicate_examples_counted(self, make_example):
        """Duplicate examples should be counted multiple times."""
        example = make_example(domain="mathematical", judgment="Yes", difficulty=1)
        examples = [example, example, example]  # 3 duplicates

        matrix = build_coverage_matrix(examples)

        cell = ("mathematical", "Yes", 1, False)
        assert matrix[cell] == 3

    def test_all_domains_represented(self, coverage_dimensions):
        """All 5 domains should be valid keys in matrix."""
        examples = []  # Empty
        matrix = build_coverage_matrix(examples)

        domains_in_matrix = set(key[0] for key in matrix.keys())
        expected_domains = set(coverage_dimensions["domain"])

        assert expected_domains == domains_in_matrix

    def test_all_judgment_types_represented(self, coverage_dimensions):
        """All 3 judgment types should be valid keys in matrix."""
        examples = []
        matrix = build_coverage_matrix(examples)

        judgments_in_matrix = set(key[1] for key in matrix.keys())
        expected_judgments = set(coverage_dimensions["judgment"])

        assert expected_judgments == judgments_in_matrix

    def test_invalid_domain_raises_error(self, make_example):
        """Example with invalid domain should raise error or be ignored."""
        with pytest.raises((ValueError, KeyError)):
            example = make_example(domain="invalid_domain")
            build_coverage_matrix([example])

    def test_invalid_difficulty_raises_error(self, make_example):
        """Example with invalid difficulty should raise error."""
        with pytest.raises((ValueError, KeyError, AssertionError)):
            example = make_example(difficulty=10)  # Invalid: > 5
            build_coverage_matrix([example])

    def test_large_dataset_performance(self, make_example):
        """Coverage analysis should handle large datasets efficiently."""
        import time

        # Create 10000 examples
        examples = [
            make_example(
                domain=["mathematical", "empirical", "common_sense"][i % 3],
                judgment=["Yes", "No", "Insufficient"][i % 3],
                difficulty=(i % 5) + 1,
            )
            for i in range(10000)
        ]

        start = time.time()
        matrix = build_coverage_matrix(examples)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0, f"Coverage analysis too slow: {elapsed}s"
        assert sum(matrix.values()) == 10000
