"""
Evidence Grounding Module

Implements textual reversion for judgment verification. Every judgment must
cite specific textual evidence from the input, operationalizing Lonergan's
principle that "fulfillment is found by REVERTING from formulation to the
more rudimentary state."

Functions:
- extract_citations: Find quoted text in completions
- extract_paraphrases: Find close paraphrases of input in completions
- citation_exists_in_input: Check for exact citation match
- citation_approximately_in_input: Check for fuzzy citation match
- citation_is_relevant: Check if citation is relevant to conditions
- compute_evidence_score: Compute overall evidence grounding quality
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Tuple, Optional


@dataclass
class EvidenceCitation:
    """A citation extracted from a completion."""
    text: str
    start_idx: int
    end_idx: int
    is_valid: bool = False
    is_relevant: bool = False
    similarity_score: float = 0.0


def extract_citations(completion: str) -> List[str]:
    """
    Extract quoted text from a completion.

    Args:
        completion: The completion text to extract citations from.

    Returns:
        List of quoted strings found in the completion.
    """
    if not completion:
        return []

    # Match text within double quotes, handling multiline
    # Use non-greedy matching to handle multiple quotes
    pattern = r'"([^"]+)"'
    matches = re.findall(pattern, completion, re.DOTALL)

    # Filter out empty strings
    citations = [m.strip() for m in matches if m.strip()]

    return citations


def extract_paraphrases(
    completion: str,
    input_text: str,
    threshold: float = 0.7
) -> List[Tuple[str, str, float]]:
    """
    Extract segments from completion that paraphrase the input text.

    Uses sliding window comparison to find segments of the completion
    that are similar to segments of the input text.

    Args:
        completion: The completion text to analyze.
        input_text: The original input text to compare against.
        threshold: Minimum similarity score to consider a paraphrase (0.0-1.0).

    Returns:
        List of tuples (completion_segment, input_segment, similarity_score).
    """
    if not completion or not input_text:
        return []

    paraphrases = []

    # Normalize whitespace for comparison
    completion_normalized = ' '.join(completion.split())
    input_normalized = ' '.join(input_text.split())

    # Split into sentences for comparison
    completion_sentences = _split_into_sentences(completion_normalized)
    input_sentences = _split_into_sentences(input_normalized)

    if not completion_sentences or not input_sentences:
        return []

    # Compare each completion sentence against each input sentence
    for comp_sent in completion_sentences:
        if len(comp_sent) < 10:  # Skip very short segments
            continue

        best_match = None
        best_score = 0.0

        for inp_sent in input_sentences:
            if len(inp_sent) < 10:
                continue

            # Compute similarity using multiple methods
            # 1. Sequence similarity
            seq_score = _compute_similarity(comp_sent.lower(), inp_sent.lower())

            # 2. Word overlap (Jaccard similarity)
            word_score = _compute_word_overlap(comp_sent.lower(), inp_sent.lower())

            # 3. Number overlap bonus
            number_bonus = _compute_number_overlap(comp_sent, inp_sent)

            # Combined score with emphasis on word overlap for paraphrases
            score = max(seq_score, word_score * 0.9) + number_bonus * 0.15

            if score > best_score and score >= threshold:
                best_score = score
                best_match = inp_sent

        if best_match is not None:
            paraphrases.append((comp_sent, best_match, best_score))

    return paraphrases


def _compute_word_overlap(s1: str, s2: str) -> float:
    """Compute similarity based on word overlap including synonyms."""
    if not s1 or not s2:
        return 0.0

    # Extract significant words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
                  'from', 'as', 'and', 'or', 'but', 'if', 'that', 'which',
                  'who', 'whom', 'this', 'these', 'those', 'it', 'its'}

    words1 = set(re.findall(r'\b\w+\b', s1.lower())) - stop_words
    words2 = set(re.findall(r'\b\w+\b', s2.lower())) - stop_words

    if not words1 or not words2:
        return 0.0

    # Common synonym/paraphrase mappings for research/study contexts
    synonym_groups = [
        {'study', 'research', 'investigation', 'experiment', 'trial'},
        {'participants', 'subjects', 'individuals', 'people', 'patients'},
        {'examined', 'studied', 'investigated', 'analyzed', 'tracked', 'followed'},
        {'showed', 'demonstrated', 'revealed', 'indicated', 'found'},
        {'month', 'months'},
        {'six', '6'},
        {'period', 'duration', 'time', 'span'},
        {'over', 'during', 'across'},
        {'results', 'findings', 'outcomes'},
        {'significant', 'significance', 'notable', 'meaningful'},
    ]

    # Expand words with synonyms
    def expand_with_synonyms(word_set):
        expanded = set(word_set)
        for word in word_set:
            for group in synonym_groups:
                if word in group:
                    expanded.update(group)
        return expanded

    words1_expanded = expand_with_synonyms(words1)
    words2_expanded = expand_with_synonyms(words2)

    # Check direct intersection
    direct_intersection = len(words1 & words2)

    # Check synonym-expanded intersection
    expanded_intersection = len(words1 & words2_expanded) + len(words2 & words1_expanded)

    # Combined intersection (avoiding double counting)
    total_matches = direct_intersection + (expanded_intersection - direct_intersection) * 0.8

    # Use smaller set size for denominator to be more generous
    min_size = min(len(words1), len(words2))

    return total_matches / min_size if min_size > 0 else 0.0


def _compute_number_overlap(s1: str, s2: str) -> float:
    """Compute overlap of numbers between two strings."""
    nums1 = set(re.findall(r'\d+', s1))
    nums2 = set(re.findall(r'\d+', s2))

    if not nums1 or not nums2:
        return 0.0

    intersection = len(nums1 & nums2)
    return min(1.0, intersection / max(len(nums1), len(nums2)))


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting on common delimiters
    sentences = re.split(r'[.!?]\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _compute_similarity(s1: str, s2: str) -> float:
    """Compute similarity between two strings using SequenceMatcher."""
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1, s2).ratio()


def citation_exists_in_input(citation: str, input_text: str) -> bool:
    """
    Check if a citation exists exactly in the input text.

    Performs case-insensitive matching with normalized whitespace.

    Args:
        citation: The citation string to find.
        input_text: The input text to search in.

    Returns:
        True if the citation exists in the input, False otherwise.
    """
    if not citation or not input_text:
        return False

    # Normalize whitespace for both
    citation_normalized = ' '.join(citation.split()).lower()
    input_normalized = ' '.join(input_text.split()).lower()

    return citation_normalized in input_normalized


def citation_approximately_in_input(
    citation: str,
    input_text: str,
    threshold: float = 0.85
) -> bool:
    """
    Check if a citation approximately matches text in the input.

    Uses fuzzy matching to find close matches above the threshold.
    IMPORTANT: Numbers must match exactly - "96%" does not approximately match "95%".

    Args:
        citation: The citation string to find.
        input_text: The input text to search in.
        threshold: Minimum similarity score (0.0-1.0).

    Returns:
        True if a close match is found, False otherwise.
    """
    if not citation or not input_text:
        return False

    # First check exact match
    if citation_exists_in_input(citation, input_text):
        return True

    # Normalize
    citation_normalized = ' '.join(citation.split()).lower()
    input_normalized = ' '.join(input_text.split()).lower()

    # CRITICAL: Check for numeric mismatch - numbers must match exactly
    # Extract all numbers from citation
    citation_numbers = set(re.findall(r'\d+\.?\d*', citation_normalized))
    input_numbers = set(re.findall(r'\d+\.?\d*', input_normalized))

    # If citation has numbers that don't appear in input, it's a fabrication
    if citation_numbers and not citation_numbers.issubset(input_numbers):
        # The citation contains numbers not in the input - likely hallucinated
        return False

    # Use sliding window to find best match
    citation_len = len(citation_normalized)
    best_score = 0.0

    # Try matching against windows of similar size
    window_sizes = [citation_len - 10, citation_len, citation_len + 10, citation_len + 20]

    for window_size in window_sizes:
        if window_size <= 0 or window_size > len(input_normalized):
            continue

        for i in range(len(input_normalized) - window_size + 1):
            window = input_normalized[i:i + window_size]
            score = _compute_similarity(citation_normalized, window)
            best_score = max(best_score, score)

            if best_score >= threshold:
                return True

    # Also try sentence-level matching
    input_sentences = _split_into_sentences(input_normalized)
    for sent in input_sentences:
        score = _compute_similarity(citation_normalized, sent.lower())
        if score >= threshold:
            return True

    return best_score >= threshold


def citation_is_relevant(citation: str, conditions: List[str]) -> bool:
    """
    Check if a citation is relevant to any of the given conditions.

    Uses keyword overlap and semantic matching to determine relevance.

    Args:
        citation: The citation string to check.
        conditions: List of condition strings.

    Returns:
        True if the citation is relevant to at least one condition.
    """
    if not conditions:
        # With no conditions, any citation is vacuously relevant
        return True

    if not citation:
        return False

    citation_lower = citation.lower()
    citation_words = set(re.findall(r'\b\w+\b', citation_lower))

    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
                  'from', 'as', 'into', 'through', 'during', 'before', 'after',
                  'above', 'below', 'between', 'under', 'again', 'further',
                  'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                  'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
                  'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                  'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
                  'until', 'while', 'although', 'though', 'that', 'which', 'who',
                  'whom', 'this', 'these', 'those', 'it', 'its'}

    citation_keywords = citation_words - stop_words

    # Check relevance to each condition
    for condition in conditions:
        condition_lower = condition.lower()
        condition_words = set(re.findall(r'\b\w+\b', condition_lower))
        condition_keywords = condition_words - stop_words

        # Check for keyword overlap
        overlap = citation_keywords & condition_keywords
        if overlap:
            return True

        # Check for numeric overlap (e.g., "100" in citation and "100" in condition)
        citation_numbers = set(re.findall(r'\d+\.?\d*', citation_lower))
        condition_numbers = set(re.findall(r'\d+\.?\d*', condition_lower))
        if citation_numbers & condition_numbers:
            return True

        # Check for semantic proximity via key concept matching
        # Special handling for domain-specific terms
        domain_concepts = {
            # Temperature/measurement
            ('degrees', 'celsius', 'temperature', 'fahrenheit'):
                {'degrees', 'celsius', 'temperature', 'fahrenheit', 'heat', 'thermal'},
            # Pressure
            ('pressure', 'atmospheric', 'kpa', 'atm'):
                {'pressure', 'atmospheric', 'kpa', 'atm', 'standard'},
            # Statistical
            ('correlation', 'significant', 'p-value', 'statistical'):
                {'correlation', 'significant', 'significance', 'p', 'value', 'statistical'},
            # Acidity
            ('ph', 'acidity', 'acid', 'acidic'):
                {'ph', 'acidity', 'acid', 'acidic', 'alkaline', 'base'},
        }

        for concept_group, related_terms in domain_concepts.items():
            # Check if citation and condition share semantic domain
            citation_in_domain = bool(citation_keywords & related_terms)
            condition_in_domain = bool(condition_keywords & related_terms)
            if citation_in_domain and condition_in_domain:
                return True

    return False


def compute_evidence_score(
    completion: str,
    input_text: str,
    conditions: List[str]
) -> float:
    """
    Compute the overall evidence grounding score for a completion.

    The score reflects how well the completion grounds its claims in the input text.

    Scoring:
    - Full credit (1.0): Relevant citations present and valid
    - Partial credit (~0.5): Citations present but not all relevant
    - Zero credit (0.0): No citations or all hallucinated

    Args:
        completion: The completion text to evaluate.
        input_text: The original input text (evidence).
        conditions: List of conditions being evaluated.

    Returns:
        Float score between 0.0 and 1.0.
    """
    if not completion:
        return 0.0

    if not input_text:
        # Cannot have valid citations without evidence
        return 0.0

    # Extract citations from completion
    citations = extract_citations(completion)

    if not citations:
        # Check for paraphrases as fallback
        paraphrases = extract_paraphrases(completion, input_text, threshold=0.6)
        if paraphrases:
            # Paraphrases get partial credit
            avg_similarity = sum(p[2] for p in paraphrases) / len(paraphrases)
            # Scale: paraphrases can get at most 0.7 score
            return min(0.7, avg_similarity * 0.7)
        return 0.0

    # Evaluate each citation
    valid_citations = []
    relevant_citations = []
    hallucinated_count = 0

    for citation in citations:
        # Check if citation exists in input (exact or approximate)
        is_valid = (citation_exists_in_input(citation, input_text) or
                   citation_approximately_in_input(citation, input_text, threshold=0.85))

        if is_valid:
            valid_citations.append(citation)
            # Check relevance to conditions
            if citation_is_relevant(citation, conditions):
                relevant_citations.append(citation)
        else:
            hallucinated_count += 1

    # Calculate score components
    total_citations = len(citations)
    valid_ratio = len(valid_citations) / total_citations if total_citations > 0 else 0
    relevant_ratio = len(relevant_citations) / total_citations if total_citations > 0 else 0

    # Penalize hallucinations heavily
    hallucination_ratio = hallucinated_count / total_citations if total_citations > 0 else 0

    # If all citations are hallucinated, return very low score
    if valid_ratio == 0:
        return 0.0

    # Scoring breakdown:
    # - Base for having valid citations: 0.35 * valid_ratio
    # - Bonus for relevant citations: 0.35 * relevant_ratio
    # - Bonus for having any valid citations: 0.15
    # - Bonus for having any relevant citations: 0.15
    # - Hallucination penalty applied multiplicatively

    base_score = (valid_ratio * 0.35 + relevant_ratio * 0.35)

    # Add bonus for having valid citations
    if valid_citations:
        base_score += 0.15

    # Add bonus for having relevant citations
    if relevant_citations:
        base_score += 0.15

    # Apply hallucination penalty (reduces score proportionally)
    # Heavy penalty: each hallucinated citation reduces score significantly
    penalized_score = base_score * (1 - hallucination_ratio * 0.7)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, penalized_score))
