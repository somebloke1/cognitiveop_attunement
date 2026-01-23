"""
Surface Analyzer: Local deterministic checks for judgment completions.

This module performs fast, local analysis of student completions to extract
surface-level features. These features are passed to the Gemini evaluator
to inform semantic analysis, and can also be used for quick filtering.

The surface analysis does NOT determine reward by itself - it provides
structured observations that the semantic evaluator interprets.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List

from .logging_config import get_eval_logger, log_function_call


@dataclass
class SurfaceAnalysis:
    """
    Surface-level analysis of a judgment completion.

    This is passed to Gemini along with the full completion for semantic evaluation.
    """

    # Judgment extraction
    judgment_extracted: Optional[str] = None  # "Yes", "No", "Insufficient", or None
    judgment_position: Optional[str] = None  # "early", "middle", "late", "absent"

    # Structural markers
    has_condition_language: bool = False
    condition_terms_found: List[str] = field(default_factory=list)

    has_fulfillment_language: bool = False
    fulfillment_terms_found: List[str] = field(default_factory=list)

    has_link_language: bool = False  # "therefore", "thus", "because", etc.
    link_terms_found: List[str] = field(default_factory=list)

    # Evidence grounding
    has_quotations: bool = False
    quotation_count: int = 0
    quotations_found: List[str] = field(default_factory=list)

    # Domain vocabulary
    domain_terms_found: List[str] = field(default_factory=list)

    # Basic metrics
    completion_length: int = 0
    sentence_count: int = 0

    # Potential issues (observations, not judgments)
    observations: List[str] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        """Format as a section for inclusion in Gemini prompt."""
        lines = ["## Surface Analysis (Local Pre-processing)"]

        lines.append(
            f"- Judgment extracted: {self.judgment_extracted or 'NONE DETECTED'}"
        )
        lines.append(f"- Judgment position: {self.judgment_position or 'N/A'}")
        lines.append(
            f"- Completion length: {self.completion_length} chars, ~{self.sentence_count} sentences"
        )

        lines.append(f"- Condition language present: {self.has_condition_language}")
        if self.condition_terms_found:
            lines.append(f"  Terms: {', '.join(self.condition_terms_found[:5])}")

        lines.append(f"- Fulfillment language present: {self.has_fulfillment_language}")
        if self.fulfillment_terms_found:
            lines.append(f"  Terms: {', '.join(self.fulfillment_terms_found[:5])}")

        lines.append(f"- Inferential connectives present: {self.has_link_language}")
        if self.link_terms_found:
            lines.append(f"  Terms: {', '.join(self.link_terms_found[:5])}")

        lines.append(f"- Evidence quotations: {self.quotation_count} found")

        if self.domain_terms_found:
            lines.append(
                f"- Domain vocabulary: {', '.join(self.domain_terms_found[:5])}"
            )

        if self.observations:
            lines.append("- Observations:")
            for obs in self.observations:
                lines.append(f"  * {obs}")

        return "\n".join(lines)


class SurfaceAnalyzer:
    """
    Performs fast local analysis of judgment completions.

    This analyzer extracts surface features without making semantic judgments.
    Its output informs the Gemini evaluator's deeper analysis.
    """

    # Judgment extraction patterns - try multiple formats
    # Pattern 1: "Judgment: Yes" or "Judgment is Yes" or "Judgment - Yes"
    JUDGMENT_PATTERN_EXPLICIT = re.compile(
        r"judgment(?:\s+is)?[\s:\-]+\s*(yes|no|insufficient)", re.IGNORECASE
    )
    # Pattern 2: Standalone at start of line or after newline: "Yes" / "No" / "Insufficient"
    JUDGMENT_PATTERN_STANDALONE = re.compile(
        r"(?:^|\n)\s*(yes|no|insufficient)\s*(?:\n|$|\.)", re.IGNORECASE
    )
    # Pattern 3: "conclude: Yes" or "conclusion: No" or "verdict: Insufficient" or "Affirm (Yes)"
    JUDGMENT_PATTERN_CONCLUSION = re.compile(
        r"(?:conclude|conclusion|verdict|answer|affirm)[\s:\-\(]+\s*(yes|no|insufficient)",
        re.IGNORECASE,
    )
    # Pattern 4: Semantic equivalents - "withhold judgment", "affirm the proposition", "deny"
    # Maps to: withhold -> Insufficient, affirm -> Yes, deny/falsify -> No
    JUDGMENT_PATTERN_SEMANTIC = re.compile(
        r"(?:therefore|thus|hence|so)?\s*,?\s*(?:we\s+)?"
        r"(withhold(?:\s+(?:the\s+)?judgment)?|"
        r"affirm(?:\s+(?:the\s+)?proposition)?|"
        r"deny(?:\s+(?:the\s+)?proposition)?|"
        r"cannot\s+(?:be\s+)?(?:affirm|determine|judge)(?:ed)?|"
        r"(?:evidence\s+is\s+)?insufficient)",
        re.IGNORECASE,
    )
    # Mapping from semantic terms to canonical judgments
    SEMANTIC_JUDGMENT_MAP = {
        "withhold": "Insufficient",
        "withhold judgment": "Insufficient",
        "withhold the judgment": "Insufficient",
        "affirm": "Yes",
        "affirm the proposition": "Yes",
        "deny": "No",
        "deny the proposition": "No",
        "cannot affirm": "Insufficient",
        "cannot determine": "Insufficient",
        "cannot judge": "Insufficient",
        "cannot be affirmed": "Insufficient",
        "cannot be determined": "Insufficient",
        "cannot be judged": "Insufficient",
        "evidence is insufficient": "Insufficient",
        "insufficient": "Insufficient",
    }

    # Condition-related terms
    CONDITION_TERMS = [
        "condition",
        "conditions",
        "criterion",
        "criteria",
        "requirement",
        "requirements",
        "prerequisite",
        "if",
        "when",
        "provided that",
        "given that",
    ]

    # Fulfillment-related terms
    FULFILLMENT_TERMS = [
        "fulfilled",
        "fulfillment",
        "satisfied",
        "met",
        "not met",
        "verified",
        "confirmed",
        "established",
        "demonstrated",
        "lacking",
        "missing",
        "absent",
        "insufficient",
    ]

    # Inferential connectives
    LINK_TERMS = [
        "therefore",
        "thus",
        "hence",
        "consequently",
        "because",
        "since",
        "as",
        "given",
        "it follows",
        "we can conclude",
        "this means",
        "the evidence shows",
        "this demonstrates",
    ]

    # Domain-specific vocabulary
    DOMAIN_VOCABULARY = {
        "mathematical": [
            "proof",
            "axiom",
            "theorem",
            "lemma",
            "corollary",
            "derive",
            "deduce",
            "follows",
            "definition",
            "qed",
            "by definition",
            "necessarily",
            "logically",
        ],
        "empirical": [
            "evidence",
            "data",
            "observation",
            "experiment",
            "measure",
            "measurement",
            "replicate",
            "verify",
            "hypothesis",
            "test",
            "sample",
            "control",
        ],
        "common_sense": [
            "practical",
            "situation",
            "context",
            "works",
            "experience",
            "typically",
            "usually",
            "common",
            "everyday",
            "obvious",
            "clearly",
        ],
        "pop_science": [
            "claimed",
            "alleged",
            "supposedly",
            "correlation",
            "unfounded",
            "unverified",
            "authority",
            "expert says",
            "study shows",
            "research suggests",
        ],
        "philosophic": [
            "coherent",
            "self-consistent",
            "performative",
            "presuppose",
            "presupposes",
            "counterposition",
            "self-defeating",
            "position",
            "contradiction",
        ],
    }

    @log_function_call
    def analyze(self, completion: str, domain: Optional[str] = None) -> SurfaceAnalysis:
        """
        Analyze a completion and return surface features.

        Args:
            completion: The student model's completion text
            domain: Optional domain hint for vocabulary matching

        Returns:
            SurfaceAnalysis with extracted features
        """
        eval_logger = get_eval_logger()
        eval_logger.log_surface_analysis_input(completion, domain or "unknown")

        if not completion:
            return SurfaceAnalysis(observations=["Empty completion"])

        analysis = SurfaceAnalysis()
        completion_lower = completion.lower()

        # Basic metrics
        analysis.completion_length = len(completion)
        analysis.sentence_count = len(re.findall(r"[.!?]+", completion))

        # Extract judgment
        analysis.judgment_extracted = self._extract_judgment(completion)
        analysis.judgment_position = self._get_judgment_position(completion)

        # Check condition language
        analysis.condition_terms_found = self._find_terms(
            completion_lower, self.CONDITION_TERMS
        )
        analysis.has_condition_language = len(analysis.condition_terms_found) > 0

        # Check fulfillment language
        analysis.fulfillment_terms_found = self._find_terms(
            completion_lower, self.FULFILLMENT_TERMS
        )
        analysis.has_fulfillment_language = len(analysis.fulfillment_terms_found) > 0

        # Check inferential connectives
        analysis.link_terms_found = self._find_terms(completion_lower, self.LINK_TERMS)
        analysis.has_link_language = len(analysis.link_terms_found) > 0

        # Extract quotations
        analysis.quotations_found = self._extract_quotations(completion)
        analysis.quotation_count = len(analysis.quotations_found)
        analysis.has_quotations = analysis.quotation_count > 0

        # Domain vocabulary
        if domain and domain.lower() in self.DOMAIN_VOCABULARY:
            vocab = self.DOMAIN_VOCABULARY[domain.lower()]
            analysis.domain_terms_found = self._find_terms(completion_lower, vocab)
        else:
            # Check all domains
            for dom, vocab in self.DOMAIN_VOCABULARY.items():
                found = self._find_terms(completion_lower, vocab)
                analysis.domain_terms_found.extend(found)

        # Generate observations
        analysis.observations = self._generate_observations(analysis)

        # Log output
        eval_logger.log_surface_analysis_output(analysis)

        return analysis

    def _extract_judgment(self, completion: str) -> Optional[str]:
        """Extract the judgment value from completion using multiple patterns."""
        # Try explicit pattern first (most reliable): "Judgment: Yes"
        matches = list(self.JUDGMENT_PATTERN_EXPLICIT.finditer(completion))
        if matches:
            return matches[-1].group(1).capitalize()

        # Try conclusion pattern: "conclude: Yes"
        matches = list(self.JUDGMENT_PATTERN_CONCLUSION.finditer(completion))
        if matches:
            return matches[-1].group(1).capitalize()

        # Try standalone pattern (less reliable, may catch false positives)
        matches = list(self.JUDGMENT_PATTERN_STANDALONE.finditer(completion))
        if matches:
            return matches[0].group(1).capitalize()  # Use first for standalone

        # Try semantic pattern: "withhold judgment", "affirm the proposition", etc.
        matches = list(self.JUDGMENT_PATTERN_SEMANTIC.finditer(completion))
        if matches:
            # Use the last semantic match (conclusion is usually at the end)
            semantic_text = matches[-1].group(1).lower().strip()
            # Look up in map, handling partial matches
            for key, value in self.SEMANTIC_JUDGMENT_MAP.items():
                if key in semantic_text or semantic_text in key:
                    return value
            # If "insufficient" is in there, it's Insufficient
            if "insufficient" in semantic_text:
                return "Insufficient"

        return None

    def _get_judgment_position(self, completion: str) -> Optional[str]:
        """Determine where in the completion the judgment appears."""
        # Try all patterns in order of reliability
        match = self.JUDGMENT_PATTERN_EXPLICIT.search(completion)
        if not match:
            match = self.JUDGMENT_PATTERN_CONCLUSION.search(completion)
        if not match:
            match = self.JUDGMENT_PATTERN_STANDALONE.search(completion)
        if not match:
            match = self.JUDGMENT_PATTERN_SEMANTIC.search(completion)
        if not match:
            return "absent"

        position = match.start() / len(completion)
        if position < 0.33:
            return "early"
        elif position < 0.66:
            return "middle"
        else:
            return "late"

    def _find_terms(self, text: str, terms: List[str]) -> List[str]:
        """Find which terms from a list appear in text."""
        found = []
        for term in terms:
            if term.lower() in text:
                found.append(term)
        return found

    def _extract_quotations(self, completion: str) -> List[str]:
        """Extract quoted text segments."""
        # Match both "..." and '...' quotations
        pattern = r'["\']([^"\']{3,})["\']'
        matches = re.findall(pattern, completion)
        return matches[:10]  # Limit to first 10

    def _generate_observations(self, analysis: SurfaceAnalysis) -> List[str]:
        """Generate observations about the completion."""
        observations = []

        # No judgment detected
        if analysis.judgment_extracted is None:
            observations.append("No explicit judgment marker detected")

        # Judgment at unusual position
        if analysis.judgment_position == "early":
            observations.append("Judgment appears early (before reasoning)")

        # Missing structural elements
        if not analysis.has_condition_language:
            observations.append("No condition-related language detected")

        if not analysis.has_fulfillment_language:
            observations.append("No fulfillment-related language detected")

        if not analysis.has_link_language:
            observations.append("No inferential connectives detected")

        # Very short completion
        if analysis.completion_length < 100:
            observations.append("Very short completion (<100 chars)")

        # No evidence grounding
        if not analysis.has_quotations:
            observations.append("No quotations from evidence detected")

        return observations
