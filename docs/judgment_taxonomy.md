# Taxonomy of Judgment Classes

A systematic account of the types of judgment derived from Lonergan's *Insight*, organized for training data generation and evaluation.

## Primary Classification: By Domain

### 1. Mathematical Judgment
**Structure**: Conditions fulfilled in intelligibility itself
**Fulfillment**: Internal coherence, proof validity
**No reversion to sense data**

Subtypes:
- **Proof judgment**: Is this proof valid?
- **Theorem judgment**: Is this theorem true (given axioms)?
- **Definition judgment**: Is this definition coherent?
- **Impossibility judgment**: Is this construction impossible?

### 2. Empirical Scientific Judgment
**Structure**: Conditions fulfilled by reversion to sensible data
**Fulfillment**: Experimental verification, observation
**Requires reversion to sense data**

Subtypes:
- **Measurement judgment**: Does the measurement confirm X?
- **Law judgment**: Is this correlation verified?
- **Causal judgment**: Does X cause Y? (requires more than correlation)
- **Replication judgment**: Do results replicate?

### 3. Common Sense Judgment
**Structure**: Conditions fulfilled in practical situation
**Fulfillment**: "It works" - pragmatic success
**Descriptive, relates things to us**

Subtypes:
- **Practical judgment**: Will this work?
- **Situation judgment**: What's going on here?
- **Character judgment**: Is this person trustworthy?
- **Timing judgment**: Is now the right time?

### 4. Pop Science / Common Nonsense Detection
**Structure**: Form of scientific judgment without substance
**Detection**: Conditions CLAIMED but NOT fulfilled

Subtypes:
- **Authority appeal**: "Scientists say..." without evidence
- **Correlation-causation conflation**: Pattern claimed as mechanism
- **Vocabulary mimicry**: Scientific terms without understanding
- **Premature certainty**: Preliminary findings as established fact

### 5. Philosophic Judgment
**Structure**: Conditions fulfilled in performative self-consistency
**Fulfillment**: Can be held coherently by one who grasps it?

Subtypes:
- **Position judgment**: Coherent with intelligent inquiry
- **Counterposition judgment**: Self-defeating on affirmation
- **Dialectical judgment**: Which interpretation invites development?
- **Metaphysical judgment**: What is the nature of X?

---

## Secondary Classification: By Probabilistic Structure

### 1. Ideal Frequency Judgment
**Question**: Is this probability theoretically sound?
**Example**: "P(heads) = 0.5 for fair coin"
**Fulfillment**: In conceptual coherence (what "fair" means)
**Type**: Mathematical/definitional

### 2. Actual Frequency Judgment
**Question**: What does the data show?
**Example**: "In 1000 flips, we observed 487 heads"
**Fulfillment**: By measurement
**Type**: Empirical observation

### 3. Statistical Judgment (Divergence)
**Question**: Is divergence systematic or nonsystematic?
**Example**: "Is 487/1000 within random variation of 0.5?"
**Fulfillment**: Whether pattern is intelligible
**Type**: Meta-judgment on randomness

### 4. Probabilistic Causal Judgment
**Question**: Does X increase probability of Y?
**Requires**: Controlled comparison, ruling out confounds
**Type**: Empirical + methodological

---

## Tertiary Classification: By Judgment Act

### 1. Affirmation (Yes)
**Conditions**: All fulfilled
**Grasp**: The virtually unconditioned IS grasped
**Commitment**: "It is so"

### 2. Denial (No)
**Conditions**: At least one identifiably NOT fulfilled
**Grasp**: The unconditioned CANNOT be grasped (conditions fail)
**Commitment**: "It is not so"

### 3. Withholding (Insufficient)
**Conditions**: Cannot determine fulfillment
**Grasp**: Neither affirmation nor denial possible
**Stance**: Epochē - suspension of judgment

---

## Cross-Cutting: Vulnerability Themes

Each judgment can be vulnerable at any cognitive level:

### Attention Vulnerability
- Overlooked data
- Selective attention
- Missing context
- Incomplete presentation

### Understanding Vulnerability
- Conceptual ambiguity
- Equivocation
- Category errors
- Ill-formed questions

### Judgment Vulnerability
- Logical gaps
- Unfulfilled conditions claimed fulfilled
- Hasty generalization
- Self-defeating claims

### Decision Vulnerability
- Unactionable knowledge
- Irresponsible application
- Values conflict
- Context-dependence of action

---

## Implementation Priority

### Phase 1 (Current)
- [x] Domain-specific judgment (5 domains)
- [x] Affirmation/Denial/Withholding distinction
- [x] Counterposition detection
- [x] Pop science detection
- [x] Four-level invulnerability verification

### Phase 2 (Future)
- [ ] Probabilistic judgment subtypes
- [ ] Statistical judgment (systematic vs nonsystematic)
- [ ] Causal judgment with confound detection
- [ ] Temporal judgment (when to judge vs when to gather more)

### Phase 3 (Advanced)
- [ ] Dialectical judgment (position vs counterposition development)
- [ ] Higher viewpoint detection (when current framework insufficient)
- [ ] Genetic method (tracking development through text)
- [ ] Universal viewpoint approximation

---

## Operational Anchors by Judgment Type

| Judgment Type | Greek | Hebrew | Sanskrit | Key Distinction |
|---------------|-------|--------|----------|-----------------|
| Mathematical | *apodeixis* (demonstration) | *chokhmah* (wisdom) | *pramana* (valid cognition) | Necessity |
| Empirical | *empeiria* (experience) | *da'at* (knowledge) | *pratyaksha* (perception) | Verification |
| Common Sense | *phronesis* (practical wisdom) | *sekhel* (common sense) | *yukti* (reasoning) | Pragmatic |
| Philosophic | *episteme* (scientific knowing) | *tevunah* (insight) | *jnana* (knowledge) | Self-consistency |
| Probabilistic | *eikos* (likely) | *rov* (majority) | *sambhavana* (probability) | Frequency |

---

## Test Case Generation Strategy

For comprehensive coverage, generate test cases that cross:
- 5 domains × 3 judgment types × 4 vulnerability levels = 60 cells
- Add probabilistic variants for empirical domain: +9 cells
- Add dialectical variants for philosophic domain: +6 cells

Total: ~75 distinct case types for comprehensive coverage.
