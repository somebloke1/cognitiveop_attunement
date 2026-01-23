# Compressed Documentation Index

Compressed versions of Lonergan-based documentation for selective inclusion in LLM system instructions.

## Design Principles

1. **Operational focus**: Extracted what's needed for evaluation, removed examples/elaboration
2. **Evaluation implications**: Each doc ends with concrete evaluation criteria
3. **Token-efficient**: ~3200 tokens total vs ~15000 in originals (~78% reduction)
4. **Selective inclusion**: Include only docs relevant to the evaluation mode

## Available Documents

| File | Tokens | Use When |
|------|--------|----------|
| `temporal_structure.md` | ~565 | Temporal reasoning is primary challenge |
| `counterpositions.md` | ~800 | Dialectical mode; pop science detection |
| `self_appropriation.md` | ~880 | Philosophic domain; performative consistency |
| `interpretation.md` | ~990 | Understanding expression structure; context reconstruction |
| `domain_schema.md` | ~1170 | Mode-specific evaluation criteria; domain differentiation |

**Total if all included**: ~4400 tokens

## Recommended Combinations by Mode

### FORMAL (mathematical, logical)
- Base synopsis only (already in system instruction)
- Optional: `domain_schema.md` for mode-specific criteria

### EMPIRICAL (scientific, observational)
- `temporal_structure.md` (evidence availability, currency)
- `domain_schema.md` (higher integration levels)
- Optional: `counterpositions.md` (for science vs pop science)

### COMMON_SENSE (practical, situational)
- `temporal_structure.md` (standpoint-relativity)
- `interpretation.md` (audience context reconstruction)
- `domain_schema.md` (varieties of common sense, interface problem)

### DIALECTICAL (philosophical, pop science)
- `counterpositions.md` (essential)
- `self_appropriation.md` (for philosophic domain)
- `domain_schema.md` (counterposition detection criteria)

## Source Mapping

| Compressed | Original |
|------------|----------|
| `temporal_structure.md` | `../temporal_structure_of_judgment.md` |
| `counterpositions.md` | `../counterpositions.md` |
| `self_appropriation.md` | `../self_appropriation.md` |
| `interpretation.md` | `../interpretation_and_inference.md` |
| `domain_schema.md` | `../principled_domain_schema.md` |

## Integration Notes

The base synopsis in `llm_evaluator.py` EVALUATOR_SYSTEM_INSTRUCTION already includes (~2645 tokens):
- 4-level cognitive pattern
- Virtually Unconditioned structure  
- Domain-specific fulfillment criteria
- Counterposition detection basics
- Artifact/artificer distinction

These compressed docs provide **deeper treatment** for specific evaluation contexts.

## Example: Building Mode-Specific System Instructions

```python
def get_system_instruction(mode: str) -> str:
    base = EVALUATOR_SYSTEM_INSTRUCTION  # ~2645 tokens
    
    if mode == "dialectical":
        base += "\n\n" + read_compressed("counterpositions.md")
        base += "\n\n" + read_compressed("self_appropriation.md")
    elif mode == "empirical":
        base += "\n\n" + read_compressed("temporal_structure.md")
    elif mode == "common_sense":
        base += "\n\n" + read_compressed("temporal_structure.md")
        base += "\n\n" + read_compressed("interpretation.md")
    
    return base  # ~3500-4500 tokens depending on mode
```
