# Evaluation Harnesses

Deep-dive evaluation tools for measuring system quality. Unlike the diagnostic scripts in `scripts/`, these are designed for systematic, repeatable measurement with structured output.

## Planned

- **Memory retrieval quality** — Given known queries, measure whether the right summaries surface (precision/recall against hand-labeled ground truth).
- **Summarization fidelity** — Compare source interactions against their summaries to detect information loss (e.g., missing keywords, over-generalization).
- **Segmentation coherence** — Score whether messages within a segment are topically related and whether topic boundaries are correctly detected.
