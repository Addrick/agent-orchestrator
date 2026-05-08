---
name: Keep orphan/outlier re-segmentation behavior
description: Decision to retain outlier_ids in summarizer rather than removing orphan re-queueing — Adam wants to refine, not eliminate
type: project
---

## Decision (2026-04-13)

Keep the `outlier_ids` mechanism in the memory summarizer tool. When the LLM marks messages as thematic outliers, they remain with `parent_summary_id=NULL` and get re-queued for future segmentation.

**Why:** Adam explicitly rejected removing outlier detection. The orphan re-segmentation produces garbage segments (SEGs 788-795 in the 2026-04-12 audit were all orphan junk), but Adam wants to refine the approach rather than accept the data loss of never re-processing outliers. Potential future refinements: quality gating on orphans, minimum content length, requiring Q/A pairs.

**How to apply:** Do not remove `outlier_ids` from the tool schema or suppress orphan re-processing without discussing with Adam first. Improvements should focus on making orphan segments better, not eliminating them.
