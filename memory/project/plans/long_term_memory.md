---
name: Long-Term Memory System Implementation Plan
description: OpenViking-inspired embedding-based semantic memory — batch segmentation, fact extraction, retrieval injection into conversation context
type: project
---

# Long-Term Memory System (OpenViking-Inspired)

## Context

The application currently has a sliding window of ~15 recent messages for conversation context. There is no long-term memory — discussions from weeks ago are invisible to the LLM. This plan adds embedding-based semantic memory: a batch agent segments and summarizes older messages by topic, and at query time the most relevant summaries are injected before the sliding window. This gives personas access to historical context without ballooning the prompt with raw messages.

Design decisions from discussion (2026-04-02): Gemini Embedding API for embeddings (free tier, already in SDK deps, no local compute — tiny EC2 constraint), sliding-window cosine similarity for topic segmentation, fact extraction over prose summarization, SQLite-native storage (no vector DB), ambient channel messages surfaced to all personas. Memory block injected as `role: "user"` with `<memory>` tags. EmbeddingService abstracts the provider so a local embedding model (google/embeddinggemma-300m via Cloudflare tunnel or similar) can be swapped in later.

Further refinements from review (2026-04-03): Scope simplified — batch processing always segments at channel+persona granularity; MemoryMode fan-out happens at retrieval time only. Recency filter prevents overlap with sliding window. Model-name filtering ensures embedding compatibility across provider changes. Live processing uses in-memory transactions to prevent partial-failure data orphaning. Historical backfill deferred until local model infrastructure is ready.

### Key Design Rationale

- **Why embeddings over topic labels:** Topic string matching is fragile — "supply chain attack" and "cybersecurity" wouldn't match. Embedding similarity handles semantic proximity deterministically (dot product, no LLM in the retrieval loop).
- **Why sliding-window centroid segmentation:** Per-message embedding + consecutive similarity drops to detect topic boundaries. Sliding window centroid (mean of last N message embeddings) is strictly better than consecutive-pair comparison — absorbs low-content messages ("yeah sounds good") without spurious cuts.
- **Why max-similarity over full window for retrieval:** Conversations span multiple topics. Averaging the window into one embedding dilutes each topic. Instead, embed each message individually (1 batched API call), score each summary against every message, keep max. Each topic in the window gets its own retrieval vote.
- **Why fact extraction over prose summaries:** Individual facts are verifiable, composable, and robust — one bad fact doesn't corrupt others (unlike a prose summary where distortion propagates).
- **Why Gemini Embedding API (not local, initially):** App runs on tiny EC2 instance. sentence-transformers + torch is ~2GB and CPU-intensive. Gemini Embedding API free tier (100 RPM, 1K RPD) is sufficient for live processing, already in SDK deps, and offloads compute. Long-term plan: self-host google/embeddinggemma-300m locally for unlimited throughput and backfill capability.
- **Why channel-level segmentation (not MemoryMode-scoped):** Conversations happen in channels, not across channels simultaneously. Segmenting at channel+persona granularity during batch processing is the natural boundary. MemoryMode (CHANNEL_ISOLATED, SERVER_WIDE, PERSONAL, GLOBAL) determines which channels' summaries to pull at retrieval time. This separation means a MemoryMode change takes effect immediately with zero reprocessing — no need to re-segment or recalculate centroids.

---

## Phase 1: Foundation (Schema + Embedding Infrastructure)

### 1.1 New SQLite Tables

Add to `create_schema()` in `src/database/memory_manager.py` (inside the `schema_sql` executescript block, using `CREATE TABLE IF NOT EXISTS`):

**Message_Embeddings** — per-message vector, PK is FK to User_Interactions
```sql
(interaction_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL,
 model_name TEXT NOT NULL DEFAULT 'text-embedding-004', created_at TIMESTAMP NOT NULL,
 FOREIGN KEY (interaction_id) REFERENCES User_Interactions(interaction_id) ON DELETE CASCADE)
```

**Memory_Segments** — contiguous topic segments with source anchors, always scoped at channel+persona level
```sql
(segment_id INTEGER PK AUTOINCREMENT, channel TEXT NOT NULL,
 server_id TEXT, persona_name TEXT NOT NULL,
 start_interaction_id INTEGER NOT NULL, end_interaction_id INTEGER NOT NULL,
 message_count INTEGER NOT NULL, created_at TIMESTAMP NOT NULL)
+ INDEX idx_segment_channel_persona ON (channel, persona_name, server_id)
```
Segments are always channel-scoped. MemoryMode fan-out (SERVER_WIDE queries all channels in a server, GLOBAL queries all channels for a persona, etc.) happens at retrieval time via `retrieve_relevant_summaries`.

**Memory_Summaries** — extracted facts per segment, with summary-level embedding. Currently 1:1 with segments (one summary per segment). Kept as a separate table for future expansion (e.g., corrected summaries, meta-summaries from consolidation, or multi-strategy extraction).
```sql
(summary_id INTEGER PK AUTOINCREMENT, segment_id INTEGER NOT NULL FK UNIQUE,
 content TEXT NOT NULL, embedding BLOB NOT NULL, model_name TEXT NOT NULL, created_at TIMESTAMP NOT NULL)
+ INDEX idx_summary_segment ON (segment_id)
```
Note: `UNIQUE` on segment_id enforces 1:1 for now; can be relaxed later if multi-summary support is added.

No ALTER TABLE migration needed — all `CREATE TABLE IF NOT EXISTS`.

### 1.2 New File: `src/embedding_service.py`

Provider-abstracted embedding service. Default provider: Gemini Embedding API (`text-embedding-004`, 768 dims, free tier 100 RPM / 1K RPD). Designed to be swappable to a local embedding model (google/embeddinggemma-300m) or other providers.

```python
class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""
    @abstractmethod
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embedding vectors (normalized)."""
    @property
    @abstractmethod
    def model_name(self) -> str: ...
    @property
    @abstractmethod
    def dimensions(self) -> int: ...

class GeminiEmbeddingProvider(EmbeddingProvider):
    """Uses Google's text-embedding-004 via the existing google.generativeai SDK."""
    model_name = "text-embedding-004"
    dimensions = 768

    async def encode(self, texts: List[str]) -> List[List[float]]:
        # Uses google.generativeai.embed_content() via asyncio.to_thread
        # Batches supported natively by the API
        # Returns normalized vectors

class EmbeddingService:
    """High-level service: provider + similarity math + BLOB serialization."""

    def __init__(self, provider: Optional[EmbeddingProvider] = None)
    # Defaults to GeminiEmbeddingProvider if none provided

    async def encode(self, texts: List[str]) -> List[bytes]  # returns float32 BLOBs
    async def encode_single(self, text: str) -> bytes
    @staticmethod cosine_similarity(blob_a, blob_b) -> float
    @staticmethod cosine_similarities(query_blob, candidate_blobs) -> List[float]
    @property model_name -> str
    @property dimensions -> int
```

- Provider ABC enables future local server or OpenAI providers
- BLOBs are raw float32 bytes — no pickle/base64
- Cosine similarity is a dot product (vectors pre-normalized by provider)
- numpy used only in similarity methods (lightweight, already a transitive dep)
- Async interface since providers are API-based (not CPU-bound local models)
- Provider should document its max input token limit (e.g., 2048 for embeddinggemma-300m).
  EmbeddingService truncates input texts exceeding the provider's limit before encoding.
  This affects both batch processing (long messages) and retrieval (long assistant responses).

### 1.3 New MemoryManager Methods

```python
def store_message_embedding(self, interaction_id, embedding, model_name, created_at)

def get_unembedded_messages(self, persona_name, channel, server_id=None,
                            limit=200, model_name=None) -> List[Dict]
# limit governed by agent config batch_size (default 200)
# model_name filter: if provided, also returns messages whose existing embedding
# was computed by a different model (supports model migration — re-embed on switch)
# Excludes suppressed interactions (JOIN against Suppressed_Interactions)
# Excludes messages with NULL or empty content (not embeddable)

def store_segment(self, channel, server_id, persona_name,
                  start_id, end_id, message_count, created_at) -> int

def store_summary(self, segment_id, content, embedding, model_name, created_at) -> int

def get_summaries_for_channel(self, channel, persona_name, server_id=None,
                              exclude_after_interaction_id=None,
                              model_name=None) -> List[Dict]
# Low-level single-channel query. Used by retrieve_relevant_summaries (Phase 3)
# as a building block, and directly by the diagnostic script.
# exclude_after_interaction_id: if set, excludes summaries whose segment's
# start_interaction_id >= this value (recency filter, see Phase 3)
# model_name: if set, only returns summaries with matching embedding model

def get_active_channels(self, model_name=None) -> List[Tuple[str, str, Optional[str]]]
# Returns distinct (channel, persona_name, server_id) tuples that have
# unprocessed messages (LEFT JOIN Message_Embeddings WHERE embedding IS NULL)
# When model_name is provided, also returns channels where existing embeddings
# were computed by a different model (supports model migration discovery)

def get_last_segment_tail_embeddings(self, channel, persona_name,
                                     server_id=None, n=3,
                                     model_name=None) -> Optional[List[bytes]]
# Returns the last N message embeddings from the most recent segment in this
# channel+persona. Used by _seed_centroid_from_previous to avoid artificial
# topic splits at batch boundaries.
# Joins Memory_Segments → User_Interactions (filtered by channel+persona+server_id,
# not just ID range) → Message_Embeddings. Returns None if no previous segment.
# model_name: if provided, only returns embeddings from the matching model.
# On model migration, returns None (cold start) rather than seeding with
# incompatible vectors from the old model.
```

`get_unembedded_messages`: `User_Interactions LEFT JOIN Message_Embeddings WHERE embedding IS NULL`, filtered by (channel, persona_name, server_id). Includes `_SUPPRESSION_SUBQUERY` and `WHERE content IS NOT NULL AND content != ''` to exclude suppressed and non-embeddable messages. When `model_name` is provided, also includes messages where `Message_Embeddings.model_name != ?` (stale embeddings from a different model).

`get_active_channels`: `SELECT DISTINCT channel, persona_name, server_id FROM User_Interactions ui LEFT JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id WHERE me.embedding IS NULL`. Includes suppression and null-content exclusions. When `model_name` is provided, the WHERE clause becomes `WHERE me.embedding IS NULL OR me.model_name != ?` — ensuring channels with stale-model embeddings are also surfaced for reprocessing. Simple — no MemoryMode awareness needed.

### 1.4 Dependencies

No new heavy dependencies. `google-generativeai` is already installed (used by TextEngine for Gemini LLM calls). `numpy` is likely already a transitive dependency; add it to requirements explicitly if not.

### 1.5 Tests

In `tests/database/test_memory_manager.py`:
- Schema creates all three new tables with correct columns
- store/retrieve round-trips for embeddings, segments, summaries
- `get_unembedded_messages` returns only unembedded; with model_name filter, also returns stale-model messages; excludes suppressed and null-content messages
- Channel+persona filtering works correctly
- `get_active_channels` returns only channels with unprocessed messages; with model_name filter, also returns channels with stale-model embeddings; excludes suppressed and null-content
- `get_last_segment_tail_embeddings` returns correct embeddings, scoped to channel+persona (not just ID range); with model_name filter, returns None for stale-model segments
- `get_summaries_for_channel` recency filter excludes segments starting inside window
- Migration on legacy DB creates tables, is idempotent

In `tests/test_embedding_service.py` (new):
- Correct blob size (768 * 4 = 3072 bytes for Gemini text-embedding-004)
- Cosine similarity math with synthetic BLOBs (unit tests, no API calls)
- Batch similarity matches individual calls
- Provider ABC contract tests with a mock provider
- GeminiEmbeddingProvider tested via `@pytest.mark.llm_live` (real API, auto-skips without credentials)

**Files:** `src/database/memory_manager.py`, `src/embedding_service.py`, `tests/database/test_memory_manager.py`, `tests/test_embedding_service.py`

---

## Phase 2: Segmentation & Summarization Agent

### 2.1 New File: `src/agents/memory_agent.py`

```python
class MemoryAgent(Agent):
    agent_name = "memory"

    def __init__(self, chat_system, agent_config=None)
    # EmbeddingService initialized lazily on first deploy(), not at construction.
    # Prevents app startup failure if embedding API key isn't configured.
    # Provider selected from agent_config["embedding_provider"] (default: "gemini").
    # On initialization, sets chat_system._embedding_service to the same instance,
    # ensuring ChatSystem uses the identical provider for query-time embedding.
    # This guarantees model_name consistency between batch and retrieval paths.

    async def deploy(self):
        # Discover channels with unprocessed messages via get_active_channels(model_name=...)
        # For each (channel, persona_name, server_id) tuple:
        #   1. Fetch unembedded messages (batch_size from agent config)
        #   2. Skip if len(messages) < min_segment_size — too few messages to form a
        #      meaningful segment. They accumulate until the next cycle. Saves API calls
        #      and avoids low-quality summaries from trivial batches (e.g. "yeah sounds good").
        #   3. Process entirely in memory: embed -> segment -> summarize -> embed summary
        #   4. On success: write all results to DB in one transaction (embeddings, segments, summaries)
        #   5. On failure: log error, skip this channel, continue to next
        # This all-or-nothing approach prevents data orphaning (e.g., embeddings stored
        # but segmentation failed, leaving messages that appear "processed" but aren't)

    def _segment_by_similarity(self, messages, embeddings) -> List[segments]:
        # Sliding window centroid: maintain running mean embedding
        # Cut when cosine_similarity(centroid, new_msg) < threshold
        # Enforce min_segment_size
        # Config: similarity_threshold (default 0.3), min_segment_size (default 3)

    async def _summarize_segment(self, segment) -> Optional[Tuple[str, bytes]]:
        # Build transcript from segment messages, format:
        #   [user] Alice: Did you see the Python 3.13 release?
        #   [assistant] Sage: Yes, the JIT compiler is a significant improvement...
        #   [user] Bob: We should test against it
        # Uses author_role and author_name from User_Interactions. This attribution
        # enables the summarizer to produce facts like "Alice reported X" rather
        # than anonymous "someone mentioned X".
        # Call memory_summarizer persona via text_engine
        # Embed the extracted facts via EmbeddingService
        # Returns (facts_text, summary_embedding) — both stored together

    def _seed_centroid_from_previous(self, channel, persona_name, server_id=None):
        # Load the last N message embeddings from the most recent segment
        # in this channel+persona via get_last_segment_tail_embeddings(model_name=...)
        # Compute mean of tail embeddings and normalize to unit vector
        # This prevents artificial topic splits at batch boundaries
        # Returns None if: no previous segment, or previous segment used a different
        # embedding model (cold start on model migration)
```

**Segmentation algorithm:**
1. Seed centroid from tail of previous segment in this channel+persona (via `_seed_centroid_from_previous`): compute mean of tail embeddings and normalize to unit vector. If no previous segment exists (or model mismatch), use first message embedding as centroid. This overlap prevents artificial topic splits at batch boundaries — if the topic hasn't changed since last cycle, new messages extend the existing segment rather than starting a new one.
2. For each message (starting from first if seeded, or second if using first message as centroid), compute similarity to running centroid
3. If similarity < threshold AND current segment >= min_size: cut, reset centroid to current message embedding (clean break — no carryover from previous topic)
4. Else: update centroid incrementally `centroid = centroid * ((n-1)/n) + vec/n`, then re-normalize (`centroid /= np.linalg.norm(centroid)`). Re-normalization is required because the incremental mean produces a non-unit vector, and cosine similarity is implemented as a dot product (assumes unit vectors). Without it, the centroid's magnitude drifts and inflates similarity scores, making cuts progressively less likely in longer segments.
5. Final remaining messages form last segment

### 2.2 System Persona

Add `memory_summarizer` to `config/system_personas.json`:
- Model: Gemma 4 26B MoE or 31B dense (API initially, local via koboldcpp later — easily changed in agent config)
- Temperature: 0.0
- Prompt: fact extraction tool — bullet list of standalone factual statements
- No tools, no context history

### 2.3 Registration

In `src/main.py`: `agent_manager.register("memory", MemoryAgent)` — unconditional (no external service dependency).

### 2.4 Config in `config/agents.json`

```json
"memory": {
    "persona": "memory_summarizer",
    "schedule": {"interval": 900},
    "action_history_limit": 5,
    "auto_start": false,
    "embedding_provider": "gemini",
    "similarity_threshold": 0.3,
    "min_segment_size": 3,
    "batch_size": 200
}
```

### 2.5 Tests

In `tests/agents/test_memory_agent.py` (new):
- DI wiring and lazy embedding service
- Segmentation with synthetic embeddings: correct cuts, all-similar, min-size enforced, too-few messages
- Centroid seeding from previous segment tail: topic continuation extends rather than splits
- Centroid re-normalization: verify unit magnitude after incremental updates
- Summarization calls LLM correctly, embeds summary content, handles missing persona
- Deploy processes all channels, respects shutdown
- Deploy skips channels with fewer messages than min_segment_size
- Transaction model: nothing written to DB on partial failure (mock DB to verify)
- Agent config injection, batch_size passed through to query limit

Post-implementation: a standalone diagnostic script (`scripts/memory_diagnostics.py`) for threshold calibration and quality evaluation. Tentative design:

```
python -m scripts.memory_diagnostics [options]

Options:
  --channel general        # specific channel or all
  --limit 100              # how many messages to analyze
  --threshold 0.3          # override threshold to test alternatives
  --verbose                # full message content + per-message similarity scores
  --judge gemini-2.5-flash # run LLM evaluation on segmentation and summary quality
```

Default output: segment count, average segment size, similarity score distribution (mean/min/max), cut point locations. Verbose adds per-message centroid similarity and full summary content. Judge mode pipes verbose output into an evaluator LLM for structured quality ratings (segmentation accuracy, summary fidelity, missing/hallucinated facts). Useful after deployment, after model changes, and for comparing threshold values.

**Files:** `src/agents/memory_agent.py`, `config/system_personas.json`, `config/agents.json`, `src/main.py`, `tests/agents/test_memory_agent.py`

---

## Phase 3: Retrieval & Context Injection

### 3.1 New MemoryManager Method

```python
def retrieve_relevant_summaries(self, persona_name, channel, server_id=None,
                                user_identifier=None, memory_mode=MemoryMode.CHANNEL_ISOLATED,
                                include_ambient=True,
                                exclude_after_interaction_id=None,
                                model_name=None) -> List[Dict]:
    # MemoryMode determines which channels' summaries to pull:
    #   CHANNEL_ISOLATED → WHERE channel = ? AND persona_name = ?
    #   SERVER_WIDE      → WHERE server_id = ? AND persona_name = ?
    #   PERSONAL         → WHERE persona_name = ? AND channel IN
    #                       (SELECT DISTINCT channel FROM User_Interactions
    #                        WHERE user_identifier = ? AND persona_name = ?)
    #                       Note: intentionally cross-server — user's full history with
    #                       this persona regardless of which server. PERSONAL mode is
    #                       currently unused; revisit if privacy concerns arise.
    #   GLOBAL           → WHERE persona_name = ?
    #   TICKET_ISOLATED  → returns empty (no long-term memory for tickets)
    #
    # include_ambient=True adds a UNION with persona_name='ambient' using same scope
    #   (ambient messages are logged per-channel with persona_name='ambient' by Discord bot)
    #
    # exclude_after_interaction_id: recency filter — excludes summaries whose segment's
    #   start_interaction_id >= this value. Uses start (not end) to avoid gap where facts
    #   from messages just outside the sliding window are lost because their segment
    #   straddles the window boundary. Minor redundancy (segment tail overlaps window)
    #   is preferable to information loss. See "Recency Filter" section below.
    #
    # model_name: if provided, only returns summaries whose embedding was computed by
    #   this model. Prevents meaningless cross-model similarity comparisons after a
    #   provider switch. Old summaries become invisible until re-embedded during backfill.
    #
    # Dict keys: summary_id, content, embedding, segment_id, persona_name, channel
    #
    # Implementation: builds a single SQL query with appropriate WHERE/UNION for the
    # selected MemoryMode, rather than calling get_summaries_for_channel per-channel
    # (avoids N+1 queries for SERVER_WIDE/GLOBAL). get_summaries_for_channel remains
    # as a simpler convenience method for diagnostics and single-channel testing.
```

### 3.2 New ChatSystem Method: `_retrieve_memory_block`

```python
async def _retrieve_memory_block(self, persona, user_identifier, channel,
                                 server_id, conversation_history,
                                 oldest_interaction_id=None) -> Optional[str]:
    # 1. Return None early if: no embedding service, or feature disabled
    # 2. Fetch summaries via retrieve_relevant_summaries, passing:
    #    - persona's MemoryMode for scope fan-out
    #    - exclude_after_interaction_id=oldest_interaction_id (recency filter)
    #    - model_name=self._embedding_service.model_name (cross-model safety)
    #    - include_ambient=True (ambient channel messages)
    # 3. Return None early if no summaries — avoids unnecessary embedding API call
    # 4. Embed text content of conversation_history messages as a batch (1 API call)
    #    - Extract `content` field from each message dict
    #    - Skip non-text messages (images, tool results) — text-only embedding models
    #      can't produce meaningful vectors from these
    #    - Truncate very long messages to embedding model's context limit (2048 tokens
    #      for embeddinggemma-300m; check provider's limit) — long assistant responses
    #      dilute semantic signal when embedded as a single vector
    # 5. For each summary: score = max(cosine_sim(summary_emb, msg_emb) for msg_emb in window)
    #    This ensures multi-topic conversations retrieve memories for ALL topics discussed,
    #    not just an averaged-out blend. Each message votes independently.
    # 6. Take top-K by max-similarity score, format as memory block
```

ChatSystem has an `_embedding_service` attribute, set by MemoryAgent on its first deploy cycle. This ensures both batch processing and query-time retrieval use the same provider and model_name. If MemoryAgent has not run yet (or is not registered), `_embedding_service` is None and retrieval gracefully degrades — no memory block, sliding window only.

`oldest_interaction_id` is returned as metadata from `_fetch_raw_history` (the interaction_id of the oldest message in the sliding window). This enables the recency filter without an extra DB query.

**Why max-similarity over the full window:** A conversation may span multiple topics. Embedding the window into a single vector (mean/concat) dilutes each topic. Instead, embed each message individually in one batched API call, then score each candidate summary against every window message and keep the max. A "supply chain attack" summary scores high if any single message in the window touches cybersecurity — even if the rest is about lunch.

### Recency Filter

Prevents injecting summaries that overlap with the sliding window (which would waste context tokens on redundant information). The filter excludes summaries whose segment's `start_interaction_id >= oldest_interaction_id` in the sliding window.

**Why filter on `start` (not `end`):** A segment may straddle the window boundary — e.g., segment covers IDs 100-107, window starts at 105. Filtering on `end_interaction_id < 105` would exclude this segment entirely, losing facts from messages 100-104 (too old for the window, filtered from memory). Filtering on `start_interaction_id >= 105` instead keeps the segment: facts from 100-104 are preserved in the memory block, and the minor redundancy from 105-107 (visible both as raw messages and summary facts) is bounded by segment size and preferable to information loss.

```
Example:
  Segment A: IDs 95-99   (start=95,  fully outside window)  → ✅ included
  Segment B: IDs 100-107 (start=100, straddles window)      → ✅ included (minor overlap)
  Segment C: IDs 108-115 (start=108, fully inside window)   → ❌ filtered
  Sliding window: IDs 105-120
```

### 3.3 Inject in `_prepare_request`

After `_build_conversation_history()`, before appending user message:
```python
memory_block = await self._retrieve_memory_block(
    ctx.persona, ctx.user_identifier,
    ctx.channel, ctx.server_id, ctx.conversation_history,
    oldest_interaction_id=ctx.oldest_interaction_id  # from _fetch_raw_history metadata
)
if memory_block:
    ctx.conversation_history.insert(0, {"role": "user", "content": memory_block})
```

`_fetch_raw_history` is modified to also return the oldest interaction_id as metadata on the RequestContext, enabling the recency filter without an extra DB query. This requires adding `interaction_id` to the SELECT list in the underlying history queries (`get_channel_history`, `get_server_history`, `get_personal_history`, `get_global_history` in `memory_manager.py`).

Note: `_retrieve_memory_block` is async (embedding is an API call, not CPU-bound), so no `asyncio.to_thread` wrapper needed. The cosine similarity math is trivial and non-blocking.

### 3.4 Memory Block Format

```
<memory>
The following are relevant facts from previous conversations:

[#security-alerts, 2 days ago]
- Alice reported TeamPCP supply chain attack affecting pip packages
- Team decided to audit all third-party dependencies

[#general, ambient, 1 week ago]
- Security review found 3 unvetted transitive dependencies
</memory>
```

Entries are ordered by similarity score (most relevant first — LLM attention is strongest at the start of the block). Labels include channel name (from summary Dict's `channel` field), ambient tag if `persona_name='ambient'`, and relative timestamp (derived from `Memory_Segments.created_at`). Relative timestamps ("2 days ago", "3 weeks ago") give the LLM temporal context for judging relevance without implying false precision. Labels are applied at format time, not embedding time — no effect on similarity scoring.

Injected as `role: "user"` — works across all three providers without special handling.

### 3.5 Config

In `config/global_config.py`:
```python
MEMORY_RETRIEVAL_ENABLED = env("MEMORY_RETRIEVAL_ENABLED", "false") == "true"
MEMORY_MAX_SUMMARIES_IN_CONTEXT = int(env("MEMORY_MAX_SUMMARIES", "5"))
```

No query window size config needed — we always embed the full sliding window (it's 1 batched API call regardless of window size).

### 3.6 Tests

In `tests/test_chat_system.py`:
- `_prepare_request` injects memory block when present, skips when None
- Feature disabled -> returns None

In `tests/test_memory_retrieval.py` (new):
- Ranking by similarity with synthetic embeddings
- Ambient inclusion/exclusion (persona_name='ambient' union)
- MemoryMode fan-out: CHANNEL_ISOLATED returns one channel, SERVER_WIDE returns all channels in server, etc.
- Recency filter: segments fully outside window included, straddling included, fully inside excluded
- Model-name filtering: old-model summaries excluded
- Format structure: score-ordered, channel labels, ambient tag, relative timestamps
- Empty history / no summaries → None (no embedding API call made)

In `tests/integration/test_memory_modes.py`:
- End-to-end: store messages -> embed -> segment -> summarize -> verify injection in _prepare_request
- Recency filter integration: verify sliding window overlap doesn't create information gaps

**Files:** `src/database/memory_manager.py`, `src/chat_system.py`, `config/global_config.py`, `tests/test_chat_system.py`, `tests/test_memory_retrieval.py`, `tests/integration/test_memory_modes.py`

---

## Phase 4: Polish

- Per-persona `include_ambient_memory` field (default True) on Persona class, with save/load backward compat
- Memory block visible in existing `dump_context` output (it's in conversation_history, so it shows up in the API payload dump already)
- Dedicated memory dump command deferred to later iteration

**Files:** `src/persona.py`, `src/utils/save_utils.py`, `tests/test_persona.py`

---

## Implementation Order

```
Phase 1 (foundation)  -> Phase 2 (agent)     -> Phase 3 (retrieval) -> Phase 4 (polish)
        ^independent^    ^needs Phase 1^        ^needs Phase 1+2^     ^needs Phase 3^
```

Phases 1 and 2 can be merged and committed together. Phase 3 is the activation step. Phase 4 is optional polish.

---

## Risk Notes

- **Gemini Embedding rate limits**: free tier is 100 RPM, 1K RPD. Batch embed (1 call per agent cycle) is fine. Query-time embedding (1 call per user message) is the RPD bottleneck — a busy day with 500+ user messages would exhaust it. Self-hosting planned as long-term fix.
- **Gemma/summarization rate limits**: Gemma 4 26B/31B at 15 RPM, 1.5K RPD. At 15 RPM, a batch producing ~67 segments takes ~4.5 min — fits within 900s interval. The 1.5K RPD caps total daily summarization. Gemma 4 26B (MoE) is likely sufficient for structured fact extraction. Self-hosting via koboldcpp planned once Gemma 4 support is available.
- **Embedding API latency**: ~100-200ms per call at query time, acceptable on top of 1-5s LLM call
- **Backward compat**: all new schema is CREATE TABLE IF NOT EXISTS, all new config keys have defaults, MEMORY_RETRIEVAL_ENABLED defaults to false (opt-in)
- **Provider swappability**: EmbeddingProvider ABC allows swapping to local server (e.g., sentence-transformers behind Cloudflare tunnel) without changing any other code
- **Backfill of historical messages**: Deferred until local model infrastructure is ready. API rate limits (especially RPD) make bulk processing of existing conversation history impractical. The memory system will initially only process new messages going forward. Historical backfill can be done offline with a local embedding model and local Gemma once koboldcpp supports Gemma 4 (or an alternative local inference server is set up). The backfill process also serves as recovery from partial failures and as a re-processing tool for improving live memories with broader context.
- **Embedding model migration**: When switching providers (e.g., Gemini API → local embeddinggemma-300m), embeddings from different models are incomparable. Safety mechanism: `model_name` filtering in retrieval ensures only same-model embeddings are compared. Old summaries become invisible until re-embedded during backfill. The `get_unembedded_messages` model_name filter supports this — it returns messages whose embeddings are missing OR from a stale model.
- **Batch boundary seams**: Incremental processing creates artificial segment splits when a topic spans two batch cycles. Centroid seeding prevents spurious *cuts within* each batch, but cannot merge segments *across* batches. If topic A spans two cycles, you get two same-topic segments with overlapping summaries, both scoring high at retrieval and occupying multiple top-K slots. This reduces effective memory depth for ongoing topics. Accepted as a known limitation until the consolidation system merges adjacent same-topic segments.
- **Post-suppression memory persistence**: If a message is suppressed after it was already embedded and summarized, its facts persist in the summary. The suppression filter prevents future processing, but existing summaries are not retroactively cleaned. The consolidation/backfill system can re-process affected segments to remove suppressed content. Low priority — suppression is rare and the window between message and summarization (≤900s) limits exposure.
- **Similarity threshold calibration (0.3)**: Default is a starting guess. A standalone diagnostic script (`scripts/memory_diagnostics.py`) will be built post-implementation to dump similarity scores, segment boundaries, and summary content, with optional LLM judge for automated quality evaluation. Threshold and min_segment_size are configurable in agent config for easy tuning.

---

## Verification

1. **Unit tests**: `pytest tests/database/test_memory_manager.py tests/test_embedding_service.py tests/agents/test_memory_agent.py tests/test_memory_retrieval.py -v`
2. **Integration**: `pytest tests/integration/test_memory_modes.py -v`
3. **Startup wiring**: `pytest tests/integration/test_startup_wiring.py -v` (verify MemoryAgent registration)
4. **Manual**: Start app with `auto_start: true` on memory agent, send messages in Discord, wait for batch cycle, verify summaries in DB (`sqlite3 user_memory.db "SELECT * FROM Memory_Summaries"`), then check `dump_context` shows memory block
5. **Lint/type**: `flake8 src/ && mypy src/ --config-file mypy.ini`

---

## Future Work

Beyond the scope of this plan but informed by its design. These build on top of the Phase 1-4 foundation.

### Memory Consolidation

The current design produces one segment per topic per batch cycle. Long-running topics that span multiple cycles create duplicate segments with overlapping summaries, wasting top-K retrieval slots. A consolidation system would periodically:

- **Merge adjacent same-topic segments**: Detect segments in the same channel+persona with high inter-summary similarity (their embeddings are already stored — just compare). Merge into a single segment with a unified summary, updating start/end interaction_ids.
- **Re-summarize with broader context**: Live summaries are extracted from narrow batches (3-10 messages). Consolidated summaries can re-process the full merged segment transcript, producing higher-quality facts with more context. This is the "re-processing live memories" idea discussed during review.
- **Clean suppressed content**: Re-process segments that contain suppressed interaction_ids, producing updated summaries that exclude the suppressed messages.
- **Deduplicate facts across segments**: Different segments may independently extract the same fact (e.g., "team uses Python 3.13" appearing in multiple conversations). Consolidation could detect and merge these.

This is conceptually analogous to biological memory consolidation — episodic memories (individual conversation segments) are gradually transformed into more stable, abstracted representations. The current system only has the episodic layer.

### Memory Decay

Older memories that are never retrieved could be deprioritized or archived. Biological memory systems strengthen memories through recall — a retrieval counter or last-accessed timestamp on summaries would enable this. Summaries that haven't been retrieved in N days could be excluded from retrieval candidates, reducing the pool size and improving relevance. Not urgently needed while total summary count is small, but becomes important as the system accumulates months of history.

### Semantic Abstraction

Repeated patterns across many segments could be elevated to general knowledge. If the summarizer extracts "Alice handles security reviews" from 10 different segments, that's a stable fact about Alice, not an episodic memory of a specific conversation. A higher-level extraction pass could identify these recurring facts and store them as persistent persona knowledge, distinct from conversation-specific memories. This would give personas a more natural "understanding" of the people they interact with.

### Local Model Infrastructure

Multiple components are planned for local hosting once hardware/software support is ready:
- **Embedding**: google/embeddinggemma-300m — eliminates RPD limits, enables backfill
- **Summarization**: Gemma 4 via koboldcpp (or alternative) — eliminates RPM/RPD limits, enables bulk processing
- **Backfill**: One-time processing of all historical messages, plus model migration re-embedding

The EmbeddingProvider ABC and agent config model field are designed to make this transition a config change rather than a code change.
