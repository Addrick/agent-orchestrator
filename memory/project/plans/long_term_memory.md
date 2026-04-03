---
name: Long-Term Memory System Implementation Plan
description: OpenViking-inspired embedding-based semantic memory — batch segmentation, fact extraction, retrieval injection into conversation context
type: project
---

# Long-Term Memory System (OpenViking-Inspired)

## Context

The application currently has a sliding window of ~15 recent messages for conversation context. There is no long-term memory — discussions from weeks ago are invisible to the LLM. This plan adds embedding-based semantic memory: a batch agent segments and summarizes older messages by topic, and at query time the most relevant summaries are injected before the sliding window. This gives personas access to historical context without ballooning the prompt with raw messages.

Design decisions from discussion (2026-04-02): Gemini Embedding API for embeddings (free tier, already in SDK deps, no local compute — tiny EC2 constraint), sliding-window cosine similarity for topic segmentation, fact extraction over prose summarization, SQLite-native storage (no vector DB), ambient channel messages surfaced to all personas. Memory block injected as `role: "user"` with `<memory>` tags. EmbeddingService abstracts the provider so a local embedding server (via Cloudflare tunnel) can be swapped in later.

### Key Design Rationale

- **Why embeddings over topic labels:** Topic string matching is fragile — "supply chain attack" and "cybersecurity" wouldn't match. Embedding similarity handles semantic proximity deterministically (dot product, no LLM in the retrieval loop).
- **Why sliding-window centroid segmentation:** Per-message embedding + consecutive similarity drops to detect topic boundaries. Sliding window centroid (mean of last N message embeddings) is strictly better than consecutive-pair comparison — absorbs low-content messages ("yeah sounds good") without spurious cuts.
- **Why max-similarity over full window for retrieval:** Conversations span multiple topics. Averaging the window into one embedding dilutes each topic. Instead, embed each message individually (1 batched API call), score each summary against every message, keep max. Each topic in the window gets its own retrieval vote.
- **Why fact extraction over prose summaries:** Individual facts are verifiable, composable, and robust — one bad fact doesn't corrupt others (unlike a prose summary where distortion propagates).
- **Why Gemini Embedding API (not local):** App runs on tiny EC2 instance. sentence-transformers + torch is ~2GB and CPU-intensive. Gemini Embedding is free tier (1500 RPM), already in SDK deps, and offloads compute.

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

**Memory_Segments** — contiguous topic segments with source anchors
```sql
(segment_id INTEGER PK AUTOINCREMENT, scope_type TEXT NOT NULL CHECK(...),
 scope_key TEXT NOT NULL, persona_name TEXT NOT NULL,
 start_interaction_id INTEGER NOT NULL, end_interaction_id INTEGER NOT NULL,
 message_count INTEGER NOT NULL, created_at TIMESTAMP NOT NULL)
+ INDEX idx_segment_scope ON (scope_type, scope_key, persona_name)
```

**Memory_Summaries** — extracted facts per segment, with summary-level embedding
```sql
(summary_id INTEGER PK AUTOINCREMENT, segment_id INTEGER NOT NULL FK,
 content TEXT NOT NULL, embedding BLOB, model_name TEXT, created_at TIMESTAMP NOT NULL)
+ INDEX idx_summary_segment ON (segment_id)
```

No ALTER TABLE migration needed — all `CREATE TABLE IF NOT EXISTS`.

### 1.2 New File: `src/embedding_service.py`

Provider-abstracted embedding service. Default provider: Gemini Embedding API (`text-embedding-004`, 768 dims, free tier 1500 RPM). Designed to be swappable to a local embedding server or other providers.

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

### 1.3 New MemoryManager Methods

```python
def store_message_embedding(self, interaction_id, embedding, model_name, created_at)
def get_unembedded_messages(self, persona_name, scope_type, scope_key, limit=500) -> List[Dict]
def store_segment(self, scope_type, scope_key, persona_name, start_id, end_id, message_count, created_at) -> int
def store_summary(self, segment_id, content, embedding, model_name, created_at) -> int
def get_summaries_for_scope(self, scope_type, scope_key, persona_name) -> List[Dict]
def get_active_memory_scopes(self, batch_limit=500) -> List[Tuple[str, str, str]]
```

`get_unembedded_messages` is `User_Interactions LEFT JOIN Message_Embeddings WHERE embedding IS NULL`, filtered by scope. `get_active_memory_scopes` returns distinct scope tuples that have unprocessed messages.

### 1.4 Dependencies

No new heavy dependencies. `google-generativeai` is already installed (used by TextEngine for Gemini LLM calls). `numpy` is likely already a transitive dependency; add it to requirements explicitly if not.

### 1.5 Tests

In `tests/database/test_memory_manager.py`:
- Schema creates all three new tables with correct columns
- store/retrieve round-trips for embeddings, segments, summaries
- `get_unembedded_messages` returns only unembedded
- Scope filtering works correctly
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
    # EmbeddingService with GeminiEmbeddingProvider (or configured provider)

    async def deploy(self):
        # Discover scopes with unprocessed messages
        # For each scope: embed -> segment -> summarize -> store

    def _segment_by_similarity(self, messages, embeddings) -> List[segments]:
        # Sliding window centroid: maintain running mean embedding
        # Cut when cosine_similarity(centroid, new_msg) < threshold
        # Enforce min_segment_size
        # Config: similarity_threshold (default 0.3), min_segment_size (default 3)

    async def _summarize_segment(self, segment) -> Optional[str]:
        # Build transcript from segment messages
        # Call memory_summarizer persona via text_engine
        # Returns extracted facts as bullet list
```

**Segmentation algorithm:**
1. Start with first message embedding as centroid
2. For each subsequent message, compute similarity to running centroid
3. If similarity < threshold AND current segment >= min_size: cut, start new centroid
4. Else: update centroid incrementally `centroid = centroid * ((n-1)/n) + vec/n`
5. Final remaining messages form last segment

### 2.2 System Persona

Add `memory_summarizer` to `config/system_personas.json`:
- Model: `gemma-3-27b-it` (local/free)
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
- Summarization calls LLM correctly, handles missing persona
- Deploy processes all scopes, respects shutdown
- Agent config injection

**Files:** `src/agents/memory_agent.py`, `config/system_personas.json`, `config/agents.json`, `src/main.py`, `tests/agents/test_memory_agent.py`

---

## Phase 3: Retrieval & Context Injection

### 3.1 New MemoryManager Method

```python
def retrieve_relevant_summaries(self, scope_type, scope_key, persona_name,
                                include_ambient=True) -> List[Dict]:
    # Returns all summaries for scope (persona + optionally ambient)
    # Dict keys: summary_id, content, embedding, segment_id, persona_name
```

### 3.2 New ChatSystem Method: `_retrieve_memory_block`

```python
async def _retrieve_memory_block(self, persona, user_identifier, channel,
                                 server_id, conversation_history) -> Optional[str]:
    # 1. Determine scope from persona's MemoryMode (same logic as _fetch_raw_history)
    # 2. Embed ALL messages in conversation_history as a batch (1 API call)
    # 3. Fetch all summaries for scope via retrieve_relevant_summaries
    # 4. For each summary: score = max(cosine_sim(summary_emb, msg_emb) for msg_emb in window)
    #    This ensures multi-topic conversations retrieve memories for ALL topics discussed,
    #    not just an averaged-out blend. Each message votes independently.
    # 5. Take top-K by max-similarity score, format as memory block
    # Returns None if: no embedding service, no summaries, feature disabled
```

ChatSystem gets a lazy `_embedding_service` property. Initialized with GeminiEmbeddingProvider by default. Returns None if embedding is not configured (graceful degradation — no memory retrieval, sliding window only).

**Why max-similarity over the full window:** A conversation may span multiple topics. Embedding the window into a single vector (mean/concat) dilutes each topic. Instead, embed each message individually in one batched API call, then score each candidate summary against every window message and keep the max. A "supply chain attack" summary scores high if any single message in the window touches cybersecurity — even if the rest is about lunch.

### 3.3 Inject in `_prepare_request`

After `_build_conversation_history()`, before appending user message:
```python
memory_block = await self._retrieve_memory_block(
    ctx.persona, ctx.user_identifier,
    ctx.channel, ctx.server_id, ctx.conversation_history
)
if memory_block:
    ctx.conversation_history.insert(0, {"role": "user", "content": memory_block})
```

Note: `_retrieve_memory_block` is async (embedding is an API call, not CPU-bound), so no `asyncio.to_thread` wrapper needed. The cosine similarity math is trivial and non-blocking.

### 3.4 Memory Block Format

```
<memory>
The following are relevant facts from previous conversations:

[conversation memory 1]
- Alice reported TeamPCP supply chain attack affecting pip packages
- Team decided to audit all third-party dependencies

[ambient memory 2]
- Security review found 3 unvetted transitive dependencies
</memory>
```

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
- Ambient inclusion/exclusion
- Scope isolation
- Format structure
- Empty history handling

In `tests/integration/test_memory_modes.py`:
- End-to-end: store messages -> embed -> segment -> summarize -> verify injection in _prepare_request

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

- **Gemini Embedding rate limits**: free tier is 1500 RPM — batch of 200 messages is 1 API call (batched natively), well within limits. Query-time embedding is 1 call per user message.
- **Gemma summarization rate limits** (30 RPM): 200 messages / 3 min_segment_size = ~67 segments max per cycle; may need to batch summarization calls or increase interval
- **Embedding API latency**: ~100-200ms per call at query time, acceptable on top of 1-5s LLM call
- **Backward compat**: all new schema is CREATE TABLE IF NOT EXISTS, all new config keys have defaults, MEMORY_RETRIEVAL_ENABLED defaults to false (opt-in)
- **Provider swappability**: EmbeddingProvider ABC allows swapping to local server (e.g., sentence-transformers behind Cloudflare tunnel) without changing any other code

---

## Verification

1. **Unit tests**: `pytest tests/database/test_memory_manager.py tests/test_embedding_service.py tests/agents/test_memory_agent.py tests/test_memory_retrieval.py -v`
2. **Integration**: `pytest tests/integration/test_memory_modes.py -v`
3. **Startup wiring**: `pytest tests/integration/test_startup_wiring.py -v` (verify MemoryAgent registration)
4. **Manual**: Start app with `auto_start: true` on memory agent, send messages in Discord, wait for batch cycle, verify summaries in DB (`sqlite3 user_memory.db "SELECT * FROM Memory_Summaries"`), then check `dump_context` shows memory block
5. **Lint/type**: `flake8 src/ && mypy src/ --config-file mypy.ini`
