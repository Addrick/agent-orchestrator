# DP-118 — `ingest_path` tool design

**Date:** 2026-05-18
**Status:** Approved (brainstorm); implementation in progress.

## Goal

Give personas an agent-callable tool that ingests a file or directory of markdown
notes into the persona's Hindsight bank. Targets the "monitor specific
directories" need without a background watcher: per-call, ad-hoc, idempotent via
a local hash cache.

## Non-goals

- No filesystem watcher or background scan loop.
- No ingestion into the SQLite turn-history backend. user_db is the wrong fit for
  long-form docs; if SQLite backend is active, the tool returns a warning.
- No chunking of large files (one file = one Hindsight retain item).
- No config-listed monitored dirs (rejected during brainstorming).

## Architecture

```
persona → LLM calls "ingest_path" tool
            ↓
   IngestPathHandler  (src/tools/ingest_path.py)
     - resolves bank: arg → persona.ingest_bank → persona.name
     - global gate: global_config.INGEST_PATH_ENABLED
     - walks path, applies glob, reads files
     - per-file sha256 → skip if cache hit (unless force=True)
     - dispatches: MemoryBackend.retain_document(...)
            ↓
   HindsightBackend.retain_document
     - one aretain item, document_id = relative path
     - update_mode replace (idempotent)
   SqliteSemanticBackend.retain_document
     - logs warning, returns None
```

## Components

### 1. `MemoryBackend.retain_document` (new ABC method)

```python
# src/memory/backend/base.py
async def retain_document(
    self,
    bank_id: str,
    document_id: str,
    content: str,
    *,
    tags: List[str],
    metadata: Dict[str, str],
    timestamp: datetime,
) -> None:
    raise NotImplementedError("retain_document not implemented on this backend")
```

- `document_id` is the stable key (relative path). Re-call with same id =
  `update_mode=replace` server-side.
- `metadata` values MUST be strings (per `hindsight_metadata_strings_only.md`).
- `timestamp` = file mtime, ISO-encoded by Hindsight impl.

### 2. `HindsightBackend.retain_document`

```python
# src/memory/backend/hindsight.py
async def retain_document(self, bank_id, document_id, content, *, tags, metadata, timestamp):
    await self._client.aretain(
        bank_id,
        [{
            "content": content,
            "tags": tags,
            "metadata": metadata,
            "timestamp": timestamp.isoformat(),
            "document_id": document_id,
        }],
        async_=True,
    )
```

Bank is assumed to exist (created lazily by existing turn-retain path). If it
doesn't, Hindsight returns 404; tool surfaces the error.

### 3. `SqliteSemanticBackend.retain_document`

```python
async def retain_document(self, bank_id, document_id, content, *, tags, metadata, timestamp):
    logger.warning(
        "retain_document: sqlite backend active; ingest is noop. "
        "Switch MEMORY_BACKEND to hindsight to use ingest_path."
    )
    return None
```

### 4. `Persona.ingest_bank` (new optional field)

- New `__init__` kwarg: `ingest_bank: Optional[str] = None`.
- Getter: `get_ingest_bank() -> Optional[str]`.
- Resolution order in tool handler: `tool_arg.bank` → `persona.ingest_bank` →
  `persona.name`.
- Passed through in `save_utils.py` persona save/load.

### 5. `global_config.INGEST_PATH_ENABLED`

```python
INGEST_PATH_ENABLED: bool = bool(int(os.environ.get("INGEST_PATH_ENABLED", "1")))
```

Tool handler short-circuits with `"ingest_path disabled globally"` when False.

### 6. `IngestPathHandler` (`src/tools/ingest_path.py`)

```python
class IngestPathHandler:
    def __init__(self, memory_backend, cache_dir: Path):
        self.memory_backend = memory_backend
        self.cache_dir = cache_dir

    def register(self, manager: ToolManager) -> None:
        manager.register("ingest_path", self._ingest_path)

    async def _ingest_path(
        self,
        path: str,
        glob: str = "**/*.md",
        bank: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]: ...
```

Flow:
1. Check `INGEST_PATH_ENABLED`. If False → return `{"status": "disabled"}`.
2. Resolve bank via turn_context (persona) + arg override.
3. Resolve path; abs. If file: candidates=[path]. If dir: rglob filter.
4. Load `cache_dir / f"{bank}.json"` (create empty if absent).
5. For each candidate:
   - Read bytes, sha256.
   - If not force and cache[relpath].sha256 == sha → skip.
   - Decode utf-8 errors='replace'.
   - Build metadata dict (all str): `source_path`, `sha256`, `file_mtime`.
   - Tags: `["ingest", "notes"]`.
   - `await backend.retain_document(...)`. Retry 3x exp backoff on Hindsight 5xx.
   - Update cache entry.
6. Persist cache.
7. Return `{"ingested": N, "skipped": M, "failed": K, "bank": bank}`.

Errors per-file are caught and counted; the call as a whole succeeds.

### 7. Tool definition

```python
# src/tools/definitions.py
{
    "type": "function",
    "is_write": True,
    "capabilities": {
        "produces_untrusted": True,
        "irreversible": False,
        "locality": "local",
        "sensitivity": "user",
    },
    "function": {
        "name": "ingest_path",
        "description": "Ingest a markdown file or directory of notes into the persona's "
                       "long-term memory bank. Idempotent: unchanged files are skipped via "
                       "a local hash cache.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory path."},
                "glob": {"type": "string", "description": "Glob filter (default '**/*.md').",
                         "default": "**/*.md"},
                "bank": {"type": "string",
                         "description": "Override target bank. Defaults to persona's ingest_bank "
                                        "or persona name."},
                "force": {"type": "boolean",
                          "description": "Bypass hash cache and re-ingest all matches.",
                          "default": False},
            },
            "required": ["path"],
        },
    },
}
```

`produces_untrusted=True` because file content originates outside the
conversation; the security framework should taint recall hits surfaced from
this bank.

### 8. ChatSystem wiring

```python
# src/chat_system.py — alongside MemoryRecallHandler
from src.tools.ingest_path import IngestPathHandler
IngestPathHandler(self.memory_backend, cache_dir=INGEST_CACHE_DIR).register(self.tool_manager)
```

`INGEST_CACHE_DIR` from global_config, default `~/.derpr/ingest_cache/`.

## Toggle layers (recap)

1. Per-persona: include/exclude `"ingest_path"` in `enabled_tools`.
2. Per-call: `force` (only behavior switch).
3. Global: `INGEST_PATH_ENABLED` env-driven kill switch.

## Cache file shape

`~/.derpr/ingest_cache/<bank>.json`:

```json
{
  "notes/foo.md": {
    "sha256": "abc...",
    "mtime": "2026-05-18T12:34:56+00:00",
    "ingested_at": "2026-05-18T13:00:00+00:00"
  }
}
```

## Testing

**Unit (`tests/tools/test_ingest_path.py`):**
- Bank resolution: arg > persona.ingest_bank > persona.name.
- Single file ingest: 1 retain call, correct document_id.
- Dir ingest: glob applies, recursive walk, N calls.
- Cache hit: unchanged sha → 0 retain calls, counted skipped.
- Cache miss: modified file → 1 retain call, cache updated.
- `force=True` bypasses cache.
- Unreadable file counted as failed; others proceed.
- Disabled flag → early return.
- Backend stub asserts metadata values all `str`.

**Backend (`tests/memory/test_*_backend.py`):**
- `SqliteSemanticBackend.retain_document` is noop, warns.
- `HindsightBackend.retain_document` posts one item w/ document_id + iso ts.

**Persona (`tests/test_persona.py`):**
- `ingest_bank` absent → getter returns None.
- `ingest_bank="my_bank"` → getter returns "my_bank".

**Integration:**
- `tests/integration/test_startup_wiring.py` auto-covers the registration.

**Live (`tests/live/test_ingest_path_live.py`, opt-in `llm_live`):**
- Ingest 2 small md files into ephemeral bank, poll until drained, assert
  fact_count > 0, delete bank.

## Files touched

- `src/memory/backend/base.py` — new ABC method.
- `src/memory/backend/hindsight.py` — impl.
- `src/memory/backend/sqlite.py` — noop+warn.
- `src/persona.py` — new field + getter.
- `src/utils/save_utils.py` — pass-through.
- `src/tools/ingest_path.py` — new file.
- `src/tools/definitions.py` — new tool entry.
- `src/chat_system.py` — handler registration.
- `config/global_config.py` — `INGEST_PATH_ENABLED`, `INGEST_CACHE_DIR`.
- Tests as listed above.

## Open items

None — design is final per brainstorming session 2026-05-18.
