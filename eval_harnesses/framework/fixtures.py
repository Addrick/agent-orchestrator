from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .scenarios import Scenario, SeededInteraction
from .variants import MemoryVariant, PromptVariant


@dataclass
class FixtureBundle:
    """Everything one cell-run needs. Returned by build_fixture()."""
    chat_system: Any
    memory_manager: Any
    db_path: str
    mock_llm: Optional["MockLLM"]  # None for live runs
    cleanup: callable


class MockLLM:
    """Scripted-LLM stand-in.

    Each scenario/variant cell may register a sequence of canned responses.
    The runner installs this in place of the real engine for the duration of
    the cell. Records every prompt sent for grader inspection.

    Wiring is suite-specific: pass this into TextEngine via patch in the
    runner. Today this is a placeholder; concrete suites override
    `_install` to bind it to their generation path.
    """

    def __init__(self, scripted_turns: Optional[List[Dict[str, Any]]] = None):
        # Each turn: {"text": str, "tool_calls": [{"name": str, "args": {...}}, ...]}
        self.scripted = list(scripted_turns or [])
        self.calls: List[Dict[str, Any]] = []  # what the system asked us
        self._cursor = 0

    def next_turn(self, prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append({"prompt": prompt_payload})
        if self._cursor >= len(self.scripted):
            return {"text": "", "tool_calls": []}
        turn = self.scripted[self._cursor]
        self._cursor += 1
        return turn

    def reset(self) -> None:
        self.calls.clear()
        self._cursor = 0


def _seed_memory(memory_manager: Any, seeds: List[SeededInteraction]) -> None:
    """Inject seed interactions/summaries into the DB.

    For determinism, prefer SeededInteraction.pre_summary so the summarizer
    pipeline is bypassed. Raw seeds are stored as User_Interactions only;
    summarization is the real pipeline's job and is not invoked here.
    """
    if not seeds:
        return
    # TODO: implement actual seed insertion against memory_manager API.
    # Sketch:
    #   for s in seeds:
    #       mm.store_user_interaction(...)  # exact API TBD per suite
    #       if s.pre_summary:
    #           mm.store_summary(segment_id=..., content=s.pre_summary, ...)
    pass


@contextmanager
def build_fixture(
    scenario: Scenario,
    memory_variant: MemoryVariant,
    prompt_variant: PromptVariant,
    *,
    live: bool = False,
    mock_llm: Optional[MockLLM] = None,
) -> Generator[FixtureBundle, None, None]:
    """Spin up an isolated ChatSystem for one cell-run.

    Always uses a temp DB. When live=False, installs MockLLM. When live=True,
    real LLM + real Hindsight (if configured). Caller is responsible for
    teardown via the context manager.
    """
    # Lazy imports — keep framework import-light so CLI can run without
    # heavy deps when only listing scenarios/variants.
    from src.memory.memory_manager import MemoryManager
    from src.engine import TextEngine
    from src.bootstrap import create_chat_system
    from src.utils.save_utils import load_personas_from_file

    # Real-DB mode: variant points at an existing user DB. Don't reschema,
    # don't unlink on teardown — treat as read-only.
    # Frozen-slice mode: scenario.meta["slice_sql"] points at a .sql dump.
    # Materialize (cached by checksum) and treat the result like real-DB mode
    # so the test data is reproducible without committing a binary DB.
    using_real_db = bool(getattr(memory_variant, "db_path", None))
    slice_sql = scenario.meta.get("slice_sql") if not using_real_db else None
    if using_real_db:
        db_path = memory_variant.db_path
        mm = MemoryManager(db_path=db_path)
    elif slice_sql:
        from eval_harnesses.suites.memory_recall.load_slice import materialize_slice
        slice_path = Path(slice_sql)
        cache_dir = Path(scenario.meta.get("slice_cache_dir", ".eval_cache/slices"))
        db_path = str(materialize_slice(slice_path, cache_dir=cache_dir))
        mm = MemoryManager(db_path=db_path)
        using_real_db = True  # don't unlink the cached slice DB
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name
        mm = MemoryManager(db_path=db_path)
        mm.create_schema()

    # Memory-variant toggles. Hindsight-off swaps in the local sqlite backend.
    if not memory_variant.hindsight:
        mm.backend = mm._action_log  # local sqlite only

    _seed_memory(mm, scenario.seed_memory)

    # Persona load + prompt-variant overrides.
    default_file = os.path.join("config", "default_personas.json")
    personas = load_personas_from_file(file_path_override=default_file)
    target_persona = prompt_variant.persona_name or scenario.persona_name
    if target_persona in personas and prompt_variant.persona_prompt:
        personas[target_persona].prompt = prompt_variant.persona_prompt

    text_engine = TextEngine()
    chat_system = create_chat_system(
        memory_manager=mm, text_engine=text_engine, user_personas=personas,
    )

    # TODO: install MockLLM hook into text_engine when live=False.
    # Concrete wire-in is engine-specific; leaving as a clear stub so each
    # suite can pick its patch point.
    active_mock = None if live else (mock_llm or MockLLM())

    def _cleanup() -> None:
        try:
            mm.close()
        finally:
            if not using_real_db:
                try:
                    os.unlink(db_path)
                except OSError:
                    pass

    bundle = FixtureBundle(
        chat_system=chat_system,
        memory_manager=mm,
        db_path=db_path,
        mock_llm=active_mock,
        cleanup=_cleanup,
    )
    try:
        yield bundle
    finally:
        _cleanup()
