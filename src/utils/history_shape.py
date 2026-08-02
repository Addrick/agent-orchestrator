# src/utils/history_shape.py
"""Provider-agnostic reshaping of the legacy ``history_object`` (DP-317).

Lives in ``utils`` — the leaf layer — because both ``src.engine.providers``
and ``src.stream_engine`` need it, and `stream_engine` sits *below* `engine`
in the layer order (setup.cfg). Putting it in `engine.providers._shared`, where
it started, meant `stream_engine` could not reach it without an upward import:
a layer violation, and a genuine import cycle, since `src.engine`'s package
``__init__`` imports `driver`, which imports `src.stream_engine`.

Pure dict manipulation, no imports beyond typing — keeps `utils` a
dependency-free leaf.
"""

from typing import Any, Dict, List, Tuple


def extract_system_prompt(history_object: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Returns (merged_system_prompt, remaining_history). A leading system turn
    in the history is folded into the persona prompt.

    **Merged, never substituted.** The persona prompt is the persona's standing
    instructions; a system turn in the history is an additional injection (e.g.
    an agent's action-history block from ``agents/base._build_history_object``),
    not a replacement for it. Two call sites used to inline their own split that
    dropped ``persona_prompt`` whenever the history opened with a system turn —
    that divergence traces to a single 2025-10-06 commit (``3921318``,
    "reimplement history limit") which added the leading-system-turn branch to
    three providers at once and transcribed it two different ways. Before that
    commit no provider had the branch at all. It was a slip, not a design
    choice; see DP-317.
    """
    system_prompt = history_object["persona_prompt"]
    history = history_object.get("message_history", history_object.get("history", []))
    if history and history[0]["role"] == "system":
        system_prompt = f"{system_prompt}\n\n{history[0]['content']}"
        history = history[1:]
    return system_prompt, history
