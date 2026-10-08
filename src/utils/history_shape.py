# src/utils/history_shape.py
"""Provider-agnostic reshaping of the legacy ``history_object`` (DP-317).

Lives in ``utils`` — the leaf layer — because it was shared with
``src.stream_engine``, which sat *below* `engine` in the layer order and so
could not import `engine.providers._shared` (retired in DP-417; the providers
re-export it from `_shared`).

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
