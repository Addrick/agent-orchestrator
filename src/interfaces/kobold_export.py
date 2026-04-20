# src/interfaces/kobold_export.py

"""DERPR message_history → kobold-lite savefile JSON.

Builds the v1 'oldui' savefile shape kobold-lite ingests via its public
load_file path. We wrap user turns with kobold's instruct placeholder tags
(`{{[INPUT]}}` / `{{[OUTPUT]}}`) so kobold's own template renderer expands
them at submit time — DERPR never picks the actual instruct tags.

See memory/project/decisions/2026-04-19-portal-phase2-approach.md.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Mirror kobold-lite's instructstartplaceholder / instructendplaceholder
# constants from portal.html. These are stable cross-version tokens; kobold
# substitutes them with the active instruct_starttag/endtag at render time.
_INPUT_PLACEHOLDER = "\n{{[INPUT]}}\n"
_OUTPUT_PLACEHOLDER = "\n{{[OUTPUT]}}\n"


def build_kobold_savefile(
    raw_history: List[Dict[str, Any]],
    system_prompt: str = "",
) -> Tuple[Dict[str, Any], int]:
    """Translate DERPR User_Interactions rows into a kobold-lite savefile dict.

    Returns (savefile_dict, skipped_count). Skipped count covers system rows,
    rows with empty content, and tool-context expansions intentionally
    dropped (kobold has no place to render tool messages).
    """
    actions: List[str] = []
    skipped = 0

    for msg in raw_history:
        role = msg.get("author_role")
        content = (msg.get("content") or "").strip()
        tool_context = msg.get("tool_context")

        # Tool calls live inside `tool_context` JSON on assistant rows.
        # We don't expand them — kobold has no representation for tool turns.
        if tool_context:
            skipped += 1

        if role == "system":
            skipped += 1
            continue
        if not content:
            skipped += 1
            continue

        if role == "user":
            actions.append(f"{_INPUT_PLACEHOLDER}{content}{_OUTPUT_PLACEHOLDER}")
        elif role == "assistant":
            actions.append(content)
        else:
            skipped += 1

    prompt = actions.pop(0) if actions else ""

    savefile: Dict[str, Any] = {
        "gamestarted": True,
        "prompt": prompt,
        "memory": system_prompt or "",
        "authorsnote": "",
        "anotetemplate": "",
        "actions": actions,
        "actions_metadata": {},
        "worldinfo": [],
        "wifolders_d": {},
        "wifolders_l": [],
    }
    return savefile, skipped
