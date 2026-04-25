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
) -> Tuple[Dict[str, Any], int]:
    """Translate DERPR User_Interactions rows into a kobold-lite savefile dict.

    Returns (savefile_dict, skipped_count). Skipped count covers system rows,
    empty-content rows, assistant rows whose only payload is a tool-call
    (tool_context present, content empty), and unrecognised roles.

    Tool-call / tool-result rendering is explicitly out of scope for Phase 2.1
    — see web_ui_roadmap backlog. Persona system prompt is pushed into
    kobold-lite's `instruct_sysprompt` setting by the UI, not into savefile
    `memory`, so the memory block stays free for future use.
    """
    actions: List[str] = []
    interaction_ids: List[int] = []
    skipped = 0

    for msg in raw_history:
        role = msg.get("author_role")
        content = (msg.get("content") or "").strip()

        if role == "system" or not content or role not in ("user", "assistant"):
            skipped += 1
            continue

        if role == "user":
            actions.append(f"{_INPUT_PLACEHOLDER}{content}{_OUTPUT_PLACEHOLDER}")
        else:  # assistant
            actions.append(content)
        
        interaction_ids.append(msg.get("interaction_id"))

    prompt = actions.pop(0) if actions else ""

    savefile: Dict[str, Any] = {
        "gamestarted": True,
        "prompt": prompt,
        "memory": "",
        "authorsnote": "",
        "anotetemplate": "",
        "actions": actions,
        "interaction_ids": interaction_ids,
        "actions_metadata": {},
        "worldinfo": [],
        "wifolders_d": {},
        "wifolders_l": [],
    }
    return savefile, skipped
