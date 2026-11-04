"""The codr ticket format (DP-488 / DP-489), shared by its writer and its reader.

stagr's `queue_codr_task` writes a ticket's TASK.md; the codr agent's
`load_tasks` reads it. Both take the format from here, so the two cannot drift:

    title: <one line>
    project: <[a-z0-9-]+>
    depends_on: CD-001, CD-004
    max_attempts: 3

    ## Spec
    ...
    ## Plan
    ...
    ## Packages
    ...

The header is `key: value` lines, ended by the first blank line; codr never
sees it. `depends_on` and `max_attempts` are optional. The body is the ticket
text codr reads.

Ticket IDs are `CD-NNN` (three digits or more), allocated by DERPR and never
reused: the next ID is one past the highest seen in the queue folder
(including dot-folders being staged and a high-water mark file) and in the
codr agent's attempt log. The attempt rows are the codr agent's
(`CODR_AGENT_NAME`, `ATTEMPT_ACTION`, trigger_context `attempt_context(id)`);
they are named here because `src.stagr` sits below `src.agents` and cannot
import them from there.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

TICKET_PREFIX = "CD"
TICKET_ID_RE = re.compile(rf"^{TICKET_PREFIX}-(\d{{3,}})$")
PROJECT_RE = re.compile(r"^[a-z0-9-]+$")
HEADER_KEYS = ("title", "project", "depends_on", "max_attempts")
SECTIONS = ("Spec", "Plan", "Packages")

CODR_AGENT_NAME = "codr"
ATTEMPT_ACTION = "task_attempt"


_HEADER_LINE = re.compile(r"([a-z][a-z_]*):[ \t]*(.*?)\s*")


def parse_header(text: str) -> Tuple[Dict[str, str], str]:
    """(header, ticket text). The header is the run of `key: value` lines at
    the top, ended by the first blank line or the first line that is not
    `key: value` (the pre-DP-488 form was one `max_attempts: N` line with the
    text right under it). No such first line = no header."""
    header: Dict[str, str] = {}
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        match = _HEADER_LINE.fullmatch(lines[i])
        if not match:
            break
        key, value = match.groups()
        if key in header:
            raise ValueError(f"`{key}:` appears twice in the header")
        header[key] = value
        i += 1
    if i < len(lines) and header and not lines[i].strip():
        i += 1
    return header, "\n".join(lines[i:])


def ticket_titles(tasks_dir: Path) -> Dict[str, str]:
    """{ticket id: title} for each queue folder whose TASK.md has a readable
    header with a title (the portal's channel labels, DP-484). Unlike the
    codr agent's `load_tickets`, nothing is validated: a label is not a start."""
    titles: Dict[str, str] = {}
    for task_md in sorted(tasks_dir.glob("*/TASK.md")) if tasks_dir.is_dir() else []:
        try:
            header, _ = parse_header(task_md.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if header.get("title"):
            titles[task_md.parent.name] = header["title"]
    return titles


def attempt_context(task_id: str) -> str:
    """The trigger_context of a codr attempt row for `task_id`."""
    return f"task:{task_id}"


def ticket_id(number: int) -> str:
    return f"{TICKET_PREFIX}-{number:03d}"


def ticket_number(task_id: str) -> Optional[int]:
    """N of a `CD-N` ticket ID, or None for any other name."""
    match = TICKET_ID_RE.match(task_id)
    return int(match.group(1)) if match else None


def render_task_md(*, title: str, project: str, depends_on: Iterable[str], max_attempts: Optional[int],
                   spec: str, plan: str, packages: str) -> str:
    """A ticket's TASK.md: the header, one blank line, then the sections.
    Values are taken as already validated (one line each)."""
    header = [f"title: {title}", f"project: {project}"]
    deps = list(depends_on)
    if deps:
        header.append(f"depends_on: {', '.join(deps)}")
    if max_attempts is not None:
        header.append(f"max_attempts: {max_attempts}")
    body = "\n\n".join(f"## {name}\n\n{text.strip()}" for name, text in zip(SECTIONS, (spec, plan, packages)))
    return "\n".join(header) + "\n\n" + body + "\n"
