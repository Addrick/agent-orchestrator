# src/memory/lite_save.py
"""KoboldCpp Lite session save → conversation transcript (DP-252).

Lite's save JSON stores the session as ONE text stream, ``prompt +
"".join(actions)``: ``actions`` are generation chunks, not turns (a user turn
can be its own action, and a single ``<think>`` block can span two). So the
stream is joined first, then split on the instruct turn delimiters.

Only the conversation survives. ``savedsettings`` (which can hold API keys),
memory, author's note, world info and alternate branches are never read into
the output. Reasoning is always *captured* into ``reasoning_content`` — the
renderer drops it — so a DB import can still round-trip it.

Lite's optional injected timestamps (``get_current_timestamp()`` in
klite.embd: ``toLocaleTimeString`` with a numeric date and 2-digit time) are
browser-locale dependent and zoneless. Only the en-US shape
``[9/18/2026, 01:44 AM]`` is recognised; it is rewritten to
``[2026-09-18 01:44]`` so the generic content-date regex (which deliberately
rejects bare M/D/YYYY as locale-ambiguous) anchors on it. Anything else is left
verbatim.

Pure: no I/O, no engine imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from typing import Any, Dict, List, Optional

# klite.embd instruct*placeholder constants — used unless raw_instruct_tags.
_PLACEHOLDERS = {
    "user": ("{{[INPUT]}}", "{{[INPUT_END]}}"),
    "assistant": ("{{[OUTPUT]}}", "{{[OUTPUT_END]}}"),
    "system": ("{{[SYSTEM]}}", "{{[SYSTEM_END]}}"),
}
_SETTING_KEYS = {
    "user": ("instruct_starttag", "instruct_starttag_end"),
    "assistant": ("instruct_endtag", "instruct_endtag_end"),
    "system": ("instruct_systag", "instruct_systag_end"),
}
_INSTRUCT_OPMODE = "4"

# Newer Chromium separates the time and AM/PM with U+202F, older with a space.
_LITE_STAMP_RE = re.compile(
    r"\[(\d{1,2})/(\d{1,2})/(\d{4}),\s(\d{1,2}):(\d{2})[   ]?([AaPp][Mm])\]"
)
# Lite's export filename: <title>_M_D_YYYY__h_mm_ss_AM.json
_FILENAME_TIME_RE = re.compile(
    r"_(\d{1,2})_(\d{1,2})_(\d{4})__(\d{1,2})_(\d{2})_(\d{2})_([AaPp][Mm])\.json$"
)


class NotALiteSave(ValueError):
    """The input is not a Lite save this parser supports; message says why."""


@dataclass
class LiteMessage:
    role: str  # "user" | "assistant" | "system"
    content: str
    reasoning_content: Optional[str] = None


@dataclass
class LiteTranscript:
    messages: List[LiteMessage] = field(default_factory=list)
    stamps: List[datetime] = field(default_factory=list)  # tz-aware, in order

    @property
    def reasoning_stripped(self) -> bool:
        return any(m.reasoning_content for m in self.messages)

    def render(self) -> str:
        """User/assistant turns as a plain transcript. System turns are setup,
        not conversation, and are left out (like Lite's memory field)."""
        blocks = []
        for m in self.messages:
            if m.role == "system":
                continue
            speaker = "User" if m.role == "user" else "Assistant"
            blocks.append(f"{speaker}: {m.content}")
        return "\n\n".join(blocks)

    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.role != "system")


def looks_like_lite_save(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("prompt"), str)
        and isinstance(data.get("actions"), list)
        and isinstance(data.get("savedsettings"), dict)
    )


def parse_lite_save(data: Any, *, tz: tzinfo) -> LiteTranscript:
    """Parse a Lite save dict. ``tz`` interprets the zoneless timestamps.

    Raises ``NotALiteSave`` for non-saves, non-instruct saves, and saves with
    no conversation turns.
    """
    if not looks_like_lite_save(data):
        raise NotALiteSave("not a KoboldCpp Lite save")
    settings: Dict[str, Any] = data["savedsettings"]
    opmode = settings.get("opmode")
    if opmode is not None and str(opmode) != _INSTRUCT_OPMODE:
        raise NotALiteSave("only instruct-mode Lite saves are supported")

    stream = data["prompt"] + "".join(a for a in data["actions"] if isinstance(a, str))
    think_open = settings.get("start_thinking_tag") or "<think>"
    think_close = settings.get("stop_thinking_tag") or "</think>"

    transcript = LiteTranscript()
    for role, raw in _split_turns(stream, _delimiters(settings)):
        body, reasoning = _strip_reasoning(raw, think_open, think_close)
        body, stamps = _normalize_stamps(body.strip(), tz)
        transcript.stamps.extend(stamps)
        if body:
            transcript.messages.append(LiteMessage(role, body, reasoning or None))
    if transcript.turn_count() == 0:
        raise NotALiteSave("Lite save has no conversation turns")
    return transcript


def export_time_from_filename(name: str, *, tz: tzinfo) -> Optional[datetime]:
    """Lite's export time from its filename, tz-aware; None if absent."""
    m = _FILENAME_TIME_RE.search(name)
    if not m:
        return None
    mo, d, y, h, mi, s, ampm = m.groups()
    return _local(int(y), int(mo), int(d), int(h), int(mi), int(s), ampm, tz)


def _delimiters(settings: Dict[str, Any]) -> Dict[str, str]:
    """Delimiter text → role. Placeholders by default; the literal instruct
    tags when the save used raw_instruct_tags. End tags are delimiters too
    (separate_end_tags) and map to None → they close a turn, open none."""
    out: Dict[str, str] = {}
    raw = bool(settings.get("raw_instruct_tags"))
    for role in ("user", "assistant", "system"):
        if raw:
            start_key, end_key = _SETTING_KEYS[role]
            start, end = settings.get(start_key) or "", settings.get(end_key) or ""
        else:
            start, end = _PLACEHOLDERS[role]
        if start:
            out[start] = role
        if end:
            out[end] = ""
    return out


def _split_turns(stream: str, delimiters: Dict[str, str]) -> List[tuple[str, str]]:
    """Split on the delimiters, longest first so a tag that prefixes another
    cannot shadow it. Text before the first delimiter is treated as system
    setup, and text after an end tag (outside any turn) is dropped."""
    if not delimiters:
        return []
    pattern = re.compile("|".join(re.escape(t) for t in sorted(delimiters, key=len, reverse=True)))
    turns: List[tuple[str, str]] = []
    role: Optional[str] = "system"
    pos = 0
    for m in pattern.finditer(stream):
        if role:
            turns.append((role, stream[pos:m.start()]))
        role = delimiters[m.group(0)] or None
        pos = m.end()
    if role:
        turns.append((role, stream[pos:]))
    return turns


def _strip_reasoning(text: str, open_tag: str, close_tag: str) -> tuple[str, str]:
    """Remove every reasoning span; return (body, reasoning).

    Mirrors Lite's handle_mismatched_think: a close tag with no opener (the
    opener was injected into the prompt) means the turn *starts* in reasoning;
    an opener with no close means it runs to the end of the turn.
    """
    traces: List[str] = []
    first_close, first_open = text.find(close_tag), text.find(open_tag)
    if first_close != -1 and (first_open == -1 or first_close < first_open):
        traces.append(text[:first_close])
        text = text[first_close + len(close_tag):]
    kept: List[str] = []
    while True:
        start = text.find(open_tag)
        if start == -1:
            kept.append(text)
            break
        kept.append(text[:start])
        end = text.find(close_tag, start + len(open_tag))
        if end == -1:
            traces.append(text[start + len(open_tag):])
            break
        traces.append(text[start + len(open_tag):end])
        text = text[end + len(close_tag):]
    reasoning = "\n\n".join(t.strip() for t in traces if t.strip())
    return "".join(kept), reasoning


def _normalize_stamps(text: str, tz: tzinfo) -> tuple[str, List[datetime]]:
    found: List[datetime] = []

    def repl(m: re.Match[str]) -> str:
        mo, d, y, h, mi, ampm = m.groups()
        dt = _local(int(y), int(mo), int(d), int(h), int(mi), 0, ampm, tz)
        if dt is None:
            return m.group(0)
        found.append(dt)
        return f"[{dt:%Y-%m-%d %H:%M}]"

    return _LITE_STAMP_RE.sub(repl, text), found


def _local(y: int, mo: int, d: int, h: int, mi: int, s: int,
           ampm: str, tz: tzinfo) -> Optional[datetime]:
    if not 1 <= h <= 12:
        return None
    h24 = h % 12 + (12 if ampm.lower() == "pm" else 0)
    try:
        return datetime(y, mo, d, h24, mi, s, tzinfo=tz)
    except ValueError:
        return None  # month 13, day 32, …
