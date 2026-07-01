"""Tool definitions split by service binding (DP-248).

Each module exports one list of tool-definition dicts. `definitions.py`
concatenates them (in this order) into `ALL_TOOL_DEFINITIONS` and owns all
capability/helper logic. Splitting is for navigability only — the assembled
list is byte-identical to the pre-split literal.
"""

from src.tools.tool_defs.search import SEARCH_TOOLS
from src.tools.tool_defs.zammad import ZAMMAD_TOOLS
from src.tools.tool_defs.agents import AGENT_TOOLS
from src.tools.tool_defs.memory import MEMORY_TOOLS
from src.tools.tool_defs.fixr import FIXR_TOOLS
from src.tools.tool_defs.voice import VOICE_TOOLS
from src.tools.tool_defs.proxmox import PROXMOX_TOOLS

__all__ = [
    "SEARCH_TOOLS",
    "ZAMMAD_TOOLS",
    "AGENT_TOOLS",
    "MEMORY_TOOLS",
    "FIXR_TOOLS",
    "VOICE_TOOLS",
    "PROXMOX_TOOLS",
]
