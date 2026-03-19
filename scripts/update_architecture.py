#!/usr/bin/env python3
"""
Regenerates docs/architecture.svg from source imports and updates the Mermaid
diagram in readme.md to reflect the current module dependency graph.

Requirements: pydeps, graphviz (system package)
Usage: python scripts/update_architecture.py
"""

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
README_PATH = PROJECT_ROOT / "readme.md"

# Modules to include in the Mermaid diagram, mapped to display labels
MODULE_LABELS: Dict[str, str] = {
    "src.main": "main",
    "src.chat_system": "ChatSystem",
    "src.message_handler": "BotLogic",
    "src.engine": "TextEngine",
    "src.persona": "Persona",
    "src.database.memory_manager": "MemoryManager",
    "src.clients.zammad_client": "ZammadClient",
    "src.tools.tool_manager": "ToolManager",
    "src.tools.definitions": "Tool Definitions",
    "src.interfaces.discord_bot": "Discord Bot",
    "src.interfaces.gmail_bot": "Gmail Bot",
    "src.interfaces.zammad_bot": "Zammad Bot",
    "src.utils.google_utils": "google_utils",
    "src.utils.model_utils": "model_utils",
    "src.utils.save_utils": "save_utils",
    "src.utils.message_utils": "message_utils",
}

# Subgraph groupings for Mermaid layout
SUBGRAPHS: Dict[str, list] = {
    "Interfaces": [
        "src.interfaces.discord_bot",
        "src.interfaces.gmail_bot",
        "src.interfaces.zammad_bot",
    ],
    "Core": [
        "src.chat_system",
        "src.message_handler",
        "src.engine",
        "src.persona",
    ],
    "Data & Tools": [
        "src.database.memory_manager",
        "src.tools.tool_manager",
        "src.tools.definitions",
        "src.clients.zammad_client",
    ],
    "Utils": [
        "src.utils.google_utils",
        "src.utils.model_utils",
        "src.utils.save_utils",
        "src.utils.message_utils",
    ],
}

# Mermaid node IDs (short, no dots)
NODE_IDS: Dict[str, str] = {
    "src.main": "Main",
    "src.chat_system": "CS",
    "src.message_handler": "BL",
    "src.engine": "Engine",
    "src.persona": "Persona",
    "src.database.memory_manager": "MM",
    "src.clients.zammad_client": "ZC",
    "src.tools.tool_manager": "TM",
    "src.tools.definitions": "TDef",
    "src.interfaces.discord_bot": "Discord",
    "src.interfaces.gmail_bot": "Gmail",
    "src.interfaces.zammad_bot": "ZBot",
    "src.utils.google_utils": "GU",
    "src.utils.model_utils": "MU",
    "src.utils.save_utils": "SU",
    "src.utils.message_utils": "MSU",
}


def module_path_to_dotted(file_path: Path) -> str:
    """Convert a file path to a dotted module name relative to project root."""
    rel = file_path.relative_to(PROJECT_ROOT)
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts)


def extract_imports(file_path: Path) -> Set[str]:
    """Extract first-party imports from a Python file using AST parsing."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return set()

    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("src.", "config.")):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(("src.", "config.")):
                imports.add(node.module)
    return imports


def resolve_import_to_module(imp: str) -> str:
    """Resolve an import string to the nearest tracked module."""
    # Direct match
    if imp in MODULE_LABELS:
        return imp
    # Walk up the module path to find a match
    parts = imp.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in MODULE_LABELS:
            return candidate
        parts.pop()
    return ""


def build_dependency_graph() -> Dict[str, Set[str]]:
    """Walk all tracked modules and build edges."""
    graph: Dict[str, Set[str]] = {mod: set() for mod in MODULE_LABELS}

    for module_dotted in MODULE_LABELS:
        parts = module_dotted.split(".")
        file_path = PROJECT_ROOT / Path(*parts).with_suffix(".py")
        if not file_path.exists():
            # Try __init__.py
            file_path = PROJECT_ROOT / Path(*parts) / "__init__.py"
            if not file_path.exists():
                continue

        raw_imports = extract_imports(file_path)
        for imp in raw_imports:
            target = resolve_import_to_module(imp)
            if target and target != module_dotted:
                graph[module_dotted].add(target)

    return graph


def generate_mermaid(graph: Dict[str, Set[str]]) -> str:
    """Generate a Mermaid graph TD block from the dependency graph."""
    lines = ["graph TD"]

    # Subgraphs
    for group_name, members in SUBGRAPHS.items():
        lines.append(f"    subgraph {group_name}")
        for mod in members:
            nid = NODE_IDS[mod]
            label = MODULE_LABELS[mod]
            if " " in label or "(" in label:
                lines.append(f'        {nid}["{label}"]')
            else:
                lines.append(f"        {nid}[{label}]")
        lines.append("    end")
        lines.append("")

    # Edges (skip src.main to avoid the fan-out clutter)
    for source, targets in sorted(graph.items()):
        if source == "src.main":
            continue
        for target in sorted(targets):
            if target == "src.main":
                continue
            src_id = NODE_IDS.get(source)
            tgt_id = NODE_IDS.get(target)
            if src_id and tgt_id:
                lines.append(f"    {src_id} --> {tgt_id}")

    return "\n".join(lines)


def update_readme(mermaid_block: str) -> bool:
    """Replace the Mermaid block in readme.md. Returns True if changed."""
    content = README_PATH.read_text(encoding="utf-8")

    start_marker = "```mermaid\n"
    end_marker = "\n```"

    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("ERROR: No mermaid block found in readme.md")
        return False

    # Find the closing ``` after the mermaid start
    search_from = start_idx + len(start_marker)
    end_idx = content.find(end_marker, search_from)
    if end_idx == -1:
        print("ERROR: No closing ``` found for mermaid block")
        return False

    old_block = content[start_idx + len(start_marker):end_idx]
    if old_block.strip() == mermaid_block.strip():
        print("Mermaid diagram is already up to date.")
        return False

    new_content = content[:start_idx + len(start_marker)] + mermaid_block + content[end_idx:]
    README_PATH.write_text(new_content, encoding="utf-8")
    print("Updated Mermaid diagram in readme.md")
    return True


def update_svg() -> bool:
    """Regenerate docs/architecture.svg via pydeps. Returns True if changed."""
    svg_path = PROJECT_ROOT / "docs" / "architecture.svg"
    old_content = svg_path.read_text(encoding="utf-8") if svg_path.exists() else ""

    try:
        subprocess.run(
            ["pydeps", "src", "--no-show"],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("WARNING: pydeps not installed, skipping SVG generation.")
        print("  Install with: pip install pydeps  (also requires graphviz)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"WARNING: pydeps failed: {e.stderr}")
        return False

    new_content = svg_path.read_text(encoding="utf-8") if svg_path.exists() else ""
    if old_content == new_content:
        print("SVG diagram is already up to date.")
        return False

    print("Updated docs/architecture.svg")
    return True


def main() -> None:
    os.chdir(PROJECT_ROOT)

    print("Scanning module imports...")
    graph = build_dependency_graph()

    edge_count = sum(len(targets) for targets in graph.values())
    print(f"Found {len(graph)} modules, {edge_count} dependency edges")

    mermaid = generate_mermaid(graph)
    readme_changed = update_readme(mermaid)
    svg_changed = update_svg()

    if readme_changed or svg_changed:
        print("\nFiles updated. Review and commit when ready.")
    else:
        print("\nNo changes needed.")


if __name__ == "__main__":
    main()
