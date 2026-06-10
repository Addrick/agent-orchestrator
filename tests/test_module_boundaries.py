# tests/test_module_boundaries.py
"""Mechanical enforcement of the DP-200 module dependency rules.

The target-architecture doc (memory repo: plans/DP-200-target-architecture.md)
declares per-layer import rules with "violation = bug". This test is the
enforcement: it AST-walks every module under src/ and asserts the rules,
so a new cross-layer import fails CI instead of silently re-tangling the
layering the refactor is paying to untangle.

Two deliberate softenings:

- Imports inside ``if TYPE_CHECKING:`` blocks are ignored — they are
  annotations, not runtime coupling.
- Current, known violations are grandfathered in KNOWN_DEBT below. Each entry
  is one edge that DP-200's remaining slices are expected to remove. The list
  is *checked for staleness*: if a grandfathered edge disappears, the test
  fails until the entry is deleted, so the list can only shrink.
"""

import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

SRC_ROOT = Path(__file__).parent.parent / "src"


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _runtime_imports(tree: ast.Module, module: str) -> Set[str]:
    """First-party (src.*) modules imported at runtime by `module`.

    Skips anything nested under an `if TYPE_CHECKING:` block. Relative
    imports are resolved against the importing module's package.
    """
    found: Set[str] = set()

    def walk(nodes: List[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.If) and _is_type_checking_guard(node):
                walk(node.orelse)
                continue
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names if a.name.startswith("src."))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = module.split(".")[: -node.level]
                    base = ".".join(parts + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                if base.startswith("src."):
                    found.add(base)
            # Recurse into any compound statement body (function defs catch
            # lazy imports, which are still runtime coupling).
            for field in ("body", "orelse", "finalbody", "handlers"):
                children = getattr(node, field, None)
                if children:
                    walk([c for c in children if isinstance(c, ast.stmt)])
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    walk(handler.body)

    walk(tree.body)
    return found


def _import_graph() -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    for py in sorted(SRC_ROOT.rglob("*.py")):
        rel = py.relative_to(SRC_ROOT.parent)
        module = ".".join(rel.with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        tree = ast.parse(py.read_text(encoding="utf-8"))
        graph[module] = _runtime_imports(tree, module)
    return graph


# (source prefix, forbidden target prefixes). A module matching the source
# prefix may not import any module matching a forbidden prefix.
CONTRACTS: List[Tuple[str, Tuple[str, ...]]] = [
    # memory/* is a lower layer: it must never reach up into orchestration,
    # commands, tools, transports, agents, or the engines.
    ("src.memory", (
        "src.chat_system", "src.message_handler", "src.tools",
        "src.interfaces", "src.agents", "src.stream_engine", "src.engine",
    )),
    # tools/* may see persona + the backend ABC (the Protocol boundary),
    # never transports/orchestrator/concrete storage. The ABC allowance is
    # expressed by forbidding the concrete modules rather than src.memory.
    ("src.tools", (
        "src.interfaces", "src.chat_system", "src.message_handler",
        "src.memory.memory_manager", "src.memory.memory_consolidation",
        "src.memory.backend.sqlite", "src.memory.backend.hindsight",
        "src.memory.router", "src.agents",
    )),
    # Transport interfaces must not import each other.
    ("src.interfaces.discord_bot", ("src.interfaces.gmail_bot", "src.interfaces.kobold_engine_adapter")),
    ("src.interfaces.gmail_bot", ("src.interfaces.discord_bot", "src.interfaces.kobold_engine_adapter")),
    ("src.interfaces.kobold_engine_adapter", ("src.interfaces.discord_bot", "src.interfaces.gmail_bot")),
    # Interfaces must not reach into agents.
    ("src.interfaces", ("src.agents",)),
    # Agents must not reach into transports or the command layer.
    ("src.agents", ("src.interfaces", "src.message_handler")),
    # utils/* is stdlib+config only.
    ("src.utils", (
        "src.chat_system", "src.message_handler", "src.engine",
        "src.stream_engine", "src.memory", "src.interfaces", "src.agents",
        "src.tools", "src.persona", "src.clients",
    )),
    # Persona persistence (DP-203: ex-utils/save_utils). May see the persona
    # domain object and tools/ — per the DP-204 inversion the LOADER (not
    # persona) runs DP-128 composition validation via tools.composition.
    # Never orchestration, engines, transports, agents, or storage.
    ("src.personas", (
        "src.chat_system", "src.message_handler", "src.engine",
        "src.stream_engine", "src.memory", "src.interfaces", "src.agents",
        "src.clients",
    )),
    # persona is a domain leaf.
    ("src.persona", (
        "src.chat_system", "src.message_handler", "src.engine",
        "src.stream_engine", "src.memory", "src.interfaces", "src.agents",
        "src.clients", "src.tools",
    )),
    # clients/* are cross-cutting leaves.
    ("src.clients", (
        "src.chat_system", "src.message_handler", "src.memory",
        "src.interfaces", "src.agents",
    )),
    # Command layer (BotLogic) takes explicit deps (DP-202): it must never
    # import the orchestrator, engines, transports, or agents at runtime —
    # collaborators are injected by the composition site (ChatSystem.__init__).
    ("src.message_handler", (
        "src.chat_system", "src.engine", "src.stream_engine",
        "src.interfaces", "src.agents", "src.turn_persistence",
        "src.memory",
    )),
    # Request assembly sits below the orchestrator: personas, tools, storage.
    ("src.request_builder", (
        "src.chat_system", "src.message_handler", "src.interfaces",
        "src.agents", "src.engine", "src.stream_engine",
    )),
    # Turn persistence sits below the orchestrator: storage + shared leaves.
    ("src.turn_persistence", (
        "src.chat_system", "src.message_handler", "src.interfaces",
        "src.agents", "src.engine", "src.stream_engine", "src.tools",
    )),
    # Confirmation parking sits below the orchestrator: tools + storage only.
    ("src.confirmations", (
        "src.chat_system", "src.message_handler", "src.interfaces",
        "src.agents", "src.engine", "src.stream_engine",
    )),
    # Engine layer never reaches up into orchestration/storage/transports.
    ("src.engine", ("src.chat_system", "src.message_handler", "src.memory", "src.interfaces", "src.agents")),
    ("src.stream_engine", ("src.chat_system", "src.message_handler", "src.memory", "src.interfaces", "src.agents")),
    ("src.text_tool_protocol", ("src.",)),
    ("src.generation_params", ("src.",)),
    ("src.generation_events", ("src.",)),
    ("src.embedding_service", ("src.",)),
    # Nothing imports the entrypoint, and only the entrypoint may use the
    # composition root (src.bootstrap) — modules must receive their deps,
    # not assemble them.
    ("src.", ("src.main", "src.bootstrap")),
]

# Grandfathered edges: real, current violations of the target rules that the
# remaining DP-200 slices are expected to remove. Removing the code edge
# without deleting its entry here fails the staleness check below.
KNOWN_DEBT: Set[Tuple[str, str]] = {
    # persona holds a ToolPolicy instance; DP-204 commit 2 moves ToolPolicy
    # to a leaf module so this last edge can be deleted.
    ("src.persona", "src.tools.policy"),
}


def _violations() -> Tuple[List[str], Set[Tuple[str, str]]]:
    graph = _import_graph()
    bad: List[str] = []
    seen_debt: Set[Tuple[str, str]] = set()
    for module, imports in graph.items():
        for src_prefix, forbidden in CONTRACTS:
            if not (module == src_prefix or module.startswith(src_prefix + ".")
                    or (src_prefix == "src." and module.startswith("src."))):
                continue
            if module == "src.main":
                continue  # the entrypoint may import anything
            for imp in imports:
                # own-package imports are always allowed
                if imp == module or imp.startswith(module + "."):
                    continue
                top_pkg = ".".join(module.split(".")[:2])
                if imp == top_pkg or imp.startswith(top_pkg + "."):
                    if not any(imp == f or imp.startswith(f) for f in forbidden if f != "src."):
                        continue
                for f in forbidden:
                    hit = imp.startswith(f) if f.endswith(".") else (imp == f or imp.startswith(f + "."))
                    if hit:
                        if (module, imp) in KNOWN_DEBT:
                            seen_debt.add((module, imp))
                        else:
                            bad.append(f"{module} -> {imp} (forbidden by '{src_prefix}' contract)")
                        break
    return bad, seen_debt


def test_no_new_cross_layer_imports():
    bad, _ = _violations()
    assert not bad, (
        "New cross-layer imports violate the DP-200 module dependency rules:\n  "
        + "\n  ".join(sorted(set(bad)))
        + "\nEither restructure the dependency or (only with good reason) add it to KNOWN_DEBT."
    )


def test_known_debt_list_is_not_stale():
    _, seen = _violations()
    stale = KNOWN_DEBT - seen
    assert not stale, (
        "These grandfathered edges no longer exist — delete them from KNOWN_DEBT "
        "so the list keeps shrinking:\n  "
        + "\n  ".join(f"{m} -> {i}" for m, i in sorted(stale))
    )
