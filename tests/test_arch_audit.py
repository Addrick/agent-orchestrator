"""Tests for the `scripts/arch_audit.py` probes.

The nesting metric drives the hotspot ranking that decides which functions get
refactored, so an over-count sends real work at the wrong file (DP-310).
"""
import ast
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "arch_audit", Path(__file__).resolve().parents[1] / "scripts" / "arch_audit.py"
)
arch_audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(arch_audit)


def _depth_of(src):
    """Nesting depth of the single function defined in `src`."""
    fn = ast.parse(src).body[0]
    return arch_audit._depth(fn)


def test_flat_function_has_no_nesting():
    assert _depth_of("def f():\n    x = 1\n    return x\n") == 0


def test_single_if_is_one_level():
    assert _depth_of("def f(a):\n    if a:\n        return 1\n") == 1


def test_elif_chain_stays_flat():
    """A dispatch chain reads flat and must not score as nesting."""
    src = (
        "def f(a):\n"
        "    if a == 1:\n"
        "        return 'a'\n"
        "    elif a == 2:\n"
        "        return 'b'\n"
        "    elif a == 3:\n"
        "        return 'c'\n"
        "    elif a == 4:\n"
        "        return 'd'\n"
        "    elif a == 5:\n"
        "        return 'e'\n"
    )
    assert _depth_of(src) == 1


def test_else_then_indented_if_is_a_real_level():
    """`else:` + an indented `if` is genuine nesting, unlike `elif`."""
    src = (
        "def f(a, b):\n"
        "    if a:\n"
        "        return 1\n"
        "    else:\n"
        "        if b:\n"
        "            return 2\n"
    )
    assert _depth_of(src) == 2


def test_body_of_an_elif_still_nests():
    """Flattening the chain must not also flatten what is inside a branch."""
    src = (
        "def f(a, b, c):\n"
        "    if a == 1:\n"
        "        return 'a'\n"
        "    elif a == 2:\n"
        "        if b:\n"
        "            for _ in c:\n"
        "                return 'b'\n"
    )
    assert _depth_of(src) == 3


def test_nested_def_depth_does_not_bubble_up():
    """A flat function holding a deep closure is not itself deep.

    The closure gets its own row in the report; counting it twice scored the
    route table for control flow it does not contain.
    """
    src = (
        "def outer(a):\n"
        "    def inner(b):\n"
        "        with b:\n"
        "            for x in b:\n"
        "                while x:\n"
        "                    return x\n"
        "    return inner\n"
    )
    assert _depth_of(src) == 0


def test_enclosing_own_nesting_still_counts():
    src = (
        "def outer(a):\n"
        "    if a:\n"
        "        def inner():\n"
        "            for x in a:\n"
        "                return x\n"
        "    return 1\n"
    )
    assert _depth_of(src) == 1


@pytest.mark.parametrize("keyword", ["for", "while", "with", "try"])
def test_other_constructs_still_count(keyword):
    header = {
        "for": "for _ in a:",
        "while": "while a:",
        "with": "with a:",
        "try": "try:",
    }[keyword]
    footer = "    except Exception:\n        pass\n" if keyword == "try" else ""
    src = f"def f(a):\n    {header}\n        pass\n{footer}"
    assert _depth_of(src) == 1
