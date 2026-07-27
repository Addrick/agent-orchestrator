# scripts/arch_audit.py
"""Maintainability audit (DP-302).

Three read-only probes over src/, each answering a different question about
why the codebase is hard to change:

  deps      package-level runtime import graph, cycles, fan-in
  dupes     rename-insensitive structural duplication between functions
  hotspots  churn x size, plus the long / deeply-nested functions

Runs all three by default:  python scripts/arch_audit.py
Or one at a time:           python scripts/arch_audit.py hotspots

Nothing here enforces anything -- `lint-imports` (setup.cfg) is the gate that
runs in CI. This script is for deciding *what to refactor next*.

Baseline recorded 2026-07-27 at 6475d8d, for comparison on later runs:
  345 runtime deps, 2 package cycles, 1.23% exact clones,
  ~125 structurally-redundant lines, 6 functions over 200 lines.
"""
import ast
import collections
import hashlib
import os
import subprocess
import sys

SKIP_DIRS = {'.venv', '__pycache__', 'node_modules', 'worktrees', '.git',
             'tests', '.pytest_cache'}


def _walk(base):
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith('.py'):
                yield os.path.join(root, f).replace(os.sep, '/')


def _parse(path):
    try:
        return ast.parse(open(path, encoding='utf-8').read())
    except (OSError, SyntaxError):
        return None


# --------------------------------------------------------------------------
# deps
# --------------------------------------------------------------------------

def _packages():
    pkgs = {'config'}
    for e in os.scandir('src'):
        if e.is_dir() and e.name != '__pycache__':
            pkgs.add('src.' + e.name)
        elif e.is_file() and e.name.endswith('.py') and e.name != '__init__.py':
            pkgs.add('src.' + e.name[:-3])
    return sorted(pkgs, key=len, reverse=True)


def _type_checking_imports(tree):
    """Imports guarded by `if TYPE_CHECKING:` -- no runtime coupling."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and 'TYPE_CHECKING' in ast.dump(n.test):
            for sub in ast.walk(n):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    out.add(id(sub))
    return out


def _imported_modules(tree):
    """Yield (module_name, lineno) for every runtime top-level import."""
    skip = _type_checking_imports(tree)
    for n in ast.walk(tree):
        if id(n) in skip:
            continue
        if isinstance(n, ast.Import):
            for alias in n.names:
                yield alias.name, n.lineno
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            yield n.module, n.lineno


def _build_edges():
    pkgs = _packages()

    def owner(mod):
        for p in pkgs:
            if mod == p or mod.startswith(p + '.'):
                return p
        return None

    edges = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in list(_walk('src')) + list(_walk('config')):
        src_pkg = owner(path[:-3].replace('/', '.'))
        tree = _parse(path)
        if not src_pkg or tree is None:
            continue
        for mod, lineno in _imported_modules(tree):
            target = owner(mod)
            if target and target != src_pkg:
                edges[src_pkg][target].append(f"{path}:{lineno}")
    return edges


def _print_cycles(edges, short):
    print("\n=== cycles (each one is a refactor target) ===")
    seen = set()
    for a in edges:
        for b in edges[a]:
            if a in edges.get(b, {}) and (b, a) not in seen:
                seen.add((a, b))
                print(f"{short(a)} <-> {short(b)}")
                for d in edges[a][b][:3] + edges[b][a][:3]:
                    print(f"    {d}")
    if not seen:
        print("(none)")


def deps():
    edges = _build_edges()

    def short(x):
        return x.replace('src.', '')

    total = sum(len(v) for d in edges.values() for v in d.values())
    print(f"=== package dependencies (runtime only, {total} imports) ===")
    for k in sorted(edges):
        print(f"{short(k):24s} -> {', '.join(sorted(short(x) for x in edges[k]))}")

    _print_cycles(edges, short)

    print("\n=== fan-in (packages importing it) ===")
    fan = collections.Counter()
    for a in edges:
        for b in edges[a]:
            fan[b] += 1
    for k, v in fan.most_common(12):
        print(f"{v:3d}  {short(k)}")


# --------------------------------------------------------------------------
# dupes
# --------------------------------------------------------------------------

MIN_NODES = 25


class _Shape(ast.NodeVisitor):
    """Reduces a function body to its control-flow shape.

    Identifiers, attributes and literals collapse to placeholders, so two
    functions that do the same thing with different names hash identically.
    This is the duplication `jscpd` cannot see.
    """

    def __init__(self):
        self.parts = []
        self.n = 0

    def generic_visit(self, node):
        self.n += 1
        self.parts.append(type(node).__name__)
        super().generic_visit(node)

    def visit_Name(self, node):
        self.n += 1
        self.parts.append('N')

    def visit_Attribute(self, node):
        self.n += 1
        self.parts.append('A')
        self.visit(node.value)

    def visit_Constant(self, node):
        self.n += 1
        self.parts.append('C')

    def visit_arg(self, node):
        self.n += 1
        self.parts.append('a')


def dupes():
    groups = collections.defaultdict(list)
    counted = 0
    for path in _walk('src'):
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            shape = _Shape()
            for stmt in node.body:
                shape.visit(stmt)
            if shape.n < MIN_NODES:
                continue
            counted += 1
            key = hashlib.sha1('|'.join(shape.parts).encode()).hexdigest()[:12]
            span = (node.end_lineno or node.lineno) - node.lineno + 1
            groups[key].append((path, node.lineno, node.name, span))

    clones = [v for v in groups.values() if len(v) > 1]
    clones.sort(key=lambda v: -(v[0][3] * (len(v) - 1)))
    redundant = sum(v[0][3] * (len(v) - 1) for v in clones)
    print(f"{counted} functions analyzed (>= {MIN_NODES} AST nodes)")
    print(f"{len(clones)} structural-clone groups, ~{redundant} redundant lines\n")
    for v in clones[:30]:
        span = v[0][3]
        print(f"[{len(v)}x ~{span}L, ~{span * (len(v) - 1)}L redundant]")
        for path, line, name, _ in sorted(v):
            print(f"    {path}:{line}  {name}()")
        print()


# --------------------------------------------------------------------------
# hotspots
# --------------------------------------------------------------------------

NESTING_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With,
                 ast.AsyncFor, ast.AsyncWith)


def _depth(node, current=0):
    best = current
    for child in ast.iter_child_nodes(node):
        nxt = current + 1 if isinstance(child, NESTING_NODES) else current
        best = max(best, _depth(child, nxt))
    return best


def _churn():
    """Commits touching each src file in the last 12 months."""
    log = subprocess.run(
        ['git', 'log', '--since=12.months', '--name-only', '--pretty=format:'],
        capture_output=True, text=True).stdout
    return collections.Counter(
        line.strip() for line in log.splitlines()
        if line.strip().startswith('src/') and line.strip().endswith('.py'))


def _measure():
    """Per-file LOC and deepest nesting, plus every function's span/nesting."""
    loc, deepest, funcs = {}, {}, []
    for path in _walk('src'):
        try:
            text = open(path, encoding='utf-8').read()
        except OSError:
            continue
        loc[path] = text.count('\n') + 1
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        worst = 0
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                span = (n.end_lineno or n.lineno) - n.lineno + 1
                nest = _depth(n)
                funcs.append((span, nest, n.name, n.lineno, path))
                worst = max(worst, nest)
        deepest[path] = worst
    return loc, deepest, funcs


def hotspots():
    churn = _churn()
    loc, deepest, funcs = _measure()

    print("=== hotspots: churn(12mo) x LOC ===")
    print(f"{'score':>8} {'edits':>5} {'LOC':>5} {'nest':>4}  file")
    rows = [(churn.get(p, 0) * loc[p], churn.get(p, 0), loc[p], deepest.get(p, 0), p)
            for p in loc]
    for score, edits, lines, nest, path in sorted(rows, reverse=True)[:20]:
        print(f"{score:8d} {edits:5d} {lines:5d} {nest:4d}  {path}")

    print("\n=== longest functions (>= 80 lines) ===")
    for span, nest, name, line, path in sorted(funcs, reverse=True):
        if span < 80:
            break
        print(f"{span:5d}L  nest={nest}  {path}:{line}  {name}()")

    print("\n=== deepest nesting (>= 5) ===")
    for span, nest, name, line, path in sorted(funcs, key=lambda f: -f[1]):
        if nest < 5:
            break
        print(f"nest={nest}  {span:5d}L  {path}:{line}  {name}()")

    print("\n=== function-size distribution ===")
    buckets = collections.Counter()
    for span, *_ in funcs:
        buckets['1-20' if span <= 20 else
                '21-50' if span <= 50 else
                '51-100' if span <= 100 else
                '101-200' if span <= 200 else '200+'] += 1
    for label in ('1-20', '21-50', '51-100', '101-200', '200+'):
        print(f"  {label:>8}: {buckets[label]}")
    print(f"  {len(funcs)} functions, {sum(loc.values())} LOC")


PROBES = {'deps': deps, 'dupes': dupes, 'hotspots': hotspots}

if __name__ == '__main__':
    chosen = sys.argv[1:] or list(PROBES)
    unknown = [c for c in chosen if c not in PROBES]
    if unknown:
        sys.exit(f"unknown probe(s): {', '.join(unknown)}; "
                 f"choose from {', '.join(PROBES)}")
    for i, name in enumerate(chosen):
        if i:
            print('\n')
        print(f"{'#' * 20} {name} {'#' * 20}")
        PROBES[name]()
